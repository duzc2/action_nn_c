/**
 * @file mlp_train_ops.c
 * @brief MLP training operations implementation
 *
 * Implements complete training pipeline:
 * - Forward propagation and loss computation
 * - Backpropagation with gradient computation
 * - Parameter updates (SGD, Adam)
 * - Checkpoint save/load with hash validation
 * - Dual-mode execution support
 */

#include "mlp_train_ops.h"
#include "mlp_infer_ops.h"
#include "mlp_layers.h"
#include "../../norm/rms_norm.h"
#include "../../dropout/dropout.h"
#include "../../residual/skip_connection.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../../utils/error.h"

#define ABI_VERSION 1
#define ADAM_EPS 1e-8f
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f

/**
 * @section mlp_train_design MLP training data flow
 *
 * This file carries almost the entire hand-written MLP training pipeline. The
 * implementation is intentionally organized in layers so profiler-generated code
 * can reuse the same primitives in both standalone and composed-graph modes:
 * 1. lightweight helpers expose layer geometry and allocate gradient storage;
 * 2. loss helpers convert predictions into output gradients;
 * 3. optimizer helpers consume accumulated gradients and update parameters;
 * 4. forward/backward helpers perform the actual dense-network training step;
 * 5. public entry points wrap the internals into checkpoints, reports, and
 *    graph-compatible APIs.
 *
 * The comments below focus on why each stage exists and which invariants it
 * maintains, because this file is one of the main places where training logic
 * and generated orchestration code meet.
 */

/**
 * @section mlp_train_helpers Low-level helper stage
 *
 * The following helpers avoid duplicating shape logic throughout the training
 * loop. They keep topology lookups, temporary storage allocation, and loss math
 * in small units so the public entry points remain readable.
 */

/**
 * @section mlp_train_grad_storage Gradient and optimizer state ownership
 *
 * Each train step needs transient gradients plus optional persistent optimizer
 * state. The helpers in this section define that ownership model explicitly so
 * context creation, teardown, and checkpoint logic all agree on where velocity
 * or Adam moment buffers live.
 */


/**
 * @brief Get layer configuration
 *
 * @param infer_ctx Inference context
 * @param idx Layer index
 * @param input_size Output input size
 * @param output_size Output output size
 */
static void get_layer_sizes(MlpInferContext* infer_ctx, size_t idx,
                            size_t* input_size, size_t* output_size) {
    if (infer_ctx == NULL || input_size == NULL || output_size == NULL) {
        return;
    }
    *input_size = 0;
    *output_size = 0;

    if (idx >= infer_ctx->layer_count) {
        return;
    }

    MlpDenseLayer* layer = infer_ctx->layers[idx];
    *input_size = layer->input_size;
    *output_size = layer->output_size;
}



/**
 * @section mlp_train_loss_helpers Loss and local-derivative helpers
 *
 * Loss helpers convert user-visible predictions and targets into the exact
 * output gradient consumed by backpropagation. Keeping them separate from the
 * reverse sweep makes it easy for graph-mode training to bypass task loss and
 * inject an externally computed dL/dY directly.
 */

/**
 * @brief Compute softmax derivative for cross-entropy loss
 *
 * For cross-entropy with softmax, gradient simplifies to:
 * d_loss/d_z = prediction - target
 *
 * @param output Layer output (after softmax)
 * @param target Target values
 * @param grad Output gradient
 * @param size Number of elements
 */
static void softmax_cross_entropy_grad(const float* output, const float* target,
                                       float* grad, size_t size) {
    size_t i;
    /* Softmax + cross-entropy collapses to prediction minus target elementwise. */
    for (i = 0; i < size; i++) {
        grad[i] = output[i] - target[i];
    }
}

/**
 * @brief Compute MSE loss
 *
 * MSE = (1/N) * sum((pred - target)^2)
 *
 * @param pred Predictions
 * @param target Targets
 * @param grad Output gradient
 * @param size Number of elements
 * @return Loss value
 */
static float mse_loss(const float* pred, const float* target,
                      float* grad, size_t size) {
    size_t i;
    float loss = 0.0f;

    /* Accumulate both scalar loss and output-layer gradient in one linear pass. */
    for (i = 0; i < size; i++) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
        grad[i] = 2.0f * diff;
    }

    return loss / (float)size;
}

/**
 * @brief Compute cross-entropy loss
 *
 * CE = -sum(target * log(pred))
 *
 * @param pred Predictions (after softmax)
 * @param target Targets
 * @param grad Output gradient
 * @param size Number of elements
 * @return Loss value
 */
static float cross_entropy_loss(const float* pred, const float* target,
                                float* grad, size_t size) {
    size_t i;
    float loss = 0.0f;
    float clip_pred;

    /* Clip probabilities to keep log() finite and gradient generation stable. */
    for (i = 0; i < size; i++) {
        clip_pred = pred[i] > 1e-7f ? pred[i] : 1e-7f;
        clip_pred = clip_pred < 1.0f - 1e-7f ? clip_pred : 1.0f - 1e-7f;
        loss -= target[i] * logf(clip_pred);
    }

    softmax_cross_entropy_grad(pred, target, grad, size);

    return loss;
}

/**
 * @section mlp_train_activation_helpers Reverse local derivative helpers
 *
 * The backward pass needs activation-specific local derivatives after each
 * layer's output gradient has been formed. Applying them in one helper keeps
 * the reverse sweep focused on data flow rather than per-activation formulas.
 */

/**
 * @brief Derivative of activation function
 *
 * @param activation Activation type
 * @param output Output values
 * @param grad Output gradient (modified in place)
 * @param size Number of elements
 */
static void activation_derivative(MlpActivationType activation,
                                   const float* restrict output,
                                   float* restrict grad, size_t size) {
    size_t i;

    /* Gradients are modified in place because the caller already owns dL/dY. */
    switch (activation) {
        case MLP_ACT_RELU:
            for (i = 0; i < size; i++) {
                grad[i] = (output[i] > 0.0f) ? grad[i] : 0.0f;
            }
            break;

        case MLP_ACT_SIGMOID:
            for (i = 0; i < size; i++) {
                float s = output[i];
                grad[i] = s * (1.0f - s) * grad[i];
            }
            break;

        case MLP_ACT_TANH:
            for (i = 0; i < size; i++) {
                float t = output[i];
                grad[i] = (1.0f - t * t) * grad[i];
            }
            break;

        case MLP_ACT_LEAKY_RELU:
            for (i = 0; i < size; i++) {
                float alpha = 0.01f;
                grad[i] = (output[i] > 0.0f) ? grad[i] : alpha * grad[i];
            }
            break;

        case MLP_ACT_SOFTMAX:
        case MLP_ACT_NONE:
        default:
            break;
    }
}

/**
 * @section mlp_train_optimizer_helpers Parameter update rules
 *
 * Once dL/dW and dL/dB have been computed for each layer, the selected
 * optimizer consumes those tensors and mutates the live inference weights.
 * The training pipeline keeps optimizer logic isolated here so changing the
 * update rule does not affect forward/backward math.
 */

/**
 * @brief Optimizer stage: apply SGD with optional momentum buffers.
 *
 * The implementation keeps classical momentum state inside the per-layer
 * gradient object so repeated training steps can reuse the same velocity
 * vectors. Weight decay is injected directly into the weight gradient, while
 * bias updates intentionally skip decay to match common dense-layer practice.
 */
static void update_sgd(float* restrict weights, float* restrict bias,
                       const MlpLayerGrad* restrict grad,
                       float lr, float momentum, float weight_decay,
                       size_t weight_count, size_t bias_count) {
    size_t i;

    /* Velocity buffers carry step-to-step momentum for each parameter tensor. */
    for (i = 0; i < weight_count; i++) {
        grad->velocity_w[i] = momentum * grad->velocity_w[i] -
                              lr * (grad->weight_grad[i] + weight_decay * weights[i]);
        weights[i] += grad->velocity_w[i];
    }

    /* Biases follow the same momentum rule without L2 decay injection. */
    for (i = 0; i < bias_count; i++) {
        grad->velocity_b[i] = momentum * grad->velocity_b[i] -
                              lr * grad->bias_grad[i];
        bias[i] += grad->velocity_b[i];
    }
}

/**
 * @brief Optimizer stage: apply Adam moments and bias correction.
 *
 * Adam keeps first and second moments inside the gradient object so the
 * optimizer state follows the layer wherever generated code stores it. Bias
 * correction uses the effective step count supplied by the caller, which keeps
 * the helper stateless apart from the moment buffers themselves.
 */
static void update_adam(float* restrict weights, float* restrict bias,
                        const MlpLayerGrad* restrict grad,
                        float lr, float weight_decay,
                        size_t t,
                        size_t weight_count, size_t bias_count) {
    size_t i;
    float lr_corr;
    float m_corr_w, v_corr_w;
    float m_corr_b, v_corr_b;
    float beta1_pow_t;
    float beta2_pow_t;

    (void)weight_decay;

    /* Caller supplies the global step so bias correction stays consistent. */
    beta1_pow_t = powf(grad->beta1, (float)t);
    beta2_pow_t = powf(grad->beta2, (float)t);

    lr_corr = lr * sqrtf(1.0f - beta2_pow_t) / (1.0f - beta1_pow_t);

    /* Update dense weights using moment estimates of the same tensor shape. */
    for (i = 0; i < weight_count; i++) {
        grad->m_w[i] = grad->beta1 * grad->m_w[i] +
                       (1.0f - grad->beta1) * grad->weight_grad[i];
        grad->v_w[i] = grad->beta2 * grad->v_w[i] +
                       (1.0f - grad->beta2) * grad->weight_grad[i] * grad->weight_grad[i];

        m_corr_w = grad->m_w[i] / (1.0f - beta1_pow_t);
        v_corr_w = grad->v_w[i] / (1.0f - beta2_pow_t);

        weights[i] -= lr_corr * m_corr_w / (sqrtf(v_corr_w) + ADAM_EPS);
    }

    /* Bias parameters carry their own first/second moments just like weights. */
    for (i = 0; i < bias_count; i++) {
        grad->m_b[i] = grad->beta1 * grad->m_b[i] +
                       (1.0f - grad->beta1) * grad->bias_grad[i];
        grad->v_b[i] = grad->beta2 * grad->v_b[i] +
                       (1.0f - grad->beta2) * grad->bias_grad[i] * grad->bias_grad[i];

        m_corr_b = grad->m_b[i] / (1.0f - beta1_pow_t);
        v_corr_b = grad->v_b[i] / (1.0f - beta2_pow_t);

        bias[i] -= lr_corr * m_corr_b / (sqrtf(v_corr_b) + ADAM_EPS);
    }
}

/**
 * @section mlp_train_execution_helpers Forward, backward, and update pipeline
 *
 * These helpers form the core execution path used by both standalone training
 * and graph-mode training. They keep the three conceptual stages separate:
 * cache activations, propagate gradients, then apply optimizer updates.
 */

/**
 * @brief Forward stage: run the network and convert predictions into loss terms.
 *
 * Training needs more than the final output: every layer input and activation is
 * cached so the backward stage can reconstruct local derivatives without
 * rerunning the network. The helper therefore performs both user-visible work
 * (latest output/loss) and bookkeeping work (activation storage).
 */
static int train_forward_pass(MlpTrainContext* ctx,
                              const float* input) {
    MlpInferContext* infer_ctx;
    MlpDenseLayer* layer;
    const float* current_input;
    size_t i;
    size_t norm_offset;
    float epsilon;

    if (ctx == NULL || ctx->infer_ctx == NULL || input == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;

    if (infer_ctx->layer_count == 0 || infer_ctx->layers == NULL) {
        return ACTION_C_ERR_DIM_MISMATCH;
    }

    /* Ensure dropout layers are in training mode. */
    if (infer_ctx->config->use_dropout && infer_ctx->dropouts != NULL) {
        for (i = 0; i < infer_ctx->layer_count - 1U; ++i) {
            if (infer_ctx->dropouts[i] != NULL) infer_ctx->dropouts[i]->training = 1;
        }
    }

    epsilon = infer_ctx->config->norm_epsilon;
    if (epsilon <= 0.0f) epsilon = 1e-5f;

    /* Cache the original sample because backprop needs it later. */
    memcpy(ctx->input_buffer, input, infer_ctx->config->input_size * sizeof(float));

    /* Materialize each layer activation so the backward pass can reuse it. */
    norm_offset = 0;
    for (i = 0; i < infer_ctx->layer_count; i++) {
        current_input = (i == 0U) ? ctx->input_buffer : ctx->activations[i - 1U];
        layer = infer_ctx->layers[i];
        mlp_dense_forward(layer, ctx->activations[i], current_input);

        if (i < infer_ctx->layer_count - 1U) {
            /* ─── Hidden layer: apply optional norm / dropout / skip ─── */
            int do_norm  = infer_ctx->config->use_rms_norm && infer_ctx->norm_gamma != NULL;
            int do_drop  = infer_ctx->config->use_dropout && infer_ctx->dropouts != NULL && infer_ctx->dropouts[i] != NULL;
            int do_skip  = infer_ctx->config->use_skip && infer_ctx->skips != NULL
                           && infer_ctx->skips[i].mode != SKIP_NONE
                           && layer->output_size == layer->input_size;

            if (do_norm) {
                /* Save pre-norm value (dense+activation output) for backward. */
                memcpy(ctx->norm_input_cache[i], ctx->activations[i],
                       layer->output_size * sizeof(float));
                /* Apply RMSNorm in-place on activations[i]. */
                rms_norm_forward(ctx->activations[i], ctx->activations[i],
                    infer_ctx->norm_gamma + norm_offset,
                    layer->output_size, epsilon);
            }

            if (do_drop) {
                /* Dropout in-place (only in training mode). */
                dropout_forward(ctx->activations[i], ctx->activations[i],
                    layer->output_size, infer_ctx->dropouts[i]);
            }

            if (do_skip) {
                /* Save pre-skip value (norm+dropout output) for backward. */
                memcpy(ctx->skip_module_cache[i], ctx->activations[i],
                       layer->output_size * sizeof(float));
                /* Apply skip in-place on activations[i]. */
                skip_forward(ctx->activations[i], ctx->activations[i],
                    current_input, layer->output_size, &infer_ctx->skips[i]);
            }

            norm_offset += layer->output_size;
        }
    }

    /* Mirror the final activation into the infer context so shared callers see fresh outputs. */
    memcpy(infer_ctx->output_buffer,
           ctx->activations[infer_ctx->layer_count - 1U],
           infer_ctx->config->output_size * sizeof(float));

    return 0;
}

/**
 * @brief Zero all accumulated parameter gradients at the start of a new batch.
 *
 * Only called when batch_step_count == 0 (or batch_size <= 1 for compatibility).
 * Clears weight_grad and bias_grad for every layer so the subsequent backward
 * passes can safely use += to accumulate.
 */
static void mlp_zero_gradients(MlpTrainContext* ctx) {
    size_t i;
    MlpInferContext* infer_ctx;

    if (ctx == NULL || ctx->infer_ctx == NULL) {
        return;
    }
    infer_ctx = (MlpInferContext*)ctx->infer_ctx;

    for (i = 0U; i < infer_ctx->layer_count; ++i) {
        size_t input_size  = (i == 0U) ? infer_ctx->config->input_size  : infer_ctx->layers[i - 1U]->output_size;
        size_t output_size = infer_ctx->layers[i]->output_size;

        if (ctx->grads[i].weight_grad != NULL) {
            (void)memset(ctx->grads[i].weight_grad, 0, input_size * output_size * sizeof(float));
        }
        if (ctx->grads[i].bias_grad != NULL) {
            (void)memset(ctx->grads[i].bias_grad, 0, output_size * sizeof(float));
        }
    }

    /* P0: zero shared-module gradient accumulators at batch start */
    if (ctx->norm_gamma_grads != NULL) {
        (void)memset(ctx->norm_gamma_grads, 0, ctx->total_norm_dim * sizeof(float));
    }
    if (ctx->skip_alpha_raw_grads != NULL) {
        (void)memset(ctx->skip_alpha_raw_grads, 0, ctx->layer_count * sizeof(float));
    }
    /* Also clear accumulated gradients on the SkipConnection objects themselves. */
    if (infer_ctx->skips != NULL) {
        for (i = 0U; i < infer_ctx->layer_count; ++i) {
            infer_ctx->skips[i].grad_alpha_raw = 0.0f;
        }
    }
}

/**
 * @brief Backward stage: propagate gradients through every dense layer.
 *
 * The reverse sweep computes three things in lockstep: activation derivatives,
 * parameter gradients for the current layer, and the input gradient that becomes
 * the next layer's output gradient when moving backward.
 *
 * Gradients are accumulated with += to support mini-batch training
 * (batch_size > 1).  The caller is responsible for zeroing gradients
 * at batch start and applying updates at batch end.
 */
static int train_backward_pass(
    MlpTrainContext* ctx,
    const float* output_gradient,
    float* input_gradient
) {
    MlpInferContext* infer_ctx;
    float* current_delta;
    float* next_delta;
    float* d_fx;          /* P0: temp buffer for skip backward d_module_out */
    float* d_skip_in;     /* P0: temp buffer for skip backward d_input       */
    const float* prev_activation;
    size_t max_size;
    size_t layer_cursor;
    size_t _arena_mark;

    if (ctx == NULL || ctx->infer_ctx == NULL || output_gradient == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;
    max_size = infer_ctx->max_buffer_size;
    /* P0: extra temp buffers for skip backward.
     * Use arena snapshot/restore to stay off the heap inside the training hot path. */
    _arena_mark = arena_snapshot(ctx->arena);
    current_delta = ARENA_CALLOC(ctx->arena, float, max_size);
    next_delta    = ARENA_CALLOC(ctx->arena, float, max_size);
    d_fx          = ARENA_CALLOC(ctx->arena, float, max_size);
    d_skip_in     = ARENA_CALLOC(ctx->arena, float, max_size);
    if (current_delta == NULL || next_delta == NULL || d_fx == NULL || d_skip_in == NULL) {
        arena_restore(ctx->arena, _arena_mark);
        return ACTION_C_ERR_NO_MEMORY;
    }

    /* Seed the reverse sweep with dL/dY from the chosen loss function. */
    memcpy(current_delta, output_gradient, infer_ctx->config->output_size * sizeof(float));

    /* Walk layers from output back to input exactly once. */
    for (layer_cursor = infer_ctx->layer_count; layer_cursor > 0U; --layer_cursor) {
        size_t layer_index = layer_cursor - 1U;
        MlpDenseLayer* layer = infer_ctx->layers[layer_index];
        MlpLayerGrad* grad = &ctx->grads[layer_index];
        size_t input_size = layer->input_size;
        size_t output_size = layer->output_size;
        size_t output_index;
        size_t input_index;
        int is_hidden = (layer_index < infer_ctx->layer_count - 1U);

        /* Reset the previous-layer accumulator before filling it. */
        memset(next_delta, 0, max_size * sizeof(float));

        /* ─── P0: apply skip / dropout / norm backward (reverse of forward) ─── */
        if (is_hidden) {
            const float* skip_module_out = NULL;
            const float* skip_input_ptr  = NULL;
            const float* norm_input_ptr  = NULL;
            size_t norm_off;
            float epsilon;

            epsilon = infer_ctx->config->norm_epsilon;
            if (epsilon <= 0.0f) epsilon = 1e-5f;

            /* ---- skip backward ---- */
            if (infer_ctx->config->use_skip && infer_ctx->skips != NULL
                && infer_ctx->skips[layer_index].mode != SKIP_NONE
                && output_size == input_size) {
                skip_module_out = ctx->skip_module_cache[layer_index];
                skip_input_ptr  = (layer_index == 0U) ? ctx->input_buffer : ctx->activations[layer_index - 1U];

                memset(d_fx, 0, max_size * sizeof(float));
                memset(d_skip_in, 0, max_size * sizeof(float));

                skip_backward(d_fx, d_skip_in, skip_module_out, skip_input_ptr,
                    current_delta, output_size, &infer_ctx->skips[layer_index]);

                /* d_fx replaces current_delta (gradient through module path). */
                memcpy(current_delta, d_fx, output_size * sizeof(float));

                /* d_skip_in is accumulated to next_delta (gradient through skip path). */
                for (output_index = 0U; output_index < output_size; ++output_index) {
                    next_delta[output_index] += d_skip_in[output_index];
                }

                /* Accumulate per-step alpha_raw gradient (batch-level via skip_alpha_raw_grads). */
                ctx->skip_alpha_raw_grads[layer_index] += infer_ctx->skips[layer_index].grad_alpha_raw;
                infer_ctx->skips[layer_index].grad_alpha_raw = 0.0f;
            }

            /* ---- dropout backward ---- */
            if (infer_ctx->config->use_dropout && infer_ctx->dropouts != NULL
                && infer_ctx->dropouts[layer_index] != NULL) {
                dropout_backward(current_delta, current_delta, output_size,
                    infer_ctx->dropouts[layer_index]);
            }

            /* ---- rms_norm backward ---- */
            if (infer_ctx->config->use_rms_norm && infer_ctx->norm_gamma != NULL) {
                /* Compute norm_offset = sum of output_sizes of layers 0..layer_index-1 */
                norm_off = 0;
                {
                    size_t k;
                    for (k = 0; k < layer_index; ++k) {
                        norm_off += infer_ctx->layers[k]->output_size;
                    }
                }
                norm_input_ptr = ctx->norm_input_cache[layer_index];
                if (norm_input_ptr != NULL) {
                    rms_norm_backward(
                        current_delta,
                        infer_ctx->norm_dgamma + norm_off,
                        current_delta,
                        norm_input_ptr,
                        infer_ctx->norm_gamma + norm_off,
                        output_size, epsilon);
                    /* Accumulate d_gamma into the training-level gradient buffer
                     * (norm_gamma_grads uses += across samples). */
                    {
                        size_t j;
                        for (j = 0; j < output_size; ++j) {
                            ctx->norm_gamma_grads[norm_off + j] += infer_ctx->norm_dgamma[norm_off + j];
                        }
                    }
                    /* Also transfer gradient to the skip_backward's expectation.
                     * Clear norm_dgamma so the next layer doesn't pick up stale values. */
                    memset(infer_ctx->norm_dgamma + norm_off, 0, output_size * sizeof(float));
                }
            }
        }

        /* Convert dL/dA into dL/dZ using the stored post-activation outputs.
         *
         * When RMSNorm was applied, the "activation" stored for this layer is
         * the *pre-norm* value.  For skip-backward we use the cached module
         * output; for norm-backward we use the stored pre-norm value.  In both
         * cases the activation_derivative should use the pre-norm activation,
         * which is still the same activations[layer_index] -- because the
         * training forward pass saved the pre-norm value to norm_input_cache
         * before applying norm in-place.
         *
         * When RMSNorm is NOT used but skip IS used, activations[layer_index]
         * was modified in-place by skip_forward.  In that case the derivative
         * needs the pre-skip value instead.  We use skip_module_cache for that.
         */
        if (is_hidden) {
            int do_skip  = infer_ctx->config->use_skip && infer_ctx->skips != NULL
                           && infer_ctx->skips[layer_index].mode != SKIP_NONE
                           && output_size == input_size;
            int do_norm  = infer_ctx->config->use_rms_norm && infer_ctx->norm_gamma != NULL;
            if (do_skip && !do_norm && ctx->skip_module_cache[layer_index] != NULL) {
                /* Use pre-skip cache as the activation for derivative computation. */
                activation_derivative(layer->activation, ctx->skip_module_cache[layer_index],
                    current_delta, output_size);
            } else {
                /* activations[layer_index] holds pre-norm (or pre-skip if no norm). */
                activation_derivative(layer->activation, ctx->activations[layer_index],
                    current_delta, output_size);
            }
        } else {
            activation_derivative(layer->activation, ctx->activations[layer_index],
                current_delta, output_size);
        }

        prev_activation = (layer_index == 0U) ? ctx->input_buffer : ctx->activations[layer_index - 1U];

        /* Compute both parameter gradients and dL/dX for the previous layer.
         * Use += so gradients accumulate across samples in a mini-batch. */
        for (output_index = 0U; output_index < output_size; ++output_index) {
            grad->bias_grad[output_index] += current_delta[output_index];
            for (input_index = 0U; input_index < input_size; ++input_index) {
                grad->weight_grad[output_index * input_size + input_index] +=
                    current_delta[output_index] * prev_activation[input_index];
                next_delta[input_index] +=
                    layer->weights[output_index * input_size + input_index] * current_delta[output_index];
            }
        }

        /* Graph mode may request dL/dX from the first layer for upstream propagation. */
        if (layer_index == 0U && input_gradient != NULL) {
            memcpy(input_gradient, next_delta, input_size * sizeof(float));
        }

        /* The freshly produced input delta becomes the next loop iteration's output delta. */
        memcpy(current_delta, next_delta, input_size * sizeof(float));
    }

    arena_restore(ctx->arena, _arena_mark);
    return 0;
}

/**
 * @brief Update stage: dispatch the accumulated gradients to the selected optimizer.
 *
 * The forward and backward stages deliberately stay optimizer-agnostic. This
 * helper is the single point where the chosen update rule is applied, which
 * keeps SGD/Adam selection out of the math-heavy parts of the training step.
 */
static void train_update(MlpTrainContext* ctx, size_t step) {
    MlpInferContext* infer_ctx;
    MlpDenseLayer* layer;
    size_t i;
    size_t input_size;
    size_t output_size;
    float lr;
    float momentum;

    if (ctx == NULL || ctx->infer_ctx == NULL) {
        return;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;
    lr = ctx->config.learning_rate;
    momentum = ctx->config.momentum;

    /* Apply the chosen optimizer layer by layer in the same order as forward execution. */
    for (i = 0; i < infer_ctx->layer_count; i++) {
        layer = infer_ctx->layers[i];
        get_layer_sizes(infer_ctx, i, &input_size, &output_size);

        if (ctx->config.optimizer == MLP_OPT_SGD) {
            update_sgd(layer->weights, layer->bias, &ctx->grads[i],
                       lr, momentum, ctx->config.weight_decay,
                       input_size * output_size, output_size);
        } else if (ctx->config.optimizer == MLP_OPT_ADAM) {
            update_adam(layer->weights, layer->bias, &ctx->grads[i],
                       lr, ctx->config.weight_decay, step,
                       input_size * output_size, output_size);
        }
    }

    /* P0: update norm_gamma using SGD with momentum */
    if (ctx->norm_gamma_grads != NULL && ctx->norm_gamma_vel != NULL
        && infer_ctx->norm_gamma != NULL) {
        size_t j;
        for (j = 0; j < ctx->total_norm_dim; ++j) {
            ctx->norm_gamma_vel[j] = momentum * ctx->norm_gamma_vel[j]
                                     - lr * ctx->norm_gamma_grads[j];
            infer_ctx->norm_gamma[j] += ctx->norm_gamma_vel[j];
        }
    }

    /* P0: update skip alpha_raw using SGD with momentum */
    if (ctx->skip_alpha_raw_grads != NULL && ctx->skip_alpha_raw_vel != NULL
        && infer_ctx->skips != NULL) {
        for (i = 0; i < infer_ctx->layer_count; ++i) {
            ctx->skip_alpha_raw_vel[i] = momentum * ctx->skip_alpha_raw_vel[i]
                                         - lr * ctx->skip_alpha_raw_grads[i];
            infer_ctx->skips[i].alpha_raw += ctx->skip_alpha_raw_vel[i];
        }
    }
}

/**
 * @brief Public API stage: create the training context that wraps inference state.
 *
 * The training context borrows the live inference network and adds everything
 * needed for learning around it: activation caches, gradient tensors, optimizer
 * state, and rolling statistics.
 */
MlpTrainContext* nn_mlp_train_create(void* infer_ctx, const MlpTrainConfig* config) {
    MlpTrainContext* ctx;
    MlpInferContext* mlp_ctx;
    size_t i;
    size_t j;
    size_t input_size;
    size_t output_size;
    size_t max_size;

    if (infer_ctx == NULL) {
        return NULL;
    }

    mlp_ctx = (MlpInferContext*)infer_ctx;

    ctx = (MlpTrainContext*)calloc(1, sizeof(MlpTrainContext));
    if (ctx == NULL) {
        return NULL;
    }

    /* Copy user configuration once so later training steps can stay allocation free. */
    if (config != NULL) {
        ctx->config = *config;
    } else {
        free(ctx);
        return NULL;
    }

    ctx->infer_ctx = infer_ctx;
    ctx->layer_count = mlp_ctx->layer_count;

    max_size = mlp_ctx->config->input_size;
    if (mlp_ctx->config->output_size > max_size) {
        max_size = mlp_ctx->config->output_size;
    }

    /* Compute total norm dimension for P0 gamma arrays. */
    {
        size_t k;
        ctx->total_norm_dim = 0;
        for (k = 0; k < ctx->layer_count; ++k) {
            ctx->total_norm_dim += mlp_ctx->layers[k]->output_size;
        }
    }

    /* Allocate all top-level arrays up front so creation fails early and cleanly. */
    ctx->grads = (MlpLayerGrad*)calloc(ctx->layer_count, sizeof(MlpLayerGrad));
    ctx->activations = (float**)calloc(ctx->layer_count, sizeof(float*));
    ctx->input_buffer = (float*)malloc(max_size * sizeof(float));
    ctx->target_buffer = (float*)malloc(max_size * sizeof(float));
    ctx->loss_buffer = (float*)malloc(max_size * sizeof(float));

    if (ctx->grads == NULL || ctx->activations == NULL || ctx->input_buffer == NULL ||
        ctx->target_buffer == NULL || ctx->loss_buffer == NULL) {
        free(ctx->grads);
        free(ctx->activations);
        free(ctx->input_buffer);
        free(ctx->target_buffer);
        free(ctx->loss_buffer);
        free(ctx);
        return NULL;
    }

    /* P0: allocate shared-module training buffers. */
    ctx->norm_gamma_grads     = NULL;
    ctx->norm_gamma_vel       = NULL;
    ctx->skip_alpha_raw_grads = NULL;
    ctx->skip_alpha_raw_vel   = NULL;
    ctx->norm_input_cache     = NULL;
    ctx->skip_module_cache    = NULL;

    if (mlp_ctx->config->use_rms_norm) {
        ctx->norm_gamma_grads = (float*)calloc(ctx->total_norm_dim, sizeof(float));
        ctx->norm_gamma_vel   = (float*)calloc(ctx->total_norm_dim, sizeof(float));
        if (ctx->norm_gamma_grads == NULL || ctx->norm_gamma_vel == NULL) {
            free(ctx->norm_gamma_grads); free(ctx->norm_gamma_vel);
            free(ctx->grads); free(ctx->activations);
            free(ctx->input_buffer); free(ctx->target_buffer); free(ctx->loss_buffer);
            free(ctx);
            return NULL;
        }
        /* norm_input_cache: per-layer pointer array */
        ctx->norm_input_cache = (float**)calloc(ctx->layer_count, sizeof(float*));
        if (ctx->norm_input_cache == NULL) {
            free(ctx->norm_gamma_grads); free(ctx->norm_gamma_vel);
            free(ctx->grads); free(ctx->activations);
            free(ctx->input_buffer); free(ctx->target_buffer); free(ctx->loss_buffer);
            free(ctx);
            return NULL;
        }
        for (i = 0; i < ctx->layer_count; ++i) {
            ctx->norm_input_cache[i] = (float*)calloc(mlp_ctx->layers[i]->output_size, sizeof(float));
            if (ctx->norm_input_cache[i] == NULL) {
                for (j = 0; j < i; ++j) free(ctx->norm_input_cache[j]);
                free(ctx->norm_input_cache);
                free(ctx->norm_gamma_grads); free(ctx->norm_gamma_vel);
                free(ctx->grads); free(ctx->activations);
                free(ctx->input_buffer); free(ctx->target_buffer); free(ctx->loss_buffer);
                free(ctx);
                return NULL;
            }
        }
    }

    if (mlp_ctx->config->use_skip) {
        ctx->skip_alpha_raw_grads = (float*)calloc(ctx->layer_count, sizeof(float));
        ctx->skip_alpha_raw_vel   = (float*)calloc(ctx->layer_count, sizeof(float));
        if (ctx->skip_alpha_raw_grads == NULL || ctx->skip_alpha_raw_vel == NULL) {
            free(ctx->skip_alpha_raw_grads); free(ctx->skip_alpha_raw_vel);
            free(ctx->norm_input_cache); /* details depend on cleanup flow below */
            free(ctx->norm_gamma_grads); free(ctx->norm_gamma_vel);
            free(ctx->grads); free(ctx->activations);
            free(ctx->input_buffer); free(ctx->target_buffer); free(ctx->loss_buffer);
            free(ctx);
            return NULL;
        }
        /* skip_module_cache: per-layer pointer array */
        ctx->skip_module_cache = (float**)calloc(ctx->layer_count, sizeof(float*));
        if (ctx->skip_module_cache == NULL) {
            free(ctx->skip_alpha_raw_grads); free(ctx->skip_alpha_raw_vel);
            if (ctx->norm_input_cache) {
                for (j = 0; j < ctx->layer_count; ++j) free(ctx->norm_input_cache[j]);
                free(ctx->norm_input_cache);
            }
            free(ctx->norm_gamma_grads); free(ctx->norm_gamma_vel);
            free(ctx->grads); free(ctx->activations);
            free(ctx->input_buffer); free(ctx->target_buffer); free(ctx->loss_buffer);
            free(ctx);
            return NULL;
        }
        for (i = 0; i < ctx->layer_count; ++i) {
            ctx->skip_module_cache[i] = (float*)calloc(mlp_ctx->layers[i]->output_size, sizeof(float));
            if (ctx->skip_module_cache[i] == NULL) {
                for (j = 0; j < i; ++j) free(ctx->skip_module_cache[j]);
                free(ctx->skip_module_cache);
                free(ctx->skip_alpha_raw_grads); free(ctx->skip_alpha_raw_vel);
                if (ctx->norm_input_cache) {
                    for (j = 0; j < ctx->layer_count; ++j) free(ctx->norm_input_cache[j]);
                    free(ctx->norm_input_cache);
                }
                free(ctx->norm_gamma_grads); free(ctx->norm_gamma_vel);
                free(ctx->grads); free(ctx->activations);
                free(ctx->input_buffer); free(ctx->target_buffer); free(ctx->loss_buffer);
                free(ctx);
                return NULL;
            }
        }
    }

    /* Each layer receives activation storage, gradients, and optimizer state. */
    for (i = 0; i < ctx->layer_count; i++) {
        get_layer_sizes(mlp_ctx, i, &input_size, &output_size);
        ctx->activations[i] = (float*)calloc(output_size, sizeof(float));
        ctx->grads[i].weight_grad = (float*)calloc(input_size * output_size, sizeof(float));
        ctx->grads[i].bias_grad = (float*)calloc(output_size, sizeof(float));

        if (ctx->activations[i] == NULL ||
            ctx->grads[i].weight_grad == NULL || ctx->grads[i].bias_grad == NULL) {
            for (j = 0; j < i; j++) {
                free(ctx->activations[j]);
                free(ctx->grads[j].weight_grad);
                free(ctx->grads[j].bias_grad);
            }
            free(ctx->grads);
            free(ctx->activations);
            free(ctx->input_buffer);
            free(ctx->target_buffer);
            free(ctx->loss_buffer);
            free(ctx);
            return NULL;
        }

        /* Optimizer-specific state is allocated only for the configured algorithm. */
        if (ctx->config.optimizer == MLP_OPT_SGD) {
            ctx->grads[i].velocity_w = (float*)calloc(input_size * output_size, sizeof(float));
            ctx->grads[i].velocity_b = (float*)calloc(output_size, sizeof(float));

            if (ctx->grads[i].velocity_w == NULL || ctx->grads[i].velocity_b == NULL) {
                for (j = 0; j <= i; j++) {
                    free(ctx->activations[j]);
                    free(ctx->grads[j].weight_grad);
                    free(ctx->grads[j].bias_grad);
                    free(ctx->grads[j].velocity_w);
                    free(ctx->grads[j].velocity_b);
                }
                free(ctx->grads);
                free(ctx->activations);
                free(ctx->input_buffer);
                free(ctx->target_buffer);
                free(ctx->loss_buffer);
                free(ctx);
                return NULL;
            }
        } else if (ctx->config.optimizer == MLP_OPT_ADAM) {
            ctx->grads[i].beta1 = ADAM_BETA1;
            ctx->grads[i].beta2 = ADAM_BETA2;
            ctx->grads[i].m_w = (float*)calloc(input_size * output_size, sizeof(float));
            ctx->grads[i].m_b = (float*)calloc(output_size, sizeof(float));
            ctx->grads[i].v_w = (float*)calloc(input_size * output_size, sizeof(float));
            ctx->grads[i].v_b = (float*)calloc(output_size, sizeof(float));

            if (ctx->grads[i].m_w == NULL || ctx->grads[i].m_b == NULL ||
                ctx->grads[i].v_w == NULL || ctx->grads[i].v_b == NULL) {
                for (j = 0; j <= i; j++) {
                    free(ctx->activations[j]);
                    free(ctx->grads[j].weight_grad);
                    free(ctx->grads[j].bias_grad);
                    free(ctx->grads[j].m_w);
                    free(ctx->grads[j].m_b);
                    free(ctx->grads[j].v_w);
                    free(ctx->grads[j].v_b);
                }
                free(ctx->grads);
                free(ctx->activations);
                free(ctx->input_buffer);
                free(ctx->target_buffer);
                free(ctx->loss_buffer);
                free(ctx);
                return NULL;
            }
        }
    }

    /* Create scratch arena for hot-path temporary allocations (backward_pass deltas).
     * 8x max_buffer_size covers: current_delta + next_delta + d_fx + d_skip_in
     * (P0 skip backward) + dummy input/target. */
    ctx->arena = arena_create(mlp_ctx->max_buffer_size * sizeof(float) * 8);
    if (ctx->arena == NULL) {
        nn_mlp_train_destroy(ctx);
        return NULL;
    }

    return ctx;
}

/**
 * @brief Release the training context and every auxiliary buffer it owns.
 *
 * Destruction must mirror creation carefully because parts of the state are
 * common to all optimizers while other parts are algorithm-specific.
 */
void nn_mlp_train_destroy(MlpTrainContext* ctx) {
    size_t i;

    if (ctx == NULL) {
        return;
    }

    /* Release the scratch arena first; it has no dependencies on the other allocations. */
    arena_destroy(ctx->arena);

    /* Free per-layer optimizer state before dropping shared arrays. */
    for (i = 0; i < ctx->layer_count; i++) {
        if (ctx->config.optimizer == MLP_OPT_SGD) {
            free(ctx->grads[i].velocity_w);
            free(ctx->grads[i].velocity_b);
        } else if (ctx->config.optimizer == MLP_OPT_ADAM) {
            free(ctx->grads[i].m_w);
            free(ctx->grads[i].m_b);
            free(ctx->grads[i].v_w);
            free(ctx->grads[i].v_b);
        }
        free(ctx->activations[i]);
        free(ctx->grads[i].weight_grad);
        free(ctx->grads[i].bias_grad);
    }

    /* P0: free shared-module training buffers */
    free(ctx->norm_gamma_grads);
    free(ctx->norm_gamma_vel);
    free(ctx->skip_alpha_raw_grads);
    free(ctx->skip_alpha_raw_vel);
    if (ctx->norm_input_cache != NULL) {
        for (i = 0; i < ctx->layer_count; ++i) free(ctx->norm_input_cache[i]);
        free(ctx->norm_input_cache);
    }
    if (ctx->skip_module_cache != NULL) {
        for (i = 0; i < ctx->layer_count; ++i) free(ctx->skip_module_cache[i]);
        free(ctx->skip_module_cache);
    }

    free(ctx->grads);
    free(ctx->activations);
    free(ctx->input_buffer);
    free(ctx->target_buffer);
    free(ctx->loss_buffer);
    free(ctx);
}

/**
 * @brief Run one full supervised update using raw input and target buffers.
 *
 * This is the canonical path used by standalone training. It executes the full
 * forward-loss-backward-update cycle and records the resulting loss history.
 */
int nn_mlp_train_step_with_data(MlpTrainContext* ctx, const float* input, const float* target) {
    float loss;
    int rc;
    MlpInferContext* infer_ctx;
    size_t i;
    const size_t batch_sz = ctx->config.batch_size;

    if (ctx == NULL || input == NULL || target == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;

    /* ── Mini-batch gradient accumulation ── */
    if (batch_sz <= 1U || ctx->batch_step_count == 0U) {
        mlp_zero_gradients(ctx);
    }

    /* Stage 1: refresh activations and mirrored inference outputs. */
    rc = train_forward_pass(ctx, input);
    if (rc != 0) {
        return rc;
    }

    /* Stage 2: compute loss and seed the output-layer gradient buffer. */
    memcpy(ctx->target_buffer, target, infer_ctx->config->output_size * sizeof(float));
    if (ctx->config.loss_func == MLP_LOSS_MSE) {
        loss = mse_loss(infer_ctx->output_buffer, target, ctx->loss_buffer,
                        infer_ctx->config->output_size);
    } else {
        loss = cross_entropy_loss(infer_ctx->output_buffer, target, ctx->loss_buffer,
                                  infer_ctx->config->output_size);
    }

    /* Stage 3: backpropagate dL/dY through the full network. */
    rc = train_backward_pass(ctx, ctx->loss_buffer, NULL);
    if (rc != 0) {
        return rc;
    }

    /* Stage 4: update parameters and training statistics (conditional on batch). */
    ctx->total_steps++;
    ctx->batch_step_count++;
    if (ctx->batch_step_count >= batch_sz) {
        ctx->batch_step_count = 0U;
        train_update(ctx, ctx->total_steps);
    }

    ctx->last_loss = loss;
    ctx->cumulative_loss += loss;
    ctx->average_loss = ctx->cumulative_loss / (float)ctx->total_steps;

    /* Keep the borrowed infer context synchronized with the latest training pass. */
    if (infer_ctx != NULL) {
        for (i = 0; i < infer_ctx->config->output_size; i++) {
            infer_ctx->output_buffer[i] = ctx->activations[infer_ctx->layer_count - 1U][i];
        }
    }

    return 0;
}

/**
 * @brief Extended single-step API that returns loss and optional predictions.
 *
 * This wrapper exposes richer per-step reporting without duplicating the core
 * forward/backward/update choreography used by the simpler entry point.
 */
int nn_mlp_train_step_ex(MlpTrainContext* ctx,
                          const MlpTrainStepInput* step_in,
                          MlpTrainStepOutput* step_out) {
    MlpInferContext* infer_ctx;
    float loss;
    int rc;
    size_t i;

    if (ctx == NULL || step_in == NULL || step_out == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    if (step_in->input == NULL || step_in->target == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;

    /* Run the same four-stage pipeline as the plain step API. */
    rc = train_forward_pass(ctx, step_in->input);
    if (rc != 0) {
        return rc;
    }

    if (ctx->config.loss_func == MLP_LOSS_MSE) {
        loss = mse_loss(infer_ctx->output_buffer, step_in->target, ctx->loss_buffer,
                        infer_ctx->config->output_size);
    } else {
        loss = cross_entropy_loss(infer_ctx->output_buffer, step_in->target, ctx->loss_buffer,
                                  infer_ctx->config->output_size);
    }

    rc = train_backward_pass(ctx, ctx->loss_buffer, NULL);
    if (rc != 0) {
        return rc;
    }

    ctx->total_steps++;
    train_update(ctx, ctx->total_steps);

    step_out->loss = loss;
    ctx->last_loss = loss;
    ctx->cumulative_loss += loss;
    ctx->average_loss = ctx->cumulative_loss / (float)ctx->total_steps;

    /* Prediction export is optional so callers can avoid copies when they only need loss. */
    if (step_out->predictions != NULL && infer_ctx != NULL) {
        for (i = 0; i < infer_ctx->config->output_size; i++) {
            step_out->predictions[i] = infer_ctx->output_buffer[i];
        }
    }

    return 0;
}

/**
 * @brief Run one graph-mode training step with an externally supplied gradient.
 *
 * Composite graphs use this path when the MLP is one leaf among many. The leaf
 * does not compute its own task loss in that case; it simply consumes dL/dY and
 * optionally returns dL/dX for the previous graph edge.
 */
int nn_mlp_train_step_with_output_gradient(
    MlpTrainContext* ctx,
    const float* input,
    const float* output_gradient,
    float* input_gradient
) {
    int rc;
    const size_t batch_sz = ctx->config.batch_size;

    if (ctx == NULL || input == NULL || output_gradient == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    /* ── Mini-batch gradient accumulation ──
     * When batch_size > 1: zero gradients at batch start, accumulate over
     * multiple samples, apply update only at batch end. */
    if (batch_sz <= 1U || ctx->batch_step_count == 0U) {
        mlp_zero_gradients(ctx);
    }

    /* Graph mode still needs a fresh forward pass to populate activations. */
    rc = train_forward_pass(ctx, input);
    if (rc != 0) {
        return rc;
    }

    /* Backprop starts from caller-supplied dL/dY instead of a locally computed loss. */
    rc = train_backward_pass(ctx, output_gradient, input_gradient);
    if (rc != 0) {
        return rc;
    }

    ctx->total_steps++;
    ctx->batch_step_count++;

    /* Apply update when the mini-batch is complete. */
    if (ctx->batch_step_count >= batch_sz) {
        ctx->batch_step_count = 0U;
        train_update(ctx, ctx->total_steps);
    }

    return 0;
}

/**
 * @brief Public API stage: run a caller-driven multi-epoch training loop.
 *
 * The loop intentionally reuses the single-step entry point rather than open-
 * coding optimizer logic a second time. That keeps loss accounting, gradient
 * clearing, and parameter updates consistent across batch and single-sample use.
 */
int nn_mlp_train_run_auto(MlpTrainContext* ctx, size_t epochs,
                           const float* input, const float* target,
                           size_t sample_count) {
    MlpInferContext* infer_ctx;
    size_t e;
    size_t s;
    int rc;

    if (ctx == NULL || input == NULL || target == NULL || sample_count == 0) {
        return ACTION_C_ERR_DIM_MISMATCH;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;

    /* Reuse the single-step path so optimizer behaviour stays centralized. */
    for (e = 0; e < epochs; e++) {
        for (s = 0; s < sample_count; ++s) {
            size_t offset_in = s * infer_ctx->config->input_size;
            size_t offset_out = s * infer_ctx->config->output_size;
            rc = nn_mlp_train_step_with_data(ctx, input + offset_in, target + offset_out);

            if (rc != 0) {
                return rc;
            }
        }

        /* Epoch accounting is advanced only after the full sample sweep succeeds. */
        ctx->total_epochs++;
    }

    return 0;
}

/**
 * @brief Compute loss only, without mutating parameters or optimizer state.
 *
 * Callers use this helper for reporting and checkpoint selection when they want
 * the configured loss semantics without running another training update.
 */
float nn_mlp_train_compute_loss(MlpTrainContext* ctx,
                                 const float* predictions,
                                 const float* targets,
                                 size_t count) {
    float loss;
    size_t index;

    if (ctx == NULL || predictions == NULL || targets == NULL || count == 0U) {
        return 0.0f;
    }

    /* Use the same configured loss policy as the training step APIs. */
    if (ctx->config.loss_func == MLP_LOSS_MSE) {
        loss = 0.0f;
        for (index = 0U; index < count; ++index) {
            float diff = predictions[index] - targets[index];
            loss += diff * diff;
        }
        loss /= (float)count;
    } else {
        loss = 0.0f;
        for (index = 0U; index < count; ++index) {
            float clip_pred = predictions[index] > 1e-7f ? predictions[index] : 1e-7f;
            clip_pred = clip_pred < 1.0f - 1e-7f ? clip_pred : 1.0f - 1e-7f;
            loss -= targets[index] * logf(clip_pred);
        }
    }

    return loss;
}

/**
 * @brief Checkpoint stage: serialize parameters plus optimizer and progress data.
 *
 * The checkpoint header captures compatibility metadata first so loaders can
 * reject stale files before mutating any in-memory tensor.
 */
int nn_mlp_train_save_checkpoint(MlpTrainContext* ctx, FILE* fp, float best_loss) {
    MlpInferContext* infer_ctx;
    MlpCheckpointHeader header;
    size_t i;
    int rc;

    if (ctx == NULL || ctx->infer_ctx == NULL || fp == NULL) {
        return 0;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;

    header.network_hash = nn_mlp_get_network_hash(infer_ctx);
    header.layout_hash = nn_mlp_get_network_hash(infer_ctx);
    header.abi_version = ABI_VERSION;
    header.epoch = (uint32_t)ctx->total_epochs;
    header.step = (uint32_t)ctx->total_steps;
    header.best_loss = best_loss;

    /* Header comes first because hash/layout validation must precede tensor reads. */
    rc = (int)fwrite(&header, sizeof(header), 1, fp);
    if (rc != 1) return 0;

    /* Parameter payload is written layer-major to match the loader exactly. */
    for (i = 0; i < infer_ctx->layer_count; i++) {
        MlpDenseLayer* layer = infer_ctx->layers[i];

        rc = (int)fwrite(layer->weights,
                         sizeof(float),
                         layer->input_size * layer->output_size,
                         fp);
        if ((size_t)rc != layer->input_size * layer->output_size) {
            return 0;
        }

        rc = (int)fwrite(layer->bias, sizeof(float), layer->output_size, fp);
        if ((size_t)rc != layer->output_size) {
            return 0;
        }
    }

    /* P0: save norm_gamma if present */
    if (infer_ctx->config->use_rms_norm && infer_ctx->norm_gamma != NULL) {
        size_t total_norm = 0;
        for (i = 0; i < infer_ctx->layer_count; ++i) total_norm += infer_ctx->layers[i]->output_size;
        rc = (int)fwrite(infer_ctx->norm_gamma, sizeof(float), total_norm, fp);
        if ((size_t)rc != total_norm) {
            return 0;
        }
    }

    return 1;
}

/**
 * @brief Checkpoint stage: restore parameters only after hash/layout validation.
 *
 * Loading is intentionally conservative: nothing is written into the live model
 * until the checkpoint header proves that network and layout hashes match.
 */
int nn_mlp_train_load_checkpoint(MlpTrainContext* ctx, FILE* fp,
                                  uint64_t current_hash,
                                  uint64_t current_layout_hash) {
    MlpInferContext* infer_ctx;
    MlpCheckpointHeader header;
    size_t i;
    int rc;

    if (ctx == NULL || ctx->infer_ctx == NULL || fp == NULL) {
        return 0;
    }

    /* Validate checkpoint compatibility before touching any layer tensors. */
    rc = (int)fread(&header, sizeof(header), 1, fp);
    if (rc != 1) return 0;

    if (header.network_hash != current_hash ||
        header.layout_hash != current_layout_hash) {
        return 0;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;

    /* Tensor order mirrors the save path so restore can remain a simple stream read. */
    for (i = 0; i < infer_ctx->layer_count; i++) {
        MlpDenseLayer* layer = infer_ctx->layers[i];

        rc = (int)fread(layer->weights,
                        sizeof(float),
                        layer->input_size * layer->output_size,
                        fp);
        if ((size_t)rc != layer->input_size * layer->output_size) {
            return 0;
        }

        rc = (int)fread(layer->bias, sizeof(float), layer->output_size, fp);
        if ((size_t)rc != layer->output_size) {
            return 0;
        }
    }

    /* P0: load norm_gamma if present */
    if (infer_ctx->config->use_rms_norm && infer_ctx->norm_gamma != NULL) {
        size_t total_norm = 0;
        for (i = 0; i < infer_ctx->layer_count; ++i) total_norm += infer_ctx->layers[i]->output_size;
        rc = (int)fread(infer_ctx->norm_gamma, sizeof(float), total_norm, fp);
        if ((size_t)rc != total_norm) {
            return 0;
        }
    }

    /* Progress counters are restored only after every tensor read succeeds. */
    ctx->total_epochs = header.epoch;
    ctx->total_steps = header.step;
    ctx->checkpoint_network_hash = header.network_hash;
    ctx->checkpoint_layout_hash = header.layout_hash;

    return 1;
}

/**
 * @brief Checkpoint stage: inspect a checkpoint header without mutating runtime state.
 *
 * Tooling can call this helper when it needs a compatibility probe but does not
 * want to allocate or modify a live training context.
 */
int nn_mlp_train_validate_checkpoint(FILE* fp,
                                      uint64_t current_hash,
                                      uint64_t current_layout_hash) {
    MlpCheckpointHeader header;
    int rc;

    if (fp == NULL) {
        return 0;
    }

    /* Validation intentionally stops after the header because no payload read is required. */
    rc = (int)fread(&header, sizeof(header), 1, fp);
    if (rc != 1) return 0;

    if (header.network_hash != current_hash ||
        header.layout_hash != current_layout_hash) {
        return 0;
    }

    return 1;
}

/**
 * @brief Report coarse training statistics accumulated in the context.
 *
 * The rolling average is intentionally based on the loss-history ring buffer so
 * callers get a recent trend instead of an average across the full lifetime.
 */
void nn_mlp_train_get_stats(MlpTrainContext* ctx,
                             size_t* out_epochs,
                             size_t* out_steps,
                             float* out_avg_loss) {
    if (ctx == NULL) {
        return;
    }

    if (out_epochs != NULL) {
        *out_epochs = ctx->total_epochs;
    }

    if (out_steps != NULL) {
        *out_steps = ctx->total_steps;
    }

    if (out_avg_loss != NULL) {
        *out_avg_loss = ctx->average_loss;
    }
}

/**
 * @brief Compatibility wrapper that exposes the minimal single-step training hook.
 *
 * The registry-facing contract uses a generic void* signature and does not pass
 * sample buffers directly. This wrapper therefore manufactures a zero-valued
 * sample of bounded size and forwards into the richer data-driven entry point.
 */
int nn_mlp_train_step(void* context) {
    MlpTrainContext* ctx;
    MlpInferContext* infer_ctx;
    float* dummy_input;
    float* dummy_target;
    size_t _arena_mark;
    int rc;

    if (context == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    ctx = (MlpTrainContext*)context;
    infer_ctx = (MlpInferContext*)ctx->infer_ctx;
    if (infer_ctx == NULL || infer_ctx->config == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    /* Arena-allocated zero buffers for the registry-compat zero-sample. */
    _arena_mark = arena_snapshot(ctx->arena);
    dummy_input = ARENA_CALLOC(ctx->arena, float, infer_ctx->config->input_size);
    dummy_target = ARENA_CALLOC(ctx->arena, float, infer_ctx->config->output_size);
    if (dummy_input == NULL || dummy_target == NULL) {
        arena_restore(ctx->arena, _arena_mark);
        return ACTION_C_ERR_NO_MEMORY;
    }

    rc = nn_mlp_train_step_with_data(ctx, dummy_input, dummy_target);
    arena_restore(ctx->arena, _arena_mark);
    return rc;
}

/* ─── VTable backend ─── */

#include "../../nn_backend.h"
#include "mlp_infer_ops.h"

static void* mlp_train_create_vtable(const void* config_blob, size_t config_size,
                                      const void* infer_config_blob, size_t infer_config_size,
                                      struct Arena* arena) {
    MlpInferContext* infer_ctx;
    MlpTrainConfig train_cfg;

    (void)arena;

    /* Decode training config blob. */
    if (config_blob == NULL || config_size < sizeof(MlpTrainConfig)) {
        return NULL;
    }
    train_cfg = *(const MlpTrainConfig*)config_blob;

    /* Create inference context. */
    infer_ctx = nn_mlp_infer_create_with_config_blob(infer_config_blob, infer_config_size, train_cfg.seed);
    if (infer_ctx == NULL) {
        return NULL;
    }

    return nn_mlp_train_create(infer_ctx, &train_cfg);
}

static int mlp_train_step_vtable(void* context, const float* input, const float* target) {
    MlpTrainContext* ctx = (MlpTrainContext*)context;
    if (ctx == NULL || input == NULL || target == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }
    return nn_mlp_train_step_with_data(ctx, input, target);
}

static int mlp_train_step_with_data_vtable(void* context, const float* input,
                                            const float* target, float* out_grad) {
    MlpTrainContext* ctx = (MlpTrainContext*)context;
    (void)out_grad;
    if (ctx == NULL || input == NULL || target == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }
    return nn_mlp_train_step_with_data(ctx, input, target);
}

static int mlp_train_save_checkpoint_vtable(void* context, FILE* fp) {
    MlpTrainContext* ctx = (MlpTrainContext*)context;
    if (ctx == NULL || fp == NULL) return -1;
    return nn_mlp_train_save_checkpoint(ctx, fp, ctx->last_loss) ? 0 : -1;
}

static int mlp_train_load_checkpoint_vtable(void* context, FILE* fp) {
    MlpTrainContext* ctx = (MlpTrainContext*)context;
    if (ctx == NULL || fp == NULL) return -1;
    return nn_mlp_train_load_checkpoint(ctx, fp,
        ctx->checkpoint_network_hash,
        ctx->checkpoint_layout_hash) ? 0 : -1;
}

const NNTrainBackend g_mlp_train_backend = {
    .type_name        = "mlp",
    .create           = mlp_train_create_vtable,
    .destroy          = (void (*)(void*))nn_mlp_train_destroy,
    .step             = mlp_train_step_vtable,
    .step_with_data   = mlp_train_step_with_data_vtable,
    .save_checkpoint  = mlp_train_save_checkpoint_vtable,
    .load_checkpoint  = mlp_train_load_checkpoint_vtable,
};

