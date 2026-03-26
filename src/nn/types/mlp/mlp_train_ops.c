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

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ABI_VERSION 1
#define ADAM_EPS 1e-8f
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f

/**
 * @brief Get inference layer at index
 *
 * @param infer_ctx Inference context
 * @param idx Layer index
 * @return Layer pointer or NULL
 */
static MlpDenseLayer* get_layer(MlpInferContext* infer_ctx, size_t idx) {
    if (infer_ctx == NULL || idx >= infer_ctx->layer_count) {
        return NULL;
    }
    return infer_ctx->layers[idx];
}

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
 * @brief Allocate gradient buffer for one layer
 *
 * @param input_size Number of input features
 * @param output_size Number of output features
 * @param optimizer Optimizer type
 * @return Gradient buffer or NULL
 */
static MlpLayerGrad* alloc_gradients(size_t input_size, size_t output_size,
                                     MlpOptimizerType optimizer) {
    MlpLayerGrad* grad;
    size_t weight_count;
    size_t bias_count;

    if (input_size == 0 || output_size == 0) {
        return NULL;
    }

    grad = (MlpLayerGrad*)calloc(1, sizeof(MlpLayerGrad));
    if (grad == NULL) {
        return NULL;
    }

    weight_count = input_size * output_size;
    bias_count = output_size;

    grad->weight_grad = (float*)calloc(weight_count, sizeof(float));
    grad->bias_grad = (float*)calloc(bias_count, sizeof(float));

    if (grad->weight_grad == NULL || grad->bias_grad == NULL) {
        free(grad->weight_grad);
        free(grad->bias_grad);
        free(grad);
        return NULL;
    }

    if (optimizer == MLP_OPT_SGD) {
        grad->velocity_w = (float*)calloc(weight_count, sizeof(float));
        grad->velocity_b = (float*)calloc(bias_count, sizeof(float));

        if (grad->velocity_w == NULL || grad->velocity_b == NULL) {
            free(grad->velocity_w);
            free(grad->velocity_b);
            free(grad->weight_grad);
            free(grad->bias_grad);
            free(grad);
            return NULL;
        }
    } else if (optimizer == MLP_OPT_ADAM) {
        grad->beta1 = ADAM_BETA1;
        grad->beta2 = ADAM_BETA2;
        grad->m_w = (float*)calloc(weight_count, sizeof(float));
        grad->m_b = (float*)calloc(bias_count, sizeof(float));
        grad->v_w = (float*)calloc(weight_count, sizeof(float));
        grad->v_b = (float*)calloc(bias_count, sizeof(float));

        if (grad->m_w == NULL || grad->m_b == NULL ||
            grad->v_w == NULL || grad->v_b == NULL) {
            free(grad->m_w);
            free(grad->m_b);
            free(grad->v_w);
            free(grad->v_b);
            free(grad->weight_grad);
            free(grad->bias_grad);
            free(grad);
            return NULL;
        }
    }

    return grad;
}

/**
 * @brief Free gradient buffer
 *
 * @param grad Gradient buffer to free
 * @param optimizer Optimizer type
 */
static void free_gradients(MlpLayerGrad* grad, MlpOptimizerType optimizer) {
    if (grad == NULL) {
        return;
    }

    free(grad->weight_grad);
    free(grad->bias_grad);

    if (optimizer == MLP_OPT_SGD) {
        free(grad->velocity_w);
        free(grad->velocity_b);
    } else if (optimizer == MLP_OPT_ADAM) {
        free(grad->m_w);
        free(grad->m_b);
        free(grad->v_w);
        free(grad->v_b);
    }

    free(grad);
}

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

    for (i = 0; i < size; i++) {
        clip_pred = pred[i] > 1e-7f ? pred[i] : 1e-7f;
        clip_pred = clip_pred < 1.0f - 1e-7f ? clip_pred : 1.0f - 1e-7f;
        loss -= target[i] * logf(clip_pred);
    }

    softmax_cross_entropy_grad(pred, target, grad, size);

    return loss;
}

/**
 * @brief Derivative of activation function
 *
 * @param activation Activation type
 * @param output Output values
 * @param grad Output gradient (modified in place)
 * @param size Number of elements
 */
static void activation_derivative(MlpActivationType activation,
                                   const float* output,
                                   float* grad, size_t size) {
    size_t i;

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
 * @brief Update parameters with SGD+momentum
 *
 * v = momentum * v - lr * grad
 * w = w + v
 *
 * @param weights Weight parameters
 * @param bias Bias parameters
 * @param grad Gradient buffer
 * @param lr Learning rate
 * @param momentum Momentum coefficient
 * @param weight_decay Weight decay coefficient
 * @param weight_count Number of weights
 * @param bias_count Number of biases
 */
static void update_sgd(float* weights, float* bias,
                       const MlpLayerGrad* grad,
                       float lr, float momentum, float weight_decay,
                       size_t weight_count, size_t bias_count) {
    size_t i;

    for (i = 0; i < weight_count; i++) {
        grad->velocity_w[i] = momentum * grad->velocity_w[i] -
                              lr * (grad->weight_grad[i] + weight_decay * weights[i]);
        weights[i] += grad->velocity_w[i];
    }

    for (i = 0; i < bias_count; i++) {
        grad->velocity_b[i] = momentum * grad->velocity_b[i] -
                              lr * grad->bias_grad[i];
        bias[i] += grad->velocity_b[i];
    }
}

/**
 * @brief Update parameters with Adam optimizer
 *
 * m = beta1 * m + (1-beta1) * grad
 * v = beta2 * v + (1-beta2) * grad^2
 * w = w - lr * m / (sqrt(v) + eps)
 *
 * @param weights Weight parameters
 * @param bias Bias parameters
 * @param grad Gradient buffer
 * @param lr Learning rate
 * @param weight_decay Weight decay coefficient
 * @param t Current step number
 * @param weight_count Number of weights
 * @param bias_count Number of biases
 */
static void update_adam(float* weights, float* bias,
                        const MlpLayerGrad* grad,
                        float lr, float weight_decay,
                        size_t t,
                        size_t weight_count, size_t bias_count) {
    size_t i;
    float lr_corr;
    float m_corr_w, v_corr_w;
    float m_corr_b, v_corr_b;
    float beta1_pow_t;
    float beta2_pow_t;

    beta1_pow_t = powf(grad->beta1, (float)t);
    beta2_pow_t = powf(grad->beta2, (float)t);

    lr_corr = lr * sqrtf(1.0f - beta2_pow_t) / (1.0f - beta1_pow_t);

    for (i = 0; i < weight_count; i++) {
        grad->m_w[i] = grad->beta1 * grad->m_w[i] +
                       (1.0f - grad->beta1) * grad->weight_grad[i];
        grad->v_w[i] = grad->beta2 * grad->v_w[i] +
                       (1.0f - grad->beta2) * grad->weight_grad[i] * grad->weight_grad[i];

        m_corr_w = grad->m_w[i] / (1.0f - beta1_pow_t);
        v_corr_w = grad->v_w[i] / (1.0f - beta2_pow_t);

        weights[i] -= lr_corr * m_corr_w / (sqrtf(v_corr_w) + ADAM_EPS);
    }

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
 * @brief Forward pass storing intermediate outputs
 *
 * For training, we need to store activations for backprop.
 * This simplified version performs forward pass and computes loss gradient.
 *
 * @param ctx Training context
 * @param input Input data
 * @param target Target data
 * @param loss Output loss value
 * @return 0 on success, -1 on failure
 */
static int train_forward_pass(MlpTrainContext* ctx,
                              const float* input) {
    MlpInferContext* infer_ctx;
    MlpDenseLayer* layer;
    const float* current_input;
    size_t i;

    if (ctx == NULL || ctx->infer_ctx == NULL || input == NULL) {
        return -1;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;

    if (infer_ctx->layer_count == 0 || infer_ctx->layers == NULL) {
        return -1;
    }

    memcpy(ctx->input_buffer, input, infer_ctx->config.input_size * sizeof(float));

    for (i = 0; i < infer_ctx->layer_count; i++) {
        current_input = (i == 0U) ? ctx->input_buffer : ctx->activations[i - 1U];
        layer = infer_ctx->layers[i];
        mlp_dense_forward(layer, ctx->activations[i], current_input);
    }

    memcpy(infer_ctx->output_buffer,
           ctx->activations[infer_ctx->layer_count - 1U],
           infer_ctx->config.output_size * sizeof(float));

    return 0;
}

static int train_backward_pass(
    MlpTrainContext* ctx,
    const float* output_gradient,
    float* input_gradient
) {
    MlpInferContext* infer_ctx;
    float* current_delta;
    float* next_delta;
    const float* prev_activation;
    size_t max_size;
    size_t layer_cursor;

    if (ctx == NULL || ctx->infer_ctx == NULL || output_gradient == NULL) {
        return -1;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;
    max_size = infer_ctx->max_buffer_size;
    current_delta = (float*)calloc(max_size, sizeof(float));
    next_delta = (float*)calloc(max_size, sizeof(float));
    if (current_delta == NULL || next_delta == NULL) {
        free(current_delta);
        free(next_delta);
        return -1;
    }

    memcpy(current_delta, output_gradient, infer_ctx->config.output_size * sizeof(float));

    for (layer_cursor = infer_ctx->layer_count; layer_cursor > 0U; --layer_cursor) {
        size_t layer_index = layer_cursor - 1U;
        MlpDenseLayer* layer = infer_ctx->layers[layer_index];
        MlpLayerGrad* grad = &ctx->grads[layer_index];
        size_t input_size = layer->input_size;
        size_t output_size = layer->output_size;
        size_t output_index;
        size_t input_index;

        memset(next_delta, 0, max_size * sizeof(float));
        activation_derivative(layer->activation, ctx->activations[layer_index], current_delta, output_size);

        prev_activation = (layer_index == 0U) ? ctx->input_buffer : ctx->activations[layer_index - 1U];

        for (output_index = 0U; output_index < output_size; ++output_index) {
            grad->bias_grad[output_index] = current_delta[output_index];
            for (input_index = 0U; input_index < input_size; ++input_index) {
                grad->weight_grad[output_index * input_size + input_index] =
                    current_delta[output_index] * prev_activation[input_index];
                next_delta[input_index] +=
                    layer->weights[output_index * input_size + input_index] * current_delta[output_index];
            }
        }

        if (layer_index == 0U && input_gradient != NULL) {
            memcpy(input_gradient, next_delta, input_size * sizeof(float));
        }

        memcpy(current_delta, next_delta, input_size * sizeof(float));
    }

    free(current_delta);
    free(next_delta);
    return 0;
}

/**
 * @brief Update parameters using stored gradients
 *
 * @param ctx Training context
 * @param step Current step number (for Adam)
 */
static void train_update(MlpTrainContext* ctx, size_t step) {
    MlpInferContext* infer_ctx;
    MlpDenseLayer* layer;
    size_t i;
    size_t input_size;
    size_t output_size;
    float lr;

    if (ctx == NULL || ctx->infer_ctx == NULL) {
        return;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;
    lr = ctx->config.learning_rate;

    for (i = 0; i < infer_ctx->layer_count; i++) {
        layer = infer_ctx->layers[i];
        get_layer_sizes(infer_ctx, i, &input_size, &output_size);

        if (ctx->config.optimizer == MLP_OPT_SGD) {
            update_sgd(layer->weights, layer->bias, &ctx->grads[i],
                       lr, ctx->config.momentum, ctx->config.weight_decay,
                       input_size * output_size, output_size);
        } else if (ctx->config.optimizer == MLP_OPT_ADAM) {
            update_adam(layer->weights, layer->bias, &ctx->grads[i],
                       lr, ctx->config.weight_decay, step,
                       input_size * output_size, output_size);
        }
    }
}

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

    if (config != NULL) {
        ctx->config = *config;
    } else {
        free(ctx);
        return NULL;
    }

    ctx->infer_ctx = infer_ctx;
    ctx->layer_count = mlp_ctx->layer_count;

    max_size = mlp_ctx->config.input_size;
    if (mlp_ctx->config.output_size > max_size) {
        max_size = mlp_ctx->config.output_size;
    }

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

    return ctx;
}

void nn_mlp_train_destroy(MlpTrainContext* ctx) {
    size_t i;

    if (ctx == NULL) {
        return;
    }

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

    free(ctx->grads);
    free(ctx->activations);
    free(ctx->input_buffer);
    free(ctx->target_buffer);
    free(ctx->loss_buffer);
    free(ctx);
}

int nn_mlp_train_step_with_data(MlpTrainContext* ctx, const float* input, const float* target) {
    float loss;
    int rc;
    MlpInferContext* infer_ctx;
    size_t i;

    if (ctx == NULL || input == NULL || target == NULL) {
        return -1;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;

    rc = train_forward_pass(ctx, input);
    if (rc != 0) {
        return rc;
    }

    memcpy(ctx->target_buffer, target, infer_ctx->config.output_size * sizeof(float));
    if (ctx->config.loss_func == MLP_LOSS_MSE) {
        loss = mse_loss(infer_ctx->output_buffer, target, ctx->loss_buffer,
                        infer_ctx->config.output_size);
    } else {
        loss = cross_entropy_loss(infer_ctx->output_buffer, target, ctx->loss_buffer,
                                  infer_ctx->config.output_size);
    }

    rc = train_backward_pass(ctx, ctx->loss_buffer, NULL);
    if (rc != 0) {
        return rc;
    }

    ctx->total_steps++;
    train_update(ctx, ctx->total_steps);

    ctx->loss_history[ctx->loss_history_count % 100] = loss;
    if (ctx->loss_history_count < 100) {
        ctx->loss_history_count++;
    }

    if (infer_ctx != NULL) {
        for (i = 0; i < infer_ctx->config.output_size; i++) {
            infer_ctx->output_buffer[i] = ctx->activations[infer_ctx->layer_count - 1U][i];
        }
    }

    return 0;
}

int nn_mlp_train_step_ex(MlpTrainContext* ctx,
                          const MlpTrainStepInput* step_in,
                          MlpTrainStepOutput* step_out) {
    MlpInferContext* infer_ctx;
    float loss;
    int rc;
    size_t i;

    if (ctx == NULL || step_in == NULL || step_out == NULL) {
        return -1;
    }

    if (step_in->input == NULL || step_in->target == NULL) {
        return -1;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;

    rc = train_forward_pass(ctx, step_in->input);
    if (rc != 0) {
        return rc;
    }

    if (ctx->config.loss_func == MLP_LOSS_MSE) {
        loss = mse_loss(infer_ctx->output_buffer, step_in->target, ctx->loss_buffer,
                        infer_ctx->config.output_size);
    } else {
        loss = cross_entropy_loss(infer_ctx->output_buffer, step_in->target, ctx->loss_buffer,
                                  infer_ctx->config.output_size);
    }

    rc = train_backward_pass(ctx, ctx->loss_buffer, NULL);
    if (rc != 0) {
        return rc;
    }

    ctx->total_steps++;
    train_update(ctx, ctx->total_steps);

    step_out->loss = loss;

    if (step_out->predictions != NULL && infer_ctx != NULL) {
        for (i = 0; i < infer_ctx->config.output_size; i++) {
            step_out->predictions[i] = infer_ctx->output_buffer[i];
        }
    }

    ctx->loss_history[ctx->loss_history_count % 100] = loss;
    if (ctx->loss_history_count < 100) {
        ctx->loss_history_count++;
    }

    return 0;
}

int nn_mlp_train_step_with_output_gradient(
    MlpTrainContext* ctx,
    const float* input,
    const float* output_gradient,
    float* input_gradient
) {
    int rc;

    if (ctx == NULL || input == NULL || output_gradient == NULL) {
        return -1;
    }

    rc = train_forward_pass(ctx, input);
    if (rc != 0) {
        return rc;
    }

    rc = train_backward_pass(ctx, output_gradient, input_gradient);
    if (rc != 0) {
        return rc;
    }

    ctx->total_steps++;
    train_update(ctx, ctx->total_steps);
    return 0;
}

int nn_mlp_train_run_auto(MlpTrainContext* ctx, size_t epochs,
                           const float* input, const float* target,
                           size_t sample_count) {
    MlpInferContext* infer_ctx;
    size_t e;
    size_t s;
    size_t batch_size;
    int rc;

    if (ctx == NULL || input == NULL || target == NULL || sample_count == 0) {
        return -1;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;
    batch_size = ctx->config.batch_size;
    if (batch_size == 0) {
        batch_size = 1;
    }

    for (e = 0; e < epochs; e++) {
        for (s = 0; s < sample_count; s += batch_size) {
            size_t offset_in = s * infer_ctx->config.input_size;
            size_t offset_out = s * infer_ctx->config.output_size;
            size_t actual_batch = batch_size;

            if (s + batch_size > sample_count) {
                actual_batch = sample_count - s;
            }

            if (actual_batch == 1) {
                rc = nn_mlp_train_step_with_data(ctx,
                                       input + offset_in,
                                       target + offset_out);
            } else {
                float batch_input[256];
                float batch_target[256];
                size_t k;

                for (k = 0; k < actual_batch; k++) {
                    memcpy(batch_input + k * infer_ctx->config.input_size,
                           input + (s + k) * infer_ctx->config.input_size,
                           infer_ctx->config.input_size * sizeof(float));
                    memcpy(batch_target + k * infer_ctx->config.output_size,
                           target + (s + k) * infer_ctx->config.output_size,
                           infer_ctx->config.output_size * sizeof(float));
                }

                rc = nn_mlp_train_step_with_data(ctx, batch_input, batch_target);
            }

            if (rc != 0) {
                return rc;
            }
        }

        ctx->total_epochs++;
    }

    return 0;
}

float nn_mlp_train_compute_loss(MlpTrainContext* ctx,
                                 const float* predictions,
                                 const float* targets,
                                 size_t count) {
    float loss;
    float grad[256];

    if (ctx == NULL || predictions == NULL || targets == NULL) {
        return 0.0f;
    }

    if (ctx->config.loss_func == MLP_LOSS_MSE) {
        loss = mse_loss(predictions, targets, grad, count);
    } else {
        loss = cross_entropy_loss(predictions, targets, grad, count);
    }

    return loss;
}

int nn_mlp_train_save_checkpoint(MlpTrainContext* ctx, FILE* fp, float best_loss) {
    MlpInferContext* infer_ctx;
    MlpCheckpointHeader header;
    size_t i;
    size_t input_size;
    size_t output_size;
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

    rc = (int)fwrite(&header, sizeof(header), 1, fp);
    if (rc != 1) return 0;

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

    return 1;
}

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

    rc = (int)fread(&header, sizeof(header), 1, fp);
    if (rc != 1) return 0;

    if (header.network_hash != current_hash ||
        header.layout_hash != current_layout_hash) {
        return 0;
    }

    infer_ctx = (MlpInferContext*)ctx->infer_ctx;

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

    ctx->total_epochs = header.epoch;
    ctx->total_steps = header.step;
    ctx->checkpoint_network_hash = header.network_hash;
    ctx->checkpoint_layout_hash = header.layout_hash;

    return 1;
}

int nn_mlp_train_validate_checkpoint(FILE* fp,
                                      uint64_t current_hash,
                                      uint64_t current_layout_hash) {
    MlpCheckpointHeader header;
    int rc;

    if (fp == NULL) {
        return 0;
    }

    rc = (int)fread(&header, sizeof(header), 1, fp);
    if (rc != 1) return 0;

    if (header.network_hash != current_hash ||
        header.layout_hash != current_layout_hash) {
        return 0;
    }

    return 1;
}

void nn_mlp_train_get_stats(MlpTrainContext* ctx,
                             size_t* out_epochs,
                             size_t* out_steps,
                             float* out_avg_loss) {
    size_t i;
    float total;

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
        *out_avg_loss = 0.0f;
        if (ctx->loss_history_count > 0) {
            total = 0.0f;
            for (i = 0; i < ctx->loss_history_count; i++) {
                total += ctx->loss_history[i];
            }
            *out_avg_loss = total / (float)ctx->loss_history_count;
        }
    }
}

/**
 * @brief Wrapper for registry-compatible train step
 *
 * Adapter function that converts void* context to MlpTrainContext*
 * for registration in the train registry.
 *
 * @param context Void pointer to MlpTrainContext
 * @return 0 on success, -1 on failure
 */
int nn_mlp_train_step(void* context) {
    MlpTrainContext* ctx;
    float dummy_input[16];
    float dummy_target[16];
    size_t i;

    if (context == NULL) {
        return -1;
    }

    ctx = (MlpTrainContext*)context;

    for (i = 0; i < 16; i++) {
        dummy_input[i] = 0.0f;
        dummy_target[i] = 0.0f;
    }

    return nn_mlp_train_step_with_data(ctx, dummy_input, dummy_target);
}
