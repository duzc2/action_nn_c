/**
 * @file cnn_train_ops.c
 * @brief Tiny CNN training backend used by generated graph code.
 *
 * The training path mirrors the compact design of the inference backend. It is
 * intentionally small, but it still exposes the important behaviour that nested
 * graph training needs: cache the forward pass, accept either direct targets or
 * externally supplied output gradients, update parameters, and optionally return
 * dL/dX to an upstream leaf.
 */

#include "cnn_train_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Recover d(activation)/d(linear) from the post-activation output value.
 */
static float cnn_activation_derivative_from_output(float output, CnnActivationType activation) {
    switch (activation) {
        case CNN_ACT_RELU:
            return output > 0.0f ? 1.0f : 0.0f;
        case CNN_ACT_TANH:
            return 1.0f - (output * output);
        case CNN_ACT_NONE:
        default:
            return 1.0f;
    }
}

/**
 * @brief Flatten one frame coordinate into the CNN input layout.
 */
static size_t cnn_frame_index(const CnnConfig* config, size_t channel, size_t row, size_t column) {
    return ((channel * config->frame_height) + row) * config->frame_width + column;
}

/**
 * @brief Flatten one convolution-weight coordinate into the CNN parameter tensor.
 */
static size_t cnn_kernel_index(
    const CnnConfig* config,
    size_t filter_index,
    size_t channel_index,
    size_t kernel_row,
    size_t kernel_column
) {
    size_t kernel_plane = config->kernel_size * config->kernel_size;
    return (((filter_index * config->channel_count) + channel_index) * kernel_plane) +
        (kernel_row * config->kernel_size) + kernel_column;
}

/**
 * @brief Count convolution windows in one frame.
 */
static size_t cnn_conv_position_count(const CnnConfig* config) {
    return (config->frame_height - config->kernel_size + 1U) *
        (config->frame_width - config->kernel_size + 1U);
}

/**
 * @brief Reset all gradient buffers before a new backward pass.
 */
static void cnn_zero_gradients(CnnTrainContext* context) {
    const CnnConfig* config;
    size_t conv_weight_count;
    size_t projection_weight_count;
    size_t pooled_cache_count;

    if (context == NULL || context->infer_ctx == NULL) {
        return;
    }

    config = &context->infer_ctx->config;
    conv_weight_count = config->filter_count * config->channel_count * config->kernel_size * config->kernel_size;
    projection_weight_count = config->feature_size * config->filter_count;
    pooled_cache_count = config->sequence_length * config->filter_count;

    (void)memset(context->pooled_gradient_cache, 0, pooled_cache_count * sizeof(float));
    (void)memset(context->conv_weight_grad, 0, conv_weight_count * sizeof(float));
    (void)memset(context->conv_bias_grad, 0, config->filter_count * sizeof(float));
    (void)memset(context->projection_weight_grad, 0, projection_weight_count * sizeof(float));
    (void)memset(context->projection_bias_grad, 0, config->feature_size * sizeof(float));
}

/**
 * @brief Apply the accumulated gradients with a simple SGD-style update.
 */
static void cnn_apply_parameter_update(CnnTrainContext* context) {
    CnnInferContext* infer_ctx;
    const CnnConfig* config;
    size_t conv_weight_count;
    size_t projection_weight_count;
    size_t weight_index;

    infer_ctx = context->infer_ctx;
    config = &infer_ctx->config;
    conv_weight_count = config->filter_count * config->channel_count * config->kernel_size * config->kernel_size;
    projection_weight_count = config->feature_size * config->filter_count;

    /* Weight decay is applied only to true weights, not to bias vectors. */
    for (weight_index = 0U; weight_index < conv_weight_count; ++weight_index) {
        infer_ctx->conv_weights[weight_index] -= context->config.learning_rate * (
            context->conv_weight_grad[weight_index] +
            (context->config.weight_decay * infer_ctx->conv_weights[weight_index])
        );
    }
    for (weight_index = 0U; weight_index < config->filter_count; ++weight_index) {
        infer_ctx->conv_bias[weight_index] -=
            context->config.learning_rate * context->conv_bias_grad[weight_index];
    }
    for (weight_index = 0U; weight_index < projection_weight_count; ++weight_index) {
        infer_ctx->projection_weights[weight_index] -= context->config.learning_rate * (
            context->projection_weight_grad[weight_index] +
            (context->config.weight_decay * infer_ctx->projection_weights[weight_index])
        );
    }
    for (weight_index = 0U; weight_index < config->feature_size; ++weight_index) {
        infer_ctx->projection_bias[weight_index] -=
            context->config.learning_rate * context->projection_bias_grad[weight_index];
    }
}

/**
 * @brief Backpropagate one externally supplied dL/dY through the CNN leaf.
 */
static int cnn_backpropagate(
    CnnTrainContext* context,
    const float* input,
    const float* output_gradient,
    float* input_gradient
) {
    CnnInferContext* infer_ctx;
    const CnnConfig* config;
    size_t frame_stride;
    size_t output_grid_width;
    size_t output_grid_height;
    size_t output_positions;
    size_t step_index;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || output_gradient == NULL) {
        return -1;
    }

    infer_ctx = context->infer_ctx;
    config = &infer_ctx->config;
    frame_stride = config->frame_width * config->frame_height * config->channel_count;
    output_grid_width = config->frame_width - config->kernel_size + 1U;
    output_grid_height = config->frame_height - config->kernel_size + 1U;
    output_positions = cnn_conv_position_count(config);
    if (output_positions == 0U) {
        return -1;
    }

    cnn_zero_gradients(context);
    if (input_gradient != NULL) {
        (void)memset(input_gradient, 0, config->total_input_size * sizeof(float));
    }

    /* Stage 1: output projection gradients and pooled-feature backprop signal. */
    for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
        size_t feature_index;
        for (feature_index = 0U; feature_index < config->feature_size; ++feature_index) {
            size_t output_index = (step_index * config->feature_size) + feature_index;
            float output_value = infer_ctx->output_buffer[output_index];
            float dz = output_gradient[output_index] *
                cnn_activation_derivative_from_output(output_value, config->output_activation);
            size_t filter_index;

            context->projection_bias_grad[feature_index] += dz;
            for (filter_index = 0U; filter_index < config->filter_count; ++filter_index) {
                size_t pooled_index = (step_index * config->filter_count) + filter_index;
                size_t weight_index = (feature_index * config->filter_count) + filter_index;
                context->projection_weight_grad[weight_index] +=
                    dz * context->pooled_activation_cache[pooled_index];
                context->pooled_gradient_cache[pooled_index] +=
                    infer_ctx->projection_weights[weight_index] * dz;
            }
        }
    }

    /* Stage 2: pooled-feature gradients back through shared convolution kernels. */
    for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
        const float* frame = input + (step_index * frame_stride);
        size_t filter_index;

        for (filter_index = 0U; filter_index < config->filter_count; ++filter_index) {
            size_t pooled_index = (step_index * config->filter_count) + filter_index;
            float pooled_output = context->pooled_activation_cache[pooled_index];
            float dpool_linear = context->pooled_gradient_cache[pooled_index] *
                cnn_activation_derivative_from_output(pooled_output, config->pooling_activation);
            float position_scale = dpool_linear / (float)output_positions;
            size_t out_row;
            size_t out_column;

            context->conv_bias_grad[filter_index] += dpool_linear;

            /* Every valid convolution window contributes equally to the pooled mean. */
            for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                    size_t channel_index;
                    for (channel_index = 0U; channel_index < config->channel_count; ++channel_index) {
                        size_t kernel_row;
                        for (kernel_row = 0U; kernel_row < config->kernel_size; ++kernel_row) {
                            size_t kernel_column;
                            for (kernel_column = 0U; kernel_column < config->kernel_size; ++kernel_column) {
                                size_t input_index = cnn_frame_index(
                                    config,
                                    channel_index,
                                    out_row + kernel_row,
                                    out_column + kernel_column
                                );
                                size_t weight_index = cnn_kernel_index(
                                    config,
                                    filter_index,
                                    channel_index,
                                    kernel_row,
                                    kernel_column
                                );
                                context->conv_weight_grad[weight_index] +=
                                    position_scale * frame[input_index];
                                if (input_gradient != NULL) {
                                    input_gradient[(step_index * frame_stride) + input_index] +=
                                        infer_ctx->conv_weights[weight_index] * position_scale;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    cnn_apply_parameter_update(context);
    return 0;
}

/**
 * @brief Create one CNN training context wrapped around an existing infer context.
 */
CnnTrainContext* nn_cnn_train_create(void* infer_ctx_ptr, const CnnTrainConfig* config) {
    CnnTrainContext* context;
    CnnInferContext* infer_ctx = (CnnInferContext*)infer_ctx_ptr;
    const CnnConfig* infer_config;
    size_t pooled_cache_count;
    size_t conv_weight_count;
    size_t projection_weight_count;

    if (infer_ctx == NULL || config == NULL) {
        return NULL;
    }

    infer_config = &infer_ctx->config;
    pooled_cache_count = infer_config->sequence_length * infer_config->filter_count;
    conv_weight_count = infer_config->filter_count * infer_config->channel_count *
        infer_config->kernel_size * infer_config->kernel_size;
    projection_weight_count = infer_config->feature_size * infer_config->filter_count;

    context = (CnnTrainContext*)calloc(1U, sizeof(CnnTrainContext));
    if (context == NULL) {
        return NULL;
    }

    context->infer_ctx = infer_ctx;
    context->config = *config;
    context->pooled_linear_cache = (float*)calloc(pooled_cache_count, sizeof(float));
    context->pooled_activation_cache = (float*)calloc(pooled_cache_count, sizeof(float));
    context->output_linear_cache = (float*)calloc(
        infer_config->sequence_length * infer_config->feature_size,
        sizeof(float)
    );
    context->pooled_gradient_cache = (float*)calloc(pooled_cache_count, sizeof(float));
    context->conv_weight_grad = (float*)calloc(conv_weight_count, sizeof(float));
    context->conv_bias_grad = (float*)calloc(infer_config->filter_count, sizeof(float));
    context->projection_weight_grad = (float*)calloc(projection_weight_count, sizeof(float));
    context->projection_bias_grad = (float*)calloc(infer_config->feature_size, sizeof(float));

    if (context->pooled_linear_cache == NULL || context->pooled_activation_cache == NULL ||
        context->output_linear_cache == NULL || context->pooled_gradient_cache == NULL ||
        context->conv_weight_grad == NULL || context->conv_bias_grad == NULL ||
        context->projection_weight_grad == NULL || context->projection_bias_grad == NULL) {
        nn_cnn_train_destroy(context);
        return NULL;
    }

    return context;
}

/**
 * @brief Free every training-side scratch buffer owned by the CNN trainer.
 */
void nn_cnn_train_destroy(CnnTrainContext* context) {
    if (context == NULL) {
        return;
    }

    free(context->pooled_linear_cache);
    free(context->pooled_activation_cache);
    free(context->output_linear_cache);
    free(context->pooled_gradient_cache);
    free(context->conv_weight_grad);
    free(context->conv_bias_grad);
    free(context->projection_weight_grad);
    free(context->projection_bias_grad);
    free(context);
}

/**
 * @brief Run one graph-mode update using a caller-supplied output gradient.
 */
int nn_cnn_train_step_with_output_gradient(
    CnnTrainContext* context,
    const float* input,
    const float* output_gradient,
    float* input_gradient
) {
    CnnInferContext* infer_ctx;
    int rc;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || output_gradient == NULL) {
        return -1;
    }

    infer_ctx = context->infer_ctx;
    rc = nn_cnn_forward_pass(
        infer_ctx,
        input,
        infer_ctx->output_buffer,
        context->pooled_linear_cache,
        context->pooled_activation_cache,
        context->output_linear_cache
    );
    if (rc != 0) {
        return rc;
    }

    rc = cnn_backpropagate(context, input, output_gradient, input_gradient);
    if (rc != 0) {
        return rc;
    }

    context->total_steps += 1U;
    return 0;
}

/**
 * @brief Run one supervised update with an explicit target tensor.
 */
int nn_cnn_train_step_with_data(CnnTrainContext* context, const float* input, const float* target) {
    CnnInferContext* infer_ctx;
    const CnnConfig* config;
    size_t output_size;
    float* output_gradient;
    size_t output_index;
    int rc;
    float loss = 0.0f;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || target == NULL) {
        return -1;
    }

    infer_ctx = context->infer_ctx;
    config = &infer_ctx->config;
    output_size = config->sequence_length * config->feature_size;

    rc = nn_cnn_forward_pass(
        infer_ctx,
        input,
        infer_ctx->output_buffer,
        context->pooled_linear_cache,
        context->pooled_activation_cache,
        context->output_linear_cache
    );
    if (rc != 0) {
        return rc;
    }

    output_gradient = (float*)calloc(output_size, sizeof(float));
    if (output_gradient == NULL) {
        return -1;
    }

    /* The standalone supervised path uses simple MSE to stay transparent. */
    for (output_index = 0U; output_index < output_size; ++output_index) {
        float diff = infer_ctx->output_buffer[output_index] - target[output_index];
        loss += diff * diff;
        output_gradient[output_index] = (2.0f * diff) / (float)output_size;
    }

    rc = cnn_backpropagate(context, input, output_gradient, NULL);
    free(output_gradient);
    if (rc != 0) {
        return rc;
    }

    context->total_steps += 1U;
    context->last_loss = loss / (float)output_size;
    context->cumulative_loss += context->last_loss;
    context->average_loss = context->cumulative_loss / (float)context->total_steps;
    return 0;
}

/**
 * @brief Report coarse training statistics consumed by generated wrappers.
 */
void nn_cnn_train_get_stats(
    CnnTrainContext* context,
    size_t* out_epochs,
    size_t* out_steps,
    float* out_avg_loss
) {
    if (context == NULL) {
        return;
    }

    if (out_epochs != NULL) {
        *out_epochs = context->total_epochs;
    }
    if (out_steps != NULL) {
        *out_steps = context->total_steps;
    }
    if (out_avg_loss != NULL) {
        *out_avg_loss = context->average_loss;
    }
}

/**
 * @brief Minimal compatibility wrapper for the registry's small train-step hook.
 */
int nn_cnn_train_step(void* ctx) {
    CnnTrainContext* context = (CnnTrainContext*)ctx;
    float* dummy_input;
    float* dummy_target;
    int rc;

    if (context == NULL || context->infer_ctx == NULL) {
        return -1;
    }

    dummy_input = (float*)calloc(context->infer_ctx->config.total_input_size, sizeof(float));
    dummy_target = (float*)calloc(
        context->infer_ctx->config.sequence_length * context->infer_ctx->config.feature_size,
        sizeof(float)
    );
    if (dummy_input == NULL || dummy_target == NULL) {
        free(dummy_input);
        free(dummy_target);
        return -1;
    }

    rc = nn_cnn_train_step_with_data(context, dummy_input, dummy_target);
    free(dummy_input);
    free(dummy_target);
    return rc;
}
