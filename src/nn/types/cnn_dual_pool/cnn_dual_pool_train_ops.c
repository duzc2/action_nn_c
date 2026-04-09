/**
 * @file cnn_dual_pool_train_ops.c
 * @brief Dual-pool CNN training backend used by generated graph code.
 */

#include "cnn_dual_pool_train_ops.h"

#include <stdlib.h>
#include <string.h>

static float cnn_dual_pool_activation_derivative_from_output(float output, CnnDualPoolActivationType activation) {
    switch (activation) {
        case CNN_DUAL_POOL_ACT_RELU:
            return output > 0.0f ? 1.0f : 0.0f;
        case CNN_DUAL_POOL_ACT_TANH:
            return 1.0f - (output * output);
        case CNN_DUAL_POOL_ACT_NONE:
        default:
            return 1.0f;
    }
}

static size_t cnn_dual_pool_frame_index(const CnnDualPoolConfig* config, size_t channel, size_t row, size_t column) {
    return ((channel * config->frame_height) + row) * config->frame_width + column;
}

static size_t cnn_dual_pool_kernel_index(const CnnDualPoolConfig* config, size_t filter_index, size_t channel_index, size_t kernel_row, size_t kernel_column) {
    size_t kernel_plane = config->kernel_size * config->kernel_size;
    return (((filter_index * config->channel_count) + channel_index) * kernel_plane) + (kernel_row * config->kernel_size) + kernel_column;
}

static size_t cnn_dual_pool_conv_position_count(const CnnDualPoolConfig* config) {
    return (config->frame_height - config->kernel_size + 1U) * (config->frame_width - config->kernel_size + 1U);
}

static size_t cnn_dual_pool_pooled_feature_count(const CnnDualPoolConfig* config) {
    return config->filter_count * 2U;
}

static void cnn_dual_pool_zero_gradients(CnnDualPoolTrainContext* context) {
    const CnnDualPoolConfig* config;
    size_t conv_weight_count;
    size_t projection_weight_count;
    size_t pooled_cache_count;

    if (context == NULL || context->infer_ctx == NULL) {
        return;
    }

    config = &context->infer_ctx->config;
    conv_weight_count = config->filter_count * config->channel_count * config->kernel_size * config->kernel_size;
    projection_weight_count = config->feature_size * cnn_dual_pool_pooled_feature_count(config);
    pooled_cache_count = config->sequence_length * cnn_dual_pool_pooled_feature_count(config);

    (void)memset(context->pooled_gradient_cache, 0, pooled_cache_count * sizeof(float));
    (void)memset(context->conv_weight_grad, 0, conv_weight_count * sizeof(float));
    (void)memset(context->conv_bias_grad, 0, config->filter_count * sizeof(float));
    (void)memset(context->projection_weight_grad, 0, projection_weight_count * sizeof(float));
    (void)memset(context->projection_bias_grad, 0, config->feature_size * sizeof(float));
}

static void cnn_dual_pool_apply_parameter_update(CnnDualPoolTrainContext* context) {
    CnnDualPoolInferContext* infer_ctx = context->infer_ctx;
    const CnnDualPoolConfig* config = &infer_ctx->config;
    size_t conv_weight_count = config->filter_count * config->channel_count * config->kernel_size * config->kernel_size;
    size_t projection_weight_count = config->feature_size * cnn_dual_pool_pooled_feature_count(config);
    size_t weight_index;

    for (weight_index = 0U; weight_index < conv_weight_count; ++weight_index) {
        infer_ctx->conv_weights[weight_index] -= context->config.learning_rate * (context->conv_weight_grad[weight_index] + (context->config.weight_decay * infer_ctx->conv_weights[weight_index]));
    }
    for (weight_index = 0U; weight_index < config->filter_count; ++weight_index) {
        infer_ctx->conv_bias[weight_index] -= context->config.learning_rate * context->conv_bias_grad[weight_index];
    }
    for (weight_index = 0U; weight_index < projection_weight_count; ++weight_index) {
        infer_ctx->projection_weights[weight_index] -= context->config.learning_rate * (context->projection_weight_grad[weight_index] + (context->config.weight_decay * infer_ctx->projection_weights[weight_index]));
    }
    for (weight_index = 0U; weight_index < config->feature_size; ++weight_index) {
        infer_ctx->projection_bias[weight_index] -= context->config.learning_rate * context->projection_bias_grad[weight_index];
    }
}

static int cnn_dual_pool_backpropagate(CnnDualPoolTrainContext* context, const float* input, const float* output_gradient, float* input_gradient) {
    CnnDualPoolInferContext* infer_ctx;
    const CnnDualPoolConfig* config;
    size_t frame_stride;
    size_t output_grid_width;
    size_t output_grid_height;
    size_t output_positions;
    size_t pooled_feature_count;
    size_t step_index;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || output_gradient == NULL) {
        return -1;
    }

    infer_ctx = context->infer_ctx;
    config = &infer_ctx->config;
    frame_stride = config->frame_width * config->frame_height * config->channel_count;
    output_grid_width = config->frame_width - config->kernel_size + 1U;
    output_grid_height = config->frame_height - config->kernel_size + 1U;
    output_positions = cnn_dual_pool_conv_position_count(config);
    pooled_feature_count = cnn_dual_pool_pooled_feature_count(config);
    if (output_positions == 0U) {
        return -1;
    }

    cnn_dual_pool_zero_gradients(context);
    if (input_gradient != NULL) {
        (void)memset(input_gradient, 0, config->total_input_size * sizeof(float));
    }

    for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
        size_t feature_index;
        for (feature_index = 0U; feature_index < config->feature_size; ++feature_index) {
            size_t output_index = (step_index * config->feature_size) + feature_index;
            float output_value = infer_ctx->output_buffer[output_index];
            float dz = output_gradient[output_index] * cnn_dual_pool_activation_derivative_from_output(output_value, config->output_activation);
            size_t pooled_index;

            context->projection_bias_grad[feature_index] += dz;
            for (pooled_index = 0U; pooled_index < pooled_feature_count; ++pooled_index) {
                size_t weight_index = (feature_index * pooled_feature_count) + pooled_index;
                size_t cache_index = (step_index * pooled_feature_count) + pooled_index;
                context->projection_weight_grad[weight_index] += dz * context->pooled_activation_cache[cache_index];
                context->pooled_gradient_cache[cache_index] += infer_ctx->projection_weights[weight_index] * dz;
            }
        }
    }

    for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
        const float* frame = input + (step_index * frame_stride);
        size_t filter_index;
        for (filter_index = 0U; filter_index < config->filter_count; ++filter_index) {
            size_t avg_cache_index = (step_index * pooled_feature_count) + (filter_index * 2U);
            size_t max_cache_index = avg_cache_index + 1U;
            float avg_output = context->pooled_activation_cache[avg_cache_index];
            float max_output = context->pooled_activation_cache[max_cache_index];
            float davg_linear = context->pooled_gradient_cache[avg_cache_index] * cnn_dual_pool_activation_derivative_from_output(avg_output, config->pooling_activation);
            float dmax_linear = context->pooled_gradient_cache[max_cache_index] * cnn_dual_pool_activation_derivative_from_output(max_output, config->pooling_activation);
            float position_scale = davg_linear / (float)output_positions;
            size_t argmax_position = context->max_index_cache[(step_index * config->filter_count) + filter_index];
            size_t out_row;
            size_t out_column;
            size_t position_counter = 0U;

            context->conv_bias_grad[filter_index] += davg_linear + dmax_linear;
            for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                    float position_gradient = position_scale;
                    size_t channel_index;
                    if (position_counter == argmax_position) {
                        position_gradient += dmax_linear;
                    }
                    for (channel_index = 0U; channel_index < config->channel_count; ++channel_index) {
                        size_t kernel_row;
                        for (kernel_row = 0U; kernel_row < config->kernel_size; ++kernel_row) {
                            size_t kernel_column;
                            for (kernel_column = 0U; kernel_column < config->kernel_size; ++kernel_column) {
                                size_t input_index = cnn_dual_pool_frame_index(config, channel_index, out_row + kernel_row, out_column + kernel_column);
                                size_t weight_index = cnn_dual_pool_kernel_index(config, filter_index, channel_index, kernel_row, kernel_column);
                                context->conv_weight_grad[weight_index] += position_gradient * frame[input_index];
                                if (input_gradient != NULL) {
                                    input_gradient[(step_index * frame_stride) + input_index] += infer_ctx->conv_weights[weight_index] * position_gradient;
                                }
                            }
                        }
                    }
                    position_counter += 1U;
                }
            }
        }
    }

    cnn_dual_pool_apply_parameter_update(context);
    return 0;
}

CnnDualPoolTrainContext* nn_cnn_dual_pool_train_create(void* infer_ctx_ptr, const CnnDualPoolTrainConfig* config) {
    CnnDualPoolTrainContext* context;
    CnnDualPoolInferContext* infer_ctx = (CnnDualPoolInferContext*)infer_ctx_ptr;
    const CnnDualPoolConfig* infer_config;
    size_t pooled_cache_count;
    size_t conv_weight_count;
    size_t projection_weight_count;

    if (infer_ctx == NULL || config == NULL) {
        return NULL;
    }

    infer_config = &infer_ctx->config;
    pooled_cache_count = infer_config->sequence_length * cnn_dual_pool_pooled_feature_count(infer_config);
    conv_weight_count = infer_config->filter_count * infer_config->channel_count * infer_config->kernel_size * infer_config->kernel_size;
    projection_weight_count = infer_config->feature_size * cnn_dual_pool_pooled_feature_count(infer_config);

    context = (CnnDualPoolTrainContext*)calloc(1U, sizeof(CnnDualPoolTrainContext));
    if (context == NULL) {
        return NULL;
    }

    context->infer_ctx = infer_ctx;
    context->config = *config;
    context->pooled_linear_cache = (float*)calloc(pooled_cache_count, sizeof(float));
    context->pooled_activation_cache = (float*)calloc(pooled_cache_count, sizeof(float));
    context->max_index_cache = (size_t*)calloc(infer_config->sequence_length * infer_config->filter_count, sizeof(size_t));
    context->output_linear_cache = (float*)calloc(infer_config->sequence_length * infer_config->feature_size, sizeof(float));
    context->pooled_gradient_cache = (float*)calloc(pooled_cache_count, sizeof(float));
    context->conv_weight_grad = (float*)calloc(conv_weight_count, sizeof(float));
    context->conv_bias_grad = (float*)calloc(infer_config->filter_count, sizeof(float));
    context->projection_weight_grad = (float*)calloc(projection_weight_count, sizeof(float));
    context->projection_bias_grad = (float*)calloc(infer_config->feature_size, sizeof(float));

    if (context->pooled_linear_cache == NULL || context->pooled_activation_cache == NULL || context->max_index_cache == NULL ||
        context->output_linear_cache == NULL || context->pooled_gradient_cache == NULL || context->conv_weight_grad == NULL ||
        context->conv_bias_grad == NULL || context->projection_weight_grad == NULL || context->projection_bias_grad == NULL) {
        nn_cnn_dual_pool_train_destroy(context);
        return NULL;
    }
    return context;
}

void nn_cnn_dual_pool_train_destroy(CnnDualPoolTrainContext* context) {
    if (context == NULL) {
        return;
    }
    free(context->pooled_linear_cache);
    free(context->pooled_activation_cache);
    free(context->max_index_cache);
    free(context->output_linear_cache);
    free(context->pooled_gradient_cache);
    free(context->conv_weight_grad);
    free(context->conv_bias_grad);
    free(context->projection_weight_grad);
    free(context->projection_bias_grad);
    free(context);
}

int nn_cnn_dual_pool_train_step_with_output_gradient(CnnDualPoolTrainContext* context, const float* input, const float* output_gradient, float* input_gradient) {
    CnnDualPoolInferContext* infer_ctx;
    int rc;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || output_gradient == NULL) {
        return -1;
    }

    infer_ctx = context->infer_ctx;
    rc = nn_cnn_dual_pool_forward_pass(infer_ctx, input, infer_ctx->output_buffer, context->pooled_linear_cache, context->pooled_activation_cache, context->max_index_cache, context->output_linear_cache);
    if (rc != 0) {
        return rc;
    }
    rc = cnn_dual_pool_backpropagate(context, input, output_gradient, input_gradient);
    if (rc != 0) {
        return rc;
    }
    context->total_steps += 1U;
    return 0;
}

int nn_cnn_dual_pool_train_step_with_data(CnnDualPoolTrainContext* context, const float* input, const float* target) {
    CnnDualPoolInferContext* infer_ctx;
    const CnnDualPoolConfig* config;
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
    rc = nn_cnn_dual_pool_forward_pass(infer_ctx, input, infer_ctx->output_buffer, context->pooled_linear_cache, context->pooled_activation_cache, context->max_index_cache, context->output_linear_cache);
    if (rc != 0) {
        return rc;
    }

    output_gradient = (float*)calloc(output_size, sizeof(float));
    if (output_gradient == NULL) {
        return -1;
    }
    for (output_index = 0U; output_index < output_size; ++output_index) {
        float diff = infer_ctx->output_buffer[output_index] - target[output_index];
        loss += diff * diff;
        output_gradient[output_index] = (2.0f * diff) / (float)output_size;
    }

    rc = cnn_dual_pool_backpropagate(context, input, output_gradient, NULL);
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

void nn_cnn_dual_pool_train_get_stats(CnnDualPoolTrainContext* context, size_t* out_epochs, size_t* out_steps, float* out_avg_loss) {
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

int nn_cnn_dual_pool_train_step(void* ctx) {
    CnnDualPoolTrainContext* context = (CnnDualPoolTrainContext*)ctx;
    float* dummy_input;
    float* dummy_target;
    int rc;

    if (context == NULL || context->infer_ctx == NULL) {
        return -1;
    }

    dummy_input = (float*)calloc(context->infer_ctx->config.total_input_size, sizeof(float));
    dummy_target = (float*)calloc(context->infer_ctx->config.sequence_length * context->infer_ctx->config.feature_size, sizeof(float));
    if (dummy_input == NULL || dummy_target == NULL) {
        free(dummy_input);
        free(dummy_target);
        return -1;
    }

    rc = nn_cnn_dual_pool_train_step_with_data(context, dummy_input, dummy_target);
    free(dummy_input);
    free(dummy_target);
    return rc;
}
