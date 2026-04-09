/**
 * @file cnn_dual_pool_infer_ops.c
 * @brief Dual-pool CNN inference backend used by generated graph code.
 */

#include "cnn_dual_pool_infer_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define CNN_DUAL_POOL_ABI_VERSION 1U

typedef struct {
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t abi_version;
    uint32_t sequence_length;
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t channel_count;
    uint32_t kernel_size;
    uint32_t filter_count;
    uint32_t feature_size;
} CnnDualPoolWeightHeader;

static uint32_t cnn_dual_pool_next_random(uint32_t* state) {
    uint32_t value = *state;
    value = value * 1664525U + 1013904223U;
    *state = value;
    return value;
}

static float cnn_dual_pool_random_weight(uint32_t* state, float scale) {
    float normalized = (float)(cnn_dual_pool_next_random(state) & 0xFFFFU) / 65535.0f;
    return (normalized - 0.5f) * scale;
}

static float cnn_dual_pool_apply_activation(float value, CnnDualPoolActivationType activation) {
    switch (activation) {
        case CNN_DUAL_POOL_ACT_RELU:
            return value > 0.0f ? value : 0.0f;
        case CNN_DUAL_POOL_ACT_TANH:
            return tanhf(value);
        case CNN_DUAL_POOL_ACT_NONE:
        default:
            return value;
    }
}

static int cnn_dual_pool_config_is_valid(const CnnDualPoolConfig* config) {
    size_t frame_stride;

    if (config == NULL || config->sequence_length == 0U) {
        return 0;
    }
    if (config->frame_width == 0U || config->frame_height == 0U || config->channel_count == 0U) {
        return 0;
    }
    if (config->kernel_size == 0U || config->kernel_size > config->frame_width || config->kernel_size > config->frame_height) {
        return 0;
    }
    if (config->filter_count == 0U || config->feature_size == 0U) {
        return 0;
    }

    frame_stride = config->frame_width * config->frame_height * config->channel_count;
    if (frame_stride == 0U || config->total_input_size != frame_stride * config->sequence_length) {
        return 0;
    }
    return 1;
}

static size_t cnn_dual_pool_conv_position_count(const CnnDualPoolConfig* config) {
    return (config->frame_height - config->kernel_size + 1U) * (config->frame_width - config->kernel_size + 1U);
}

static size_t cnn_dual_pool_pooled_feature_count(const CnnDualPoolConfig* config) {
    return config->filter_count * 2U;
}

static uint64_t cnn_dual_pool_compute_layout_hash(const CnnDualPoolConfig* config) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    const uint64_t prime = 0x100000001b3ULL;

    if (config == NULL) {
        return hash;
    }
    hash ^= (uint64_t)config->total_input_size; hash *= prime;
    hash ^= (uint64_t)config->sequence_length; hash *= prime;
    hash ^= (uint64_t)config->frame_width; hash *= prime;
    hash ^= (uint64_t)config->frame_height; hash *= prime;
    hash ^= (uint64_t)config->channel_count; hash *= prime;
    hash ^= (uint64_t)config->kernel_size; hash *= prime;
    hash ^= (uint64_t)config->filter_count; hash *= prime;
    hash ^= (uint64_t)config->feature_size; hash *= prime;
    hash ^= (uint64_t)config->pooling_activation; hash *= prime;
    hash ^= (uint64_t)config->output_activation; hash *= prime;
    return hash;
}

static size_t cnn_dual_pool_frame_index(const CnnDualPoolConfig* config, size_t channel, size_t row, size_t column) {
    return ((channel * config->frame_height) + row) * config->frame_width + column;
}

static size_t cnn_dual_pool_kernel_index(const CnnDualPoolConfig* config, size_t filter_index, size_t channel_index, size_t kernel_row, size_t kernel_column) {
    size_t kernel_plane = config->kernel_size * config->kernel_size;
    return (((filter_index * config->channel_count) + channel_index) * kernel_plane) + (kernel_row * config->kernel_size) + kernel_column;
}

static size_t cnn_dual_pool_conv_weight_count(const CnnDualPoolConfig* config) {
    return config->filter_count * config->channel_count * config->kernel_size * config->kernel_size;
}

static size_t cnn_dual_pool_projection_weight_count(const CnnDualPoolConfig* config) {
    return config->feature_size * cnn_dual_pool_pooled_feature_count(config);
}

CnnDualPoolInferContext* nn_cnn_dual_pool_infer_create(void) {
    return NULL;
}

CnnDualPoolInferContext* nn_cnn_dual_pool_infer_create_with_config(const CnnDualPoolConfig* config, uint32_t seed) {
    CnnDualPoolInferContext* context;
    size_t conv_weight_count;
    size_t projection_weight_count;
    size_t pooled_feature_count;
    size_t value_index;

    if (!cnn_dual_pool_config_is_valid(config)) {
        return NULL;
    }

    context = (CnnDualPoolInferContext*)calloc(1U, sizeof(CnnDualPoolInferContext));
    if (context == NULL) {
        return NULL;
    }

    context->config = *config;
    context->rng_state = seed != 0U ? seed : (config->seed != 0U ? config->seed : 1U);
    conv_weight_count = cnn_dual_pool_conv_weight_count(config);
    projection_weight_count = cnn_dual_pool_projection_weight_count(config);
    pooled_feature_count = cnn_dual_pool_pooled_feature_count(config);

    context->conv_weights = (float*)calloc(conv_weight_count, sizeof(float));
    context->conv_bias = (float*)calloc(config->filter_count, sizeof(float));
    context->projection_weights = (float*)calloc(projection_weight_count, sizeof(float));
    context->projection_bias = (float*)calloc(config->feature_size, sizeof(float));
    context->input_buffer = (float*)calloc(config->total_input_size, sizeof(float));
    context->output_buffer = (float*)calloc(config->sequence_length * config->feature_size, sizeof(float));
    context->pooled_values = (float*)calloc(pooled_feature_count, sizeof(float));

    if (context->conv_weights == NULL || context->conv_bias == NULL || context->projection_weights == NULL ||
        context->projection_bias == NULL || context->input_buffer == NULL || context->output_buffer == NULL ||
        context->pooled_values == NULL) {
        nn_cnn_dual_pool_infer_destroy(context);
        return NULL;
    }

    for (value_index = 0U; value_index < conv_weight_count; ++value_index) {
        context->conv_weights[value_index] = cnn_dual_pool_random_weight(&context->rng_state, 0.24f);
    }
    for (value_index = 0U; value_index < config->filter_count; ++value_index) {
        context->conv_bias[value_index] = cnn_dual_pool_random_weight(&context->rng_state, 0.05f);
    }
    for (value_index = 0U; value_index < projection_weight_count; ++value_index) {
        context->projection_weights[value_index] = cnn_dual_pool_random_weight(&context->rng_state, 0.18f);
    }
    for (value_index = 0U; value_index < config->feature_size; ++value_index) {
        context->projection_bias[value_index] = cnn_dual_pool_random_weight(&context->rng_state, 0.05f);
    }
    return context;
}

void nn_cnn_dual_pool_infer_destroy(void* ctx) {
    CnnDualPoolInferContext* context = (CnnDualPoolInferContext*)ctx;
    if (context == NULL) {
        return;
    }
    free(context->conv_weights);
    free(context->conv_bias);
    free(context->projection_weights);
    free(context->projection_bias);
    free(context->input_buffer);
    free(context->output_buffer);
    free(context->pooled_values);
    free(context);
}

void nn_cnn_dual_pool_infer_set_input(void* ctx, const float* input, size_t size) {
    CnnDualPoolInferContext* context = (CnnDualPoolInferContext*)ctx;
    if (context == NULL || input == NULL || size != context->config.total_input_size) {
        return;
    }
    (void)memcpy(context->input_buffer, input, size * sizeof(float));
}

void nn_cnn_dual_pool_infer_get_output(void* ctx, float* output, size_t size) {
    CnnDualPoolInferContext* context = (CnnDualPoolInferContext*)ctx;
    size_t expected_size;

    if (context == NULL || output == NULL) {
        return;
    }
    expected_size = context->config.sequence_length * context->config.feature_size;
    if (size != expected_size) {
        return;
    }
    (void)memcpy(output, context->output_buffer, size * sizeof(float));
}

int nn_cnn_dual_pool_forward_pass(CnnDualPoolInferContext* context, const float* input, float* output, float* pooled_linear_cache, float* pooled_activation_cache, size_t* max_index_cache, float* output_linear_cache) {
    const CnnDualPoolConfig* config;
    size_t frame_stride;
    size_t output_grid_width;
    size_t output_grid_height;
    size_t output_positions;
    size_t pooled_feature_count;
    size_t step_index;
    size_t filter_index;
    size_t feature_index;

    if (context == NULL || input == NULL || output == NULL) {
        return -1;
    }

    config = &context->config;
    frame_stride = config->frame_width * config->frame_height * config->channel_count;
    output_grid_width = config->frame_width - config->kernel_size + 1U;
    output_grid_height = config->frame_height - config->kernel_size + 1U;
    output_positions = cnn_dual_pool_conv_position_count(config);
    pooled_feature_count = cnn_dual_pool_pooled_feature_count(config);
    if (output_positions == 0U) {
        return -1;
    }

    for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
        const float* frame = input + (step_index * frame_stride);
        float* pooled_values = context->pooled_values;

        if (pooled_values == NULL) {
            return -1;
        }

        for (filter_index = 0U; filter_index < config->filter_count; ++filter_index) {
            float pooled_sum = 0.0f;
            float pooled_max = 0.0f;
            size_t pooled_max_index = 0U;
            int have_value = 0;
            size_t out_row;
            size_t out_column;
            size_t avg_index = (step_index * pooled_feature_count) + (filter_index * 2U);
            size_t max_index = avg_index + 1U;
            size_t position_counter = 0U;

            for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                    float conv_value = context->conv_bias[filter_index];
                    size_t channel_index;
                    for (channel_index = 0U; channel_index < config->channel_count; ++channel_index) {
                        size_t kernel_row;
                        for (kernel_row = 0U; kernel_row < config->kernel_size; ++kernel_row) {
                            size_t kernel_column;
                            for (kernel_column = 0U; kernel_column < config->kernel_size; ++kernel_column) {
                                size_t input_index = cnn_dual_pool_frame_index(config, channel_index, out_row + kernel_row, out_column + kernel_column);
                                size_t weight_index = cnn_dual_pool_kernel_index(config, filter_index, channel_index, kernel_row, kernel_column);
                                conv_value += frame[input_index] * context->conv_weights[weight_index];
                            }
                        }
                    }
                    pooled_sum += conv_value;
                    if (!have_value || conv_value > pooled_max) {
                        pooled_max = conv_value;
                        pooled_max_index = position_counter;
                        have_value = 1;
                    }
                    position_counter += 1U;
                }
            }

            pooled_values[filter_index * 2U] = cnn_dual_pool_apply_activation(pooled_sum / (float)output_positions, config->pooling_activation);
            pooled_values[(filter_index * 2U) + 1U] = cnn_dual_pool_apply_activation(pooled_max, config->pooling_activation);
            if (pooled_linear_cache != NULL) {
                pooled_linear_cache[avg_index] = pooled_sum / (float)output_positions;
                pooled_linear_cache[max_index] = pooled_max;
            }
            if (pooled_activation_cache != NULL) {
                pooled_activation_cache[avg_index] = pooled_values[filter_index * 2U];
                pooled_activation_cache[max_index] = pooled_values[(filter_index * 2U) + 1U];
            }
            if (max_index_cache != NULL) {
                max_index_cache[(step_index * config->filter_count) + filter_index] = pooled_max_index;
            }
        }

        for (feature_index = 0U; feature_index < config->feature_size; ++feature_index) {
            float linear_value = context->projection_bias[feature_index];
            size_t pooled_index;
            for (pooled_index = 0U; pooled_index < pooled_feature_count; ++pooled_index) {
                linear_value += context->projection_weights[(feature_index * pooled_feature_count) + pooled_index] * pooled_values[pooled_index];
            }
            if (output_linear_cache != NULL) {
                output_linear_cache[(step_index * config->feature_size) + feature_index] = linear_value;
            }
            output[(step_index * config->feature_size) + feature_index] = cnn_dual_pool_apply_activation(linear_value, config->output_activation);
        }
    }

    return 0;
}

int nn_cnn_dual_pool_infer_step(void* ctx) {
    CnnDualPoolInferContext* context = (CnnDualPoolInferContext*)ctx;
    if (context == NULL) {
        return -1;
    }
    return nn_cnn_dual_pool_forward_pass(context, context->input_buffer, context->output_buffer, NULL, NULL, NULL, NULL);
}

int nn_cnn_dual_pool_infer_auto_run(void* ctx, const float* input, float* output) {
    CnnDualPoolInferContext* context = (CnnDualPoolInferContext*)ctx;
    size_t output_size;

    if (context == NULL || input == NULL || output == NULL) {
        return -1;
    }
    output_size = context->config.sequence_length * context->config.feature_size;
    nn_cnn_dual_pool_infer_set_input(context, input, context->config.total_input_size);
    if (nn_cnn_dual_pool_infer_step(context) != 0) {
        return -1;
    }
    nn_cnn_dual_pool_infer_get_output(context, output, output_size);
    return 0;
}

int nn_cnn_dual_pool_load_weights(void* ctx, FILE* fp) {
    CnnDualPoolInferContext* context = (CnnDualPoolInferContext*)ctx;
    CnnDualPoolWeightHeader header;
    size_t conv_weight_count;
    size_t projection_weight_count;

    if (context == NULL || fp == NULL) {
        return 0;
    }
    if (fread(&header, sizeof(header), 1, fp) != 1U || header.abi_version != CNN_DUAL_POOL_ABI_VERSION) {
        return 0;
    }
    if (context->expected_network_hash != 0U && header.network_hash != context->expected_network_hash) {
        return 0;
    }
    if (context->expected_layout_hash != 0U && header.layout_hash != context->expected_layout_hash) {
        return 0;
    }
    if (header.sequence_length != context->config.sequence_length || header.frame_width != context->config.frame_width ||
        header.frame_height != context->config.frame_height || header.channel_count != context->config.channel_count ||
        header.kernel_size != context->config.kernel_size || header.filter_count != context->config.filter_count ||
        header.feature_size != context->config.feature_size) {
        return 0;
    }

    conv_weight_count = cnn_dual_pool_conv_weight_count(&context->config);
    projection_weight_count = cnn_dual_pool_projection_weight_count(&context->config);
    if (fread(context->conv_weights, sizeof(float), conv_weight_count, fp) != conv_weight_count) {
        return 0;
    }
    if (fread(context->conv_bias, sizeof(float), context->config.filter_count, fp) != context->config.filter_count) {
        return 0;
    }
    if (fread(context->projection_weights, sizeof(float), projection_weight_count, fp) != projection_weight_count) {
        return 0;
    }
    if (fread(context->projection_bias, sizeof(float), context->config.feature_size, fp) != context->config.feature_size) {
        return 0;
    }
    return 1;
}

int nn_cnn_dual_pool_save_weights(void* ctx, FILE* fp) {
    CnnDualPoolInferContext* context = (CnnDualPoolInferContext*)ctx;
    CnnDualPoolWeightHeader header;
    size_t conv_weight_count;
    size_t projection_weight_count;

    if (context == NULL || fp == NULL) {
        return 0;
    }

    header.network_hash = context->expected_network_hash != 0U ? context->expected_network_hash : cnn_dual_pool_compute_layout_hash(&context->config);
    header.layout_hash = context->expected_layout_hash != 0U ? context->expected_layout_hash : cnn_dual_pool_compute_layout_hash(&context->config);
    header.abi_version = CNN_DUAL_POOL_ABI_VERSION;
    header.sequence_length = (uint32_t)context->config.sequence_length;
    header.frame_width = (uint32_t)context->config.frame_width;
    header.frame_height = (uint32_t)context->config.frame_height;
    header.channel_count = (uint32_t)context->config.channel_count;
    header.kernel_size = (uint32_t)context->config.kernel_size;
    header.filter_count = (uint32_t)context->config.filter_count;
    header.feature_size = (uint32_t)context->config.feature_size;

    conv_weight_count = cnn_dual_pool_conv_weight_count(&context->config);
    projection_weight_count = cnn_dual_pool_projection_weight_count(&context->config);
    if (fwrite(&header, sizeof(header), 1, fp) != 1U) {
        return 0;
    }
    if (fwrite(context->conv_weights, sizeof(float), conv_weight_count, fp) != conv_weight_count) {
        return 0;
    }
    if (fwrite(context->conv_bias, sizeof(float), context->config.filter_count, fp) != context->config.filter_count) {
        return 0;
    }
    if (fwrite(context->projection_weights, sizeof(float), projection_weight_count, fp) != projection_weight_count) {
        return 0;
    }
    if (fwrite(context->projection_bias, sizeof(float), context->config.feature_size, fp) != context->config.feature_size) {
        return 0;
    }
    return 1;
}

uint64_t nn_cnn_dual_pool_get_network_hash(const void* ctx) {
    const CnnDualPoolInferContext* context = (const CnnDualPoolInferContext*)ctx;
    if (context == NULL) {
        return 0U;
    }
    return cnn_dual_pool_compute_layout_hash(&context->config);
}
