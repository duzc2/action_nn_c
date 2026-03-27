/**
 * @file cnn_infer_ops.c
 * @brief Tiny CNN inference backend used by generated graph code.
 *
 * The implementation is intentionally modest rather than numerically ambitious.
 * Its job is to expose a real convolution-style leaf that the profiler can
 * create, execute, and serialize inside nested graphs. Comments therefore focus
 * on lifecycle, tensor layout, and the reasons behind each stage.
 */

#include "cnn_infer_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define CNN_ABI_VERSION 1U

/**
 * @brief Serialized header written before CNN parameter arrays.
 */
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
} CnnWeightHeader;

/**
 * @brief Advance the tiny deterministic RNG used for parameter init.
 */
static uint32_t cnn_next_random(uint32_t* state) {
    uint32_t value = *state;
    value = value * 1664525U + 1013904223U;
    *state = value;
    return value;
}

/**
 * @brief Generate a small centered random weight.
 */
static float cnn_random_weight(uint32_t* state, float scale) {
    float normalized = (float)(cnn_next_random(state) & 0xFFFFU) / 65535.0f;
    return (normalized - 0.5f) * scale;
}

/**
 * @brief Apply the activation configured for one stage.
 */
static float cnn_apply_activation(float value, CnnActivationType activation) {
    switch (activation) {
        case CNN_ACT_RELU:
            return value > 0.0f ? value : 0.0f;
        case CNN_ACT_TANH:
            return tanhf(value);
        case CNN_ACT_NONE:
        default:
            return value;
    }
}

/**
 * @brief Validate that the type config stays within the backend's fixed bounds.
 */
static int cnn_config_is_valid(const CnnConfig* config) {
    size_t frame_stride;

    if (config == NULL) {
        return 0;
    }
    if (config->sequence_length == 0U ||
        config->sequence_length > CNN_MAX_SEQUENCE_LENGTH) {
        return 0;
    }
    if (config->frame_width == 0U || config->frame_width > CNN_MAX_FRAME_WIDTH ||
        config->frame_height == 0U || config->frame_height > CNN_MAX_FRAME_HEIGHT) {
        return 0;
    }
    if (config->channel_count == 0U || config->channel_count > CNN_MAX_CHANNEL_COUNT) {
        return 0;
    }
    if (config->kernel_size == 0U || config->kernel_size > CNN_MAX_KERNEL_SIZE) {
        return 0;
    }
    if (config->kernel_size > config->frame_width ||
        config->kernel_size > config->frame_height) {
        return 0;
    }
    if (config->filter_count == 0U || config->filter_count > CNN_MAX_FILTER_COUNT) {
        return 0;
    }
    if (config->feature_size == 0U || config->feature_size > CNN_MAX_FEATURE_COUNT) {
        return 0;
    }

    frame_stride = config->frame_width * config->frame_height * config->channel_count;
    if (frame_stride == 0U) {
        return 0;
    }
    if (config->total_input_size != frame_stride * config->sequence_length) {
        return 0;
    }
    if (config->total_input_size > CNN_MAX_TOTAL_INPUT_SIZE) {
        return 0;
    }

    return 1;
}

/**
 * @brief Count valid convolution windows in one frame.
 */
static size_t cnn_conv_position_count(const CnnConfig* config) {
    return (config->frame_height - config->kernel_size + 1U) *
        (config->frame_width - config->kernel_size + 1U);
}

/**
 * @brief Compute a compact structural hash from the active CNN config.
 */
static uint64_t cnn_compute_layout_hash(const CnnConfig* config) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    const uint64_t prime = 0x100000001b3ULL;

    if (config == NULL) {
        return hash;
    }

    hash ^= (uint64_t)config->total_input_size;
    hash *= prime;
    hash ^= (uint64_t)config->sequence_length;
    hash *= prime;
    hash ^= (uint64_t)config->frame_width;
    hash *= prime;
    hash ^= (uint64_t)config->frame_height;
    hash *= prime;
    hash ^= (uint64_t)config->channel_count;
    hash *= prime;
    hash ^= (uint64_t)config->kernel_size;
    hash *= prime;
    hash ^= (uint64_t)config->filter_count;
    hash *= prime;
    hash ^= (uint64_t)config->feature_size;
    hash *= prime;
    hash ^= (uint64_t)config->pooling_activation;
    hash *= prime;
    hash ^= (uint64_t)config->output_activation;
    hash *= prime;

    return hash;
}

/**
 * @brief Flatten a frame coordinate into the input tensor layout.
 */
static size_t cnn_frame_index(const CnnConfig* config, size_t channel, size_t row, size_t column) {
    return ((channel * config->frame_height) + row) * config->frame_width + column;
}

/**
 * @brief Flatten a convolution-weight coordinate into the parameter tensor.
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
 * @brief Compute the total number of convolution weights owned by the context.
 */
static size_t cnn_conv_weight_count(const CnnConfig* config) {
    return config->filter_count * config->channel_count * config->kernel_size * config->kernel_size;
}

/**
 * @brief Compute the total number of projection weights owned by the context.
 */
static size_t cnn_projection_weight_count(const CnnConfig* config) {
    return config->feature_size * config->filter_count;
}

/**
 * @brief Deliberately reject implicit construction without typed config metadata.
 */
CnnInferContext* nn_cnn_infer_create(void) {
    return NULL;
}

/**
 * @brief Allocate and initialize one typed CNN inference context.
 */
CnnInferContext* nn_cnn_infer_create_with_config(const CnnConfig* config, uint32_t seed) {
    CnnInferContext* context;
    size_t conv_weight_count;
    size_t projection_weight_count;
    size_t value_index;

    if (!cnn_config_is_valid(config)) {
        return NULL;
    }

    context = (CnnInferContext*)calloc(1U, sizeof(CnnInferContext));
    if (context == NULL) {
        return NULL;
    }

    context->config = *config;
    context->rng_state = seed != 0U ? seed : (config->seed != 0U ? config->seed : 1U);
    conv_weight_count = cnn_conv_weight_count(config);
    projection_weight_count = cnn_projection_weight_count(config);

    context->conv_weights = (float*)calloc(conv_weight_count, sizeof(float));
    context->conv_bias = (float*)calloc(config->filter_count, sizeof(float));
    context->projection_weights = (float*)calloc(projection_weight_count, sizeof(float));
    context->projection_bias = (float*)calloc(config->feature_size, sizeof(float));
    context->input_buffer = (float*)calloc(config->total_input_size, sizeof(float));
    context->output_buffer = (float*)calloc(
        config->sequence_length * config->feature_size,
        sizeof(float)
    );

    if (context->conv_weights == NULL || context->conv_bias == NULL ||
        context->projection_weights == NULL || context->projection_bias == NULL ||
        context->input_buffer == NULL || context->output_buffer == NULL) {
        nn_cnn_infer_destroy(context);
        return NULL;
    }

    /* Initialize every learnable tensor deterministically so demos stay repeatable. */
    for (value_index = 0U; value_index < conv_weight_count; ++value_index) {
        context->conv_weights[value_index] = cnn_random_weight(&context->rng_state, 0.24f);
    }
    for (value_index = 0U; value_index < config->filter_count; ++value_index) {
        context->conv_bias[value_index] = cnn_random_weight(&context->rng_state, 0.05f);
    }
    for (value_index = 0U; value_index < projection_weight_count; ++value_index) {
        context->projection_weights[value_index] = cnn_random_weight(&context->rng_state, 0.18f);
    }
    for (value_index = 0U; value_index < config->feature_size; ++value_index) {
        context->projection_bias[value_index] = cnn_random_weight(&context->rng_state, 0.05f);
    }

    return context;
}

/**
 * @brief Free every owned CNN resource in the reverse order of construction.
 */
void nn_cnn_infer_destroy(void* ctx) {
    CnnInferContext* context = (CnnInferContext*)ctx;

    if (context == NULL) {
        return;
    }

    free(context->conv_weights);
    free(context->conv_bias);
    free(context->projection_weights);
    free(context->projection_bias);
    free(context->input_buffer);
    free(context->output_buffer);
    free(context);
}

/**
 * @brief Copy caller input into the owned inference buffer.
 */
void nn_cnn_infer_set_input(void* ctx, const float* input, size_t size) {
    CnnInferContext* context = (CnnInferContext*)ctx;

    if (context == NULL || input == NULL || size != context->config.total_input_size) {
        return;
    }

    (void)memcpy(context->input_buffer, input, size * sizeof(float));
}

/**
 * @brief Copy the latest output into caller-owned storage.
 */
void nn_cnn_infer_get_output(void* ctx, float* output, size_t size) {
    CnnInferContext* context = (CnnInferContext*)ctx;
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

/**
 * @brief Run the CNN forward pipeline with optional training caches.
 */
int nn_cnn_forward_pass(
    CnnInferContext* context,
    const float* input,
    float* output,
    float* pooled_linear_cache,
    float* pooled_activation_cache,
    float* output_linear_cache
) {
    const CnnConfig* config;
    size_t frame_stride;
    size_t output_grid_width;
    size_t output_grid_height;
    size_t output_positions;
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
    output_positions = cnn_conv_position_count(config);
    if (output_positions == 0U) {
        return -1;
    }

    /* Each step reuses the same filters so the CNN acts as a shared frame encoder. */
    for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
        const float* frame = input + (step_index * frame_stride);
        float pooled_values[CNN_MAX_FILTER_COUNT];

        for (filter_index = 0U; filter_index < config->filter_count; ++filter_index) {
            float pooled_linear = 0.0f;
            float pooled_activation;
            size_t out_row;
            size_t out_column;

            /* Convolve one filter over the full spatial field, then average positions. */
            for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                    float conv_value = context->conv_bias[filter_index];
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
                                conv_value += frame[input_index] * context->conv_weights[weight_index];
                            }
                        }
                    }

                    pooled_linear += conv_value;
                }
            }

            pooled_linear /= (float)output_positions;
            pooled_activation = cnn_apply_activation(pooled_linear, config->pooling_activation);
            pooled_values[filter_index] = pooled_activation;
            if (pooled_linear_cache != NULL) {
                pooled_linear_cache[(step_index * config->filter_count) + filter_index] = pooled_linear;
            }
            if (pooled_activation_cache != NULL) {
                pooled_activation_cache[(step_index * config->filter_count) + filter_index] =
                    pooled_activation;
            }
        }

        /* Project pooled filter responses into a compact feature vector for the RNN leaf. */
        for (feature_index = 0U; feature_index < config->feature_size; ++feature_index) {
            float linear_value = context->projection_bias[feature_index];
            size_t local_filter_index;

            for (local_filter_index = 0U; local_filter_index < config->filter_count; ++local_filter_index) {
                linear_value += context->projection_weights[
                    (feature_index * config->filter_count) + local_filter_index
                ] * pooled_values[local_filter_index];
            }

            if (output_linear_cache != NULL) {
                output_linear_cache[(step_index * config->feature_size) + feature_index] = linear_value;
            }
            output[(step_index * config->feature_size) + feature_index] =
                cnn_apply_activation(linear_value, config->output_activation);
        }
    }

    return 0;
}

/**
 * @brief Execute one inference step using the context-owned input and output buffers.
 */
int nn_cnn_infer_step(void* ctx) {
    CnnInferContext* context = (CnnInferContext*)ctx;

    if (context == NULL) {
        return -1;
    }

    return nn_cnn_forward_pass(
        context,
        context->input_buffer,
        context->output_buffer,
        NULL,
        NULL,
        NULL
    );
}

/**
 * @brief Convenience wrapper that performs input copy, forward pass, and output copy.
 */
int nn_cnn_infer_auto_run(void* ctx, const float* input, float* output) {
    CnnInferContext* context = (CnnInferContext*)ctx;
    size_t output_size;

    if (context == NULL || input == NULL || output == NULL) {
        return -1;
    }

    output_size = context->config.sequence_length * context->config.feature_size;
    nn_cnn_infer_set_input(context, input, context->config.total_input_size);
    if (nn_cnn_infer_step(context) != 0) {
        return -1;
    }
    nn_cnn_infer_get_output(context, output, output_size);
    return 0;
}

/**
 * @brief Load CNN parameters after validating topology and ABI metadata.
 */
int nn_cnn_load_weights(void* ctx, FILE* fp) {
    CnnInferContext* context = (CnnInferContext*)ctx;
    CnnWeightHeader header;
    size_t conv_weight_count;
    size_t projection_weight_count;

    if (context == NULL || fp == NULL) {
        return 0;
    }
    if (fread(&header, sizeof(header), 1, fp) != 1U) {
        return 0;
    }
    if (header.abi_version != CNN_ABI_VERSION) {
        return 0;
    }
    if (context->expected_network_hash != 0U && header.network_hash != context->expected_network_hash) {
        return 0;
    }
    if (context->expected_layout_hash != 0U && header.layout_hash != context->expected_layout_hash) {
        return 0;
    }
    if (header.sequence_length != context->config.sequence_length ||
        header.frame_width != context->config.frame_width ||
        header.frame_height != context->config.frame_height ||
        header.channel_count != context->config.channel_count ||
        header.kernel_size != context->config.kernel_size ||
        header.filter_count != context->config.filter_count ||
        header.feature_size != context->config.feature_size) {
        return 0;
    }

    conv_weight_count = cnn_conv_weight_count(&context->config);
    projection_weight_count = cnn_projection_weight_count(&context->config);

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

/**
 * @brief Save CNN parameters together with compatibility metadata.
 */
int nn_cnn_save_weights(void* ctx, FILE* fp) {
    CnnInferContext* context = (CnnInferContext*)ctx;
    CnnWeightHeader header;
    size_t conv_weight_count;
    size_t projection_weight_count;

    if (context == NULL || fp == NULL) {
        return 0;
    }

    header.network_hash = context->expected_network_hash != 0U ?
        context->expected_network_hash : cnn_compute_layout_hash(&context->config);
    header.layout_hash = context->expected_layout_hash != 0U ?
        context->expected_layout_hash : cnn_compute_layout_hash(&context->config);
    header.abi_version = CNN_ABI_VERSION;
    header.sequence_length = (uint32_t)context->config.sequence_length;
    header.frame_width = (uint32_t)context->config.frame_width;
    header.frame_height = (uint32_t)context->config.frame_height;
    header.channel_count = (uint32_t)context->config.channel_count;
    header.kernel_size = (uint32_t)context->config.kernel_size;
    header.filter_count = (uint32_t)context->config.filter_count;
    header.feature_size = (uint32_t)context->config.feature_size;

    conv_weight_count = cnn_conv_weight_count(&context->config);
    projection_weight_count = cnn_projection_weight_count(&context->config);

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

/**
 * @brief Expose the current structural hash to persistence callers.
 */
uint64_t nn_cnn_get_network_hash(const void* ctx) {
    const CnnInferContext* context = (const CnnInferContext*)ctx;

    if (context == NULL) {
        return 0U;
    }

    return cnn_compute_layout_hash(&context->config);
}
