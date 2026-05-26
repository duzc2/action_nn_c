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
#include "../../../utils/error.h"

#define CNN_ABI_VERSION 2U

/* Diagnostic: forward pass call counter (used for [FWD] log step tracking) */
static size_t fwd_call_counter = 0;

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
    uint32_t stride;
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
    float normalized = (float)cnn_next_random(state) / 4294967295.0f;
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
        case CNN_ACT_RELU6:
            return value > 6.0f ? 6.0f : (value > 0.0f ? value : 0.0f);
        case CNN_ACT_LEAKY_RELU:
            return value > 0.0f ? value : 0.01f * value;
        case CNN_ACT_NONE:
        default:
            return value;
    }
}

/**
 * @brief Validate that the type config is structurally coherent.
 */
static int cnn_config_is_valid(const CnnConfig* config) {
    size_t frame_stride;

    if (config == NULL) {
        return 0;
    }
    if (config->sequence_length == 0U) {
        return 0;
    }
    if (config->frame_width == 0U || config->frame_height == 0U) {
        return 0;
    }
    if (config->channel_count == 0U) {
        return 0;
    }
    if (config->kernel_size == 0U) {
        return 0;
    }
    if (config->kernel_size > config->frame_width ||
        config->kernel_size > config->frame_height) {
        return 0;
    }
    if (config->filter_count == 0U) {
        return 0;
    }
    /* feature_size is unused when pooling is disabled (no projection layer). */
    if (config->pooling_mode != CNN_POOL_NONE && config->feature_size == 0U) {
        return 0;
    }
    /* depthwise conv requires filter_count == channel_count */
    if (config->conv_mode == CNN_CONV_DEPTHWISE &&
        config->filter_count != config->channel_count) {
        return 0;
    }
    if (config->stride == 0U) {
        return 0;
    }
    if (config->use_batch_norm && config->bn_epsilon < 0.0f) {
        return 0;
    }

    frame_stride = config->frame_width * config->frame_height * config->channel_count;
    if (frame_stride == 0U) {
        return 0;
    }
    if (config->total_input_size != frame_stride * config->sequence_length) {
        return 0;
    }

    return 1;
}

#include "cnn_common.h"

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
    hash ^= (uint64_t)config->pooling_mode;
    hash *= prime;
    hash ^= (uint64_t)config->conv_mode;
    hash *= prime;
    hash ^= (uint64_t)config->stride;
    hash *= prime;
    hash ^= (uint64_t)config->use_batch_norm;
    hash *= prime;

    return hash;
}


/**
 * @brief Compute the total output size when pooling_mode == CNN_POOL_NONE.
 *
 * Without pooling or projection the output is the raw 2-D conv activation map,
 * laid out as [step][filter][row][col] — the same interleaved-channel format
 * that cnn_frame_index() uses for input, so a downstream CNN can consume it
 * directly by setting channel_count = filter_count.
 */
static size_t cnn_none_output_size(const CnnConfig* config) {
    size_t grid_h = (config->frame_height - config->kernel_size) / config->stride + 1U;
    size_t grid_w = (config->frame_width - config->kernel_size) / config->stride + 1U;
    return config->sequence_length * config->filter_count * grid_h * grid_w;
}


/**
 * @brief Allocate and initialize one typed CNN inference context.
 */
CnnInferContext* nn_cnn_infer_create_with_config(const CnnConfig* config, uint32_t seed) {
    CnnInferContext* context;
    size_t conv_weight_count;
    size_t projection_weight_count;
    size_t pooled_value_count;
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
    pooled_value_count = cnn_pooled_value_count(config);
    projection_weight_count = pooled_value_count > 0U ? config->feature_size * pooled_value_count : 0U;

    context->conv_weights = (float*)calloc(conv_weight_count, sizeof(float));
    context->conv_bias = (float*)calloc(config->filter_count, sizeof(float));
    context->input_buffer = (float*)calloc(config->total_input_size, sizeof(float));

    if (config->pooling_mode == CNN_POOL_NONE) {
        /* No pooling or projection — output is the raw 2-D conv activation map. */
        context->projection_weights = NULL;
        context->projection_bias = NULL;
        context->pooled_values = NULL;
        context->max_index_cache = NULL;
        context->output_buffer = (float*)calloc(cnn_none_output_size(config), sizeof(float));
    } else {
        context->projection_weights = (float*)calloc(projection_weight_count, sizeof(float));
        context->projection_bias = (float*)calloc(config->feature_size, sizeof(float));
        context->output_buffer = (float*)calloc(
            config->sequence_length * config->feature_size, sizeof(float));
        context->pooled_values = (float*)calloc(pooled_value_count, sizeof(float));
        if (config->pooling_mode != CNN_POOL_AVG) {
            context->max_index_cache = (size_t*)calloc(config->filter_count, sizeof(size_t));
        } else {
            context->max_index_cache = NULL;
        }
    }

    /* Batch Normalization buffers: per-filter running stats and learnable params */
    if (config->use_batch_norm) {
        size_t fi;
        context->bn_running_mean = (float*)calloc(config->filter_count, sizeof(float));
        context->bn_running_var  = (float*)calloc(config->filter_count, sizeof(float));
        context->bn_gamma        = (float*)calloc(config->filter_count, sizeof(float));
        context->bn_beta         = (float*)calloc(config->filter_count, sizeof(float));
        context->bn_training_pre_cache   = NULL;
        context->bn_training_spatial_var = NULL;
        if (context->bn_running_mean == NULL || context->bn_running_var == NULL ||
            context->bn_gamma == NULL || context->bn_beta == NULL) {
            nn_cnn_infer_destroy(context);
            return NULL;
        }
        for (fi = 0U; fi < config->filter_count; ++fi) {
            context->bn_gamma[fi] = 1.0f;
            context->bn_running_var[fi] = 1.0f;
        }
    } else {
        context->bn_running_mean = NULL;
        context->bn_running_var  = NULL;
        context->bn_gamma        = NULL;
        context->bn_beta         = NULL;
        context->bn_training_pre_cache   = NULL;
        context->bn_training_spatial_var = NULL;
    }

    if (context->conv_weights == NULL || context->conv_bias == NULL ||
        context->input_buffer == NULL || context->output_buffer == NULL) {
        nn_cnn_infer_destroy(context);
        return NULL;
    }
    if (config->pooling_mode != CNN_POOL_NONE) {
        if (context->projection_weights == NULL || context->projection_bias == NULL ||
            context->pooled_values == NULL) {
            nn_cnn_infer_destroy(context);
            return NULL;
        }
    }

    /* ── He (Kaiming) uniform initialization ──
     * cnn_random_weight(state, scale) returns uniform samples in [-scale/2, scale/2].
     * He uniform wants U(-s, s) with s = sqrt(6 / fan_in).
     * Therefore we pass scale = 2 * sqrt(6 / fan_in). */
    {
        float fan_in;
        float he_conv_scale;

        if (config->conv_mode == CNN_CONV_DEPTHWISE) {
            fan_in = (float)(config->kernel_size * config->kernel_size);
        } else {
            fan_in = (float)(config->channel_count * config->kernel_size * config->kernel_size);
        }
        he_conv_scale = 2.0f * sqrtf(6.0f / (fan_in > 0.0f ? fan_in : 1.0f));

        for (value_index = 0U; value_index < conv_weight_count; ++value_index) {
            context->conv_weights[value_index] = cnn_random_weight(&context->rng_state, he_conv_scale);
        }
    }
    /* Bias: small uniform init for numerical stability */
    for (value_index = 0U; value_index < config->filter_count; ++value_index) {
        context->conv_bias[value_index] = cnn_random_weight(&context->rng_state, 0.02f);
    }
    /* Projection weights: He uniform based on pooled value count (fan_in) */
    if (config->pooling_mode != CNN_POOL_NONE && pooled_value_count > 0U) {
        float proj_fan_in = (float)pooled_value_count;
        float he_proj_scale = 2.0f * sqrtf(6.0f / proj_fan_in);

        for (value_index = 0U; value_index < projection_weight_count; ++value_index) {
            context->projection_weights[value_index] = cnn_random_weight(&context->rng_state, he_proj_scale);
        }
        for (value_index = 0U; value_index < config->feature_size; ++value_index) {
            context->projection_bias[value_index] = cnn_random_weight(&context->rng_state, 0.02f);
        }
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
    free(context->pooled_values);
    free(context->max_index_cache);
    free(context->bn_running_mean);
    free(context->bn_running_var);
    free(context->bn_gamma);
    free(context->bn_beta);
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

    expected_size = (context->config.pooling_mode == CNN_POOL_NONE) ?
        cnn_none_output_size(&context->config) :
        (context->config.sequence_length * context->config.feature_size);
    if (size != expected_size) {
        return;
    }

    (void)memcpy(output, context->output_buffer, size * sizeof(float));
}

/**
 * @brief Run the CNN forward pipeline with optional training caches.
 */
int nn_cnn_forward_pass(
    CnnInferContext* restrict context,
    const float* restrict input,
    float* restrict output,
    float* restrict pooled_linear_cache,
    float* restrict pooled_activation_cache,
    size_t* restrict max_index_cache,
    float* restrict output_linear_cache,
    float* restrict bn_pre_cache,
    float* restrict bn_spatial_var,
    float dropout_rate,
    float* restrict dropout_mask
) {
    const CnnConfig* config;
    size_t frame_stride;
    size_t output_grid_width;
    size_t output_grid_height;
    size_t output_positions;
    size_t pooled_value_count;
    size_t step_index;
    size_t filter_index;
    size_t feature_index;

    if (context == NULL || input == NULL || output == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    fwd_call_counter++;

    config = &context->config;
    frame_stride = config->frame_width * config->frame_height * config->channel_count;
    output_grid_width = (config->frame_width - config->kernel_size) / config->stride + 1U;
    output_grid_height = (config->frame_height - config->kernel_size) / config->stride + 1U;
    output_positions = cnn_conv_position_count(config);
    pooled_value_count = (config->pooling_mode == CNN_POOL_DUAL) ?
        (config->filter_count * 2U) : config->filter_count;
    if (output_positions == 0U) {
        return ACTION_C_ERR_DIM_MISMATCH;
    }

    /* Each step reuses the same filters so the CNN acts as a shared frame encoder. */
    for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
        const float* frame = input + (step_index * frame_stride);

        if (config->pooling_mode == CNN_POOL_NONE) {
            /* POOL_NONE: raw conv2d with activation, no pooling or projection.
             * Output layout per step: [filter][row][col] — interleaved-channel
             * compatible with cnn_frame_index() for downstream CNN cascading.
             *
             * Training mode (bn_pre_cache != NULL): two-pass spatial BN.
             * Inference mode (bn_pre_cache == NULL): single-pass with running stats. */
            size_t grid_plane = output_grid_height * output_grid_width;
            size_t output_stride = config->filter_count * grid_plane;
            float* step_output = output + (step_index * output_stride);
            size_t kernel_plane = config->kernel_size * config->kernel_size;
            int is_dw = (config->conv_mode == CNN_CONV_DEPTHWISE) ? 1 : 0;
            int has_bn = (config->use_batch_norm && context->bn_gamma != NULL) ? 1 : 0;
            int training_mode = (has_bn && bn_pre_cache != NULL) ? 1 : 0;

            for (filter_index = 0U; filter_index < config->filter_count; ++filter_index) {
                size_t out_row;
                size_t out_column;
                float* filter_cache = bn_pre_cache +
                    (step_index * config->filter_count * grid_plane) +
                    (filter_index * grid_plane);

                if (training_mode) {
                    /* ── Training: conv + running-statistics BN ──
                     * Pass 1: compute all conv values, accumulate spatial stats
                     *         for running statistic EMA updates.
                     * Pass 2: normalize using RUNNING statistics (not per-sample
                     *         spatial stats) — preserves per-sample variation
                     *         which is critical for batch_size=1 classification. */
                    float spatial_sum = 0.0f;
                    float spatial_sum_sq = 0.0f;
                    size_t pos = 0U;

                    for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                        for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                            float conv_value = context->conv_bias[filter_index];
                            size_t channel_index;
                            for (channel_index = 0U; channel_index < config->channel_count; ++channel_index) {
                                size_t kernel_row;
                                if (is_dw && channel_index != filter_index) continue;
                                for (kernel_row = 0U; kernel_row < config->kernel_size; ++kernel_row) {
                                    size_t kernel_column;
                                    for (kernel_column = 0U; kernel_column < config->kernel_size; ++kernel_column) {
                                        size_t input_index = cnn_frame_index(
                                            config, channel_index,
                                            out_row * config->stride + kernel_row,
                                            out_column * config->stride + kernel_column);
                                        size_t weight_index;
                                        if (is_dw) {
                                            weight_index = filter_index * kernel_plane +
                                                kernel_row * config->kernel_size + kernel_column;
                                        } else {
                                            weight_index = cnn_kernel_index(
                                                config, filter_index, channel_index, kernel_row, kernel_column);
                                        }
                                        conv_value += frame[input_index] * context->conv_weights[weight_index];
                                    }
                                }
                            }
                            filter_cache[pos] = conv_value;
                            spatial_sum += conv_value;
                            spatial_sum_sq += conv_value * conv_value;
                            pos++;
                        }
                    }

                    /* Compute per-sample spatial stats for running statistic update.
                     * We use RUNNING mean/var for normalization (not spatial stats)
                     * to preserve per-sample variation. The spatial stats are only
                     * used to update the EMA running estimates. */
                    {
                        float inv_n = 1.0f / (float)output_positions;
                        float spatial_mean = spatial_sum * inv_n;
                        float spatial_var = spatial_sum_sq * inv_n - spatial_mean * spatial_mean;
                        if (spatial_var < 0.0f) spatial_var = 0.0f;
                        float momentum = config->bn_momentum;

                        /* Store spatial variance (for pooled-path backward compatibility) */
                        if (bn_spatial_var != NULL) {
                            bn_spatial_var[step_index * config->filter_count + filter_index] = spatial_var;
                        }

                        /* Update running statistics via EMA with per-sample spatial stats */
                        context->bn_running_mean[filter_index] =
                            momentum * context->bn_running_mean[filter_index] +
                            (1.0f - momentum) * spatial_mean;
                        context->bn_running_var[filter_index] =
                            momentum * context->bn_running_var[filter_index] +
                            (1.0f - momentum) * spatial_var;

                        /* Pass 2: normalize using RUNNING statistics (global averages),
                         * store x_hat for backward pass, apply BN + activation */
                        {
                            float gamma = context->bn_gamma[filter_index];
                            float beta  = context->bn_beta[filter_index];
                            float bn_mean = context->bn_running_mean[filter_index];
                            float bn_var  = context->bn_running_var[filter_index];
                            float var_eps = bn_var + config->bn_epsilon;
                            float inv_std = 1.0f / sqrtf(var_eps > 0.0f ? var_eps : config->bn_epsilon);

                            for (pos = 0U; pos < output_positions; ++pos) {
                                float x_hat = (filter_cache[pos] - bn_mean) * inv_std;
                                filter_cache[pos] = x_hat;  /* store x_hat for backward pass */
                                float bn_out = gamma * x_hat + beta;
                                step_output[((filter_index * output_grid_height) +
                                    (pos / output_grid_width)) * output_grid_width +
                                    (pos % output_grid_width)] =
                                    cnn_apply_activation(bn_out, config->output_activation);
                            }
                        }
                    }
                } else {
                    /* ── Inference: single-pass with running statistics ── */
                    for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                        for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                            float conv_value = context->conv_bias[filter_index];
                            size_t channel_index;
                            for (channel_index = 0U; channel_index < config->channel_count; ++channel_index) {
                                size_t kernel_row;
                                if (is_dw && channel_index != filter_index) continue;
                                for (kernel_row = 0U; kernel_row < config->kernel_size; ++kernel_row) {
                                    size_t kernel_column;
                                    for (kernel_column = 0U; kernel_column < config->kernel_size; ++kernel_column) {
                                        size_t input_index = cnn_frame_index(
                                            config, channel_index,
                                            out_row * config->stride + kernel_row,
                                            out_column * config->stride + kernel_column);
                                        size_t weight_index;
                                        if (is_dw) {
                                            weight_index = filter_index * kernel_plane +
                                                kernel_row * config->kernel_size + kernel_column;
                                        } else {
                                            weight_index = cnn_kernel_index(
                                                config, filter_index, channel_index, kernel_row, kernel_column);
                                        }
                                        conv_value += frame[input_index] * context->conv_weights[weight_index];
                                    }
                                }
                            }
                            if (has_bn) {
                                float bn_mean = context->bn_running_mean[filter_index];
                                float bn_var  = context->bn_running_var[filter_index];
                                float bn_eps  = config->bn_epsilon;
                                float x_hat   = (conv_value - bn_mean) / sqrtf((bn_var + bn_eps) > 0.0f ? (bn_var + bn_eps) : bn_eps);
                                conv_value    = context->bn_gamma[filter_index] * x_hat +
                                                context->bn_beta[filter_index];
                            }
                            step_output[((filter_index * output_grid_height) + out_row) * output_grid_width + out_column] =
                                cnn_apply_activation(conv_value, config->output_activation);
                        }
                    }
                }
            }
            continue;
        }

        float* pooled_values = context->pooled_values;

        if (pooled_values == NULL) {
            return ACTION_C_ERR_NULL_POINTER;
        }

        for (filter_index = 0U; filter_index < config->filter_count; ++filter_index) {
            size_t out_row;
            size_t out_column;
            size_t pooled_avg_index;
            size_t pooled_max_index;

            if (config->pooling_mode == CNN_POOL_AVG) {
                /* Single average pooling. */
                float pooled_sum = 0.0f;
                float pooled_activation;
                int is_dw = (config->conv_mode == CNN_CONV_DEPTHWISE) ? 1 : 0;
                size_t kernel_plane = config->kernel_size * config->kernel_size;

                for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                    for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                        float conv_value = context->conv_bias[filter_index];
                        size_t channel_index;
                        for (channel_index = 0U; channel_index < config->channel_count; ++channel_index) {
                            size_t kernel_row;
                            if (is_dw && channel_index != filter_index) continue;
                            for (kernel_row = 0U; kernel_row < config->kernel_size; ++kernel_row) {
                                size_t kernel_column;
                                for (kernel_column = 0U; kernel_column < config->kernel_size; ++kernel_column) {
                                    size_t input_index = cnn_frame_index(
                                        config, channel_index,
                                        out_row * config->stride + kernel_row,
                                        out_column * config->stride + kernel_column);
                                    size_t weight_index;
                                    if (is_dw) {
                                        weight_index = filter_index * kernel_plane +
                                            kernel_row * config->kernel_size + kernel_column;
                                    } else {
                                        weight_index = cnn_kernel_index(
                                            config, filter_index, channel_index, kernel_row, kernel_column);
                                    }
                                    conv_value += frame[input_index] * context->conv_weights[weight_index];
                                }
                            }
                        }
                        pooled_sum += conv_value;
                    }
                }

                pooled_sum /= (float)output_positions;
                /* Apply BN to pooled value before activation */
                if (config->use_batch_norm && context->bn_gamma != NULL) {
                    float bn_mean = context->bn_running_mean[filter_index];
                    float bn_var  = context->bn_running_var[filter_index];
                    float bn_eps  = config->bn_epsilon;
                    float pooled_linear_raw = pooled_sum;

                    /* Compute normalization and cache x_hat + inv_std BEFORE the
                     * EMA running-stat update.  This fixes Bug #5: the backward
                     * pass now reads these cached values so forward and backward
                     * use strictly identical BN parameters — no more timing
                     * mismatch between pre-EMA forward norm and post-EMA backward
                     * reconstruction. */
                    float inv_std = 1.0f / sqrtf((bn_var + bn_eps) > 0.0f ? (bn_var + bn_eps) : bn_eps);
                    float x_hat = (pooled_sum - bn_mean) * inv_std;

                    /* Cache x_hat and inv_std for backward pass (pre-EMA values) */
                    if (bn_pre_cache != NULL) {
                        size_t cache_idx = (step_index * config->filter_count) + filter_index;
                        bn_pre_cache[cache_idx] = x_hat;
                        if (bn_spatial_var != NULL) {
                            bn_spatial_var[cache_idx] = inv_std;
                        }
                    }

                    float gamma_fwd = context->bn_gamma[filter_index];
                    float beta_fwd  = context->bn_beta[filter_index];
                    float bn_out_fwd = gamma_fwd * x_hat + beta_fwd;

                    /* EMA update — running statistics track activation distribution.
                     * Performed AFTER caching x_hat/inv_std so the backward pass
                     * always reads the normalization values that were actually used. */
                    if (bn_pre_cache != NULL) {
                        float mom = config->bn_momentum;
                        float old_mean = bn_mean;
                        context->bn_running_mean[filter_index] =
                            mom * old_mean + (1.0f - mom) * pooled_linear_raw;
                        float delta = pooled_linear_raw - old_mean;
                        context->bn_running_var[filter_index] =
                            mom * bn_var +
                            (1.0f - mom) * delta * delta;
                    }

                    /* [FWD] Point A: log normalization parameters used (pre-EMA snapshot) */
                    if (context->debug_level >= 2 && filter_index < 4U) {
                        fprintf(stderr, "[FWD] layer=%d filter=%zu step=%zu raw=%.6f bn_mean(for_norm)=%.6f bn_var(for_norm)=%.6f x_hat=%.6f gamma=%.6f beta=%.6f bn_out=%.6f\n",
                                context->debug_layer_index, filter_index, fwd_call_counter,
                                (double)pooled_linear_raw, (double)bn_mean, (double)bn_var,
                                (double)x_hat, (double)gamma_fwd, (double)beta_fwd,
                                (double)bn_out_fwd);
                    }
                    /* [FWD] Point B: log running stats after EMA update */
                    if (context->debug_level >= 2 && filter_index < 4U) {
                        fprintf(stderr, "[FWD] layer=%d filter=%zu step=%zu bn_mean(after_ema)=%.6f bn_var(after_ema)=%.6f\n",
                                context->debug_layer_index, filter_index, fwd_call_counter,
                                (double)context->bn_running_mean[filter_index],
                                (double)context->bn_running_var[filter_index]);
                    }

                    pooled_sum = bn_out_fwd;
                    /* In training mode store pre-BN raw value for batch statistics */
                    if (pooled_linear_cache != NULL) {
                        pooled_linear_cache[(step_index * config->filter_count) + filter_index] =
                            (bn_pre_cache != NULL) ? pooled_linear_raw : pooled_sum;
                    }
                } else {
                    if (pooled_linear_cache != NULL) {
                        pooled_linear_cache[(step_index * config->filter_count) + filter_index] = pooled_sum;
                    }
                }
                pooled_activation = cnn_apply_activation(pooled_sum, config->pooling_activation);
                pooled_values[filter_index] = pooled_activation;
                if (pooled_activation_cache != NULL) {
                    pooled_activation_cache[(step_index * config->filter_count) + filter_index] = pooled_activation;
                }
            } else {
                /* MAX or DUAL pooling — track both avg and max within the same convolution loops. */
                float pooled_sum = 0.0f;
                float pooled_max = 0.0f;
                size_t pooled_max_pos = 0U;
                size_t position_counter = 0U;
                int have_max = 0;
                int is_dw = (config->conv_mode == CNN_CONV_DEPTHWISE) ? 1 : 0;
                size_t kernel_plane = config->kernel_size * config->kernel_size;

                pooled_avg_index = (step_index * pooled_value_count) + (filter_index * 2U);
                pooled_max_index = pooled_avg_index + 1U;

                for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                    for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                        float conv_value = context->conv_bias[filter_index];
                        size_t channel_index;
                        for (channel_index = 0U; channel_index < config->channel_count; ++channel_index) {
                            size_t kernel_row;
                            if (is_dw && channel_index != filter_index) continue;
                            for (kernel_row = 0U; kernel_row < config->kernel_size; ++kernel_row) {
                                size_t kernel_column;
                                for (kernel_column = 0U; kernel_column < config->kernel_size; ++kernel_column) {
                                    size_t input_index = cnn_frame_index(
                                        config, channel_index,
                                        out_row * config->stride + kernel_row,
                                        out_column * config->stride + kernel_column);
                                    size_t weight_index;
                                    if (is_dw) {
                                        weight_index = filter_index * kernel_plane +
                                            kernel_row * config->kernel_size + kernel_column;
                                    } else {
                                        weight_index = cnn_kernel_index(
                                            config, filter_index, channel_index, kernel_row, kernel_column);
                                    }
                                    conv_value += frame[input_index] * context->conv_weights[weight_index];
                                }
                            }
                        }
                        pooled_sum += conv_value;
                        if (!have_max || conv_value > pooled_max) {
                            pooled_max = conv_value;
                            pooled_max_pos = position_counter;
                            have_max = 1;
                        }
                        position_counter += 1U;
                    }
                }

                if (max_index_cache != NULL) {
                    max_index_cache[(step_index * config->filter_count) + filter_index] = pooled_max_pos;
                }
                if (context->max_index_cache != NULL) {
                    context->max_index_cache[filter_index] = pooled_max_pos;
                }

                if (config->pooling_mode == CNN_POOL_MAX) {
                    /* Apply BN to max-pooled value before activation */
                    if (config->use_batch_norm && context->bn_gamma != NULL) {
                        float bn_mean = context->bn_running_mean[filter_index];
                        float bn_var  = context->bn_running_var[filter_index];
                        float bn_eps  = config->bn_epsilon;
                        float pooled_max_raw = pooled_max;
                        /* Update BN running statistics from per-sample max-pooled value. */
                        if (bn_pre_cache != NULL) {
                            float mom = config->bn_momentum;
                            float old_mean = context->bn_running_mean[filter_index];
                            context->bn_running_mean[filter_index] =
                                mom * old_mean + (1.0f - mom) * pooled_max_raw;
                            float delta = pooled_max_raw - old_mean;
                            context->bn_running_var[filter_index] =
                                mom * context->bn_running_var[filter_index] +
                                (1.0f - mom) * delta * delta;
                            if (bn_spatial_var != NULL) {
                                bn_spatial_var[step_index * config->filter_count + filter_index] =
                                    context->bn_running_var[filter_index];
                            }
                        }
                        float x_hat   = (pooled_max - bn_mean) / sqrtf((bn_var + bn_eps) > 0.0f ? (bn_var + bn_eps) : bn_eps);
                        pooled_max    = context->bn_gamma[filter_index] * x_hat +
                                        context->bn_beta[filter_index];
                        if (pooled_linear_cache != NULL) {
                            pooled_linear_cache[(step_index * config->filter_count) + filter_index] =
                                (bn_pre_cache != NULL) ? pooled_max_raw : pooled_max;
                        }
                    } else {
                        if (pooled_linear_cache != NULL) {
                            pooled_linear_cache[(step_index * config->filter_count) + filter_index] = pooled_max;
                        }
                    }
                    float pooled_max_act = cnn_apply_activation(pooled_max, config->pooling_activation);
                    pooled_values[filter_index] = pooled_max_act;
                    if (pooled_activation_cache != NULL) {
                        pooled_activation_cache[(step_index * config->filter_count) + filter_index] = pooled_max_act;
                    }
                } else {
                    /* CNN_POOL_DUAL */
                    float pooled_avg_linear = pooled_sum / (float)output_positions;
                    float pooled_avg_linear_raw = pooled_avg_linear;
                    float pooled_max_raw = pooled_max;
                    /* In training mode, update BN running statistics from dual-pooled values.
                     * Use average of avg and max as a representative sample. */
                    if (bn_pre_cache != NULL && config->use_batch_norm && context->bn_gamma != NULL) {
                        float mom = config->bn_momentum;
                        float raw_avg = (pooled_avg_linear_raw + pooled_max_raw) * 0.5f;
                        float old_mean = context->bn_running_mean[filter_index];
                        context->bn_running_mean[filter_index] =
                            mom * old_mean + (1.0f - mom) * raw_avg;
                        float delta = raw_avg - old_mean;
                        context->bn_running_var[filter_index] =
                            mom * context->bn_running_var[filter_index] +
                            (1.0f - mom) * delta * delta;
                        if (bn_spatial_var != NULL) {
                            bn_spatial_var[step_index * config->filter_count + filter_index] =
                                context->bn_running_var[filter_index];
                        }
                    }
                    float bn_mean = 0.0f, bn_var = 1.0f, bn_eps = 0.0f;
                    float bn_gamma_val = 1.0f, bn_beta_val = 0.0f;
                    /* Apply BN to both avg and max pooled values before activation */
                    if (config->use_batch_norm && context->bn_gamma != NULL) {
                        bn_mean = context->bn_running_mean[filter_index];
                        bn_var  = context->bn_running_var[filter_index];
                        bn_eps  = config->bn_epsilon;
                        bn_gamma_val = context->bn_gamma[filter_index];
                        bn_beta_val  = context->bn_beta[filter_index];
                        {
                            float x_hat_avg = (pooled_avg_linear - bn_mean) / sqrtf((bn_var + bn_eps) > 0.0f ? (bn_var + bn_eps) : bn_eps);
                            pooled_avg_linear = bn_gamma_val * x_hat_avg + bn_beta_val;
                        }
                        {
                            float x_hat_max = (pooled_max - bn_mean) / sqrtf((bn_var + bn_eps) > 0.0f ? (bn_var + bn_eps) : bn_eps);
                            pooled_max = bn_gamma_val * x_hat_max + bn_beta_val;
                        }
                    }
                    float pooled_avg_act = cnn_apply_activation(pooled_avg_linear, config->pooling_activation);
                    float pooled_max_act = cnn_apply_activation(pooled_max, config->pooling_activation);
                    pooled_values[filter_index * 2U] = pooled_avg_act;
                    pooled_values[(filter_index * 2U) + 1U] = pooled_max_act;
                    if (pooled_linear_cache != NULL) {
                        /* In training mode store pre-BN raw values for batch statistics */
                        if (bn_pre_cache != NULL && config->use_batch_norm && context->bn_gamma != NULL) {
                            pooled_linear_cache[pooled_avg_index] = pooled_avg_linear_raw;
                            pooled_linear_cache[pooled_max_index] = pooled_max_raw;
                        } else {
                            pooled_linear_cache[pooled_avg_index] = pooled_avg_linear;
                            pooled_linear_cache[pooled_max_index] = pooled_max;
                        }
                    }
                    if (pooled_activation_cache != NULL) {
                        pooled_activation_cache[pooled_avg_index] = pooled_avg_act;
                        pooled_activation_cache[pooled_max_index] = pooled_max_act;
                    }
                }
            }
        }

        /* ── Dropout: apply between pooling and projection ──
         * Inverted dropout: keep prob = 1-rate, scale kept by 1/(1-rate).
         * Mask stored for backward pass. Only active when pooling is used. */
        if (dropout_rate > 0.0f && dropout_mask != NULL) {
            float keep_prob = 1.0f - dropout_rate;
            float scale = 1.0f / (keep_prob > 0.0f ? keep_prob : 0.001f);
            size_t pool_idx;
            for (pool_idx = 0U; pool_idx < pooled_value_count; ++pool_idx) {
                float r = (float)cnn_next_random(&context->rng_state) / 4294967295.0f;
                float mask = (r < keep_prob) ? scale : 0.0f;
                pooled_values[pool_idx] *= mask;
                dropout_mask[(step_index * pooled_value_count) + pool_idx] = mask;
            }
        }

        /* Project pooled filter responses into a compact feature vector for the downstream leaf. */
        for (feature_index = 0U; feature_index < config->feature_size; ++feature_index) {
            float linear_value = context->projection_bias[feature_index];
            size_t pooled_index;

            for (pooled_index = 0U; pooled_index < pooled_value_count; ++pooled_index) {
                linear_value += context->projection_weights[
                    (feature_index * pooled_value_count) + pooled_index
                ] * pooled_values[pooled_index];
            }

            if (output_linear_cache != NULL) {
                output_linear_cache[(step_index * config->feature_size) + feature_index] = linear_value;
            }
            output[(step_index * config->feature_size) + feature_index] =
                cnn_apply_activation(linear_value, config->output_activation);
        }
    }

    /* ── BN training post-processing for pooled paths ──
     * During training, re-normalize per-step pooled linear values using
     * batch statistics computed across all sequence_length steps.  This
     * replaces the per-step inference-style BN that uses running stats only.
     *
     * NOTE: Skip when batch statistics are unreliable (single value per
     * filter → variance is always zero → normalization collapses output
     * to a constant beta).  The running-stats BN applied during the
     * initial forward pass is sufficient in this case. */
    if (config->pooling_mode != CNN_POOL_NONE &&
        config->use_batch_norm && context->bn_gamma != NULL &&
        bn_spatial_var != NULL && bn_pre_cache != NULL &&
        pooled_linear_cache != NULL && pooled_activation_cache != NULL) {
        size_t filt;
        int is_dual = (config->pooling_mode == CNN_POOL_DUAL) ? 1 : 0;

        /* Require at least 2 values per filter for meaningful batch statistics.
         * sequence_length=1 with non-dual pooling → 1 value → skip. */
        if (config->sequence_length < 2U && !is_dual) {
            return 0;
        }

        for (filt = 0U; filt < config->filter_count; ++filt) {
            float bmean = 0.0f;
            float bvar  = 0.0f;
            size_t n = 0U;
            size_t si, sii;

            /* Pass 1: compute batch mean from pre-BN raw values */
            for (si = 0U; si < config->sequence_length; ++si) {
                for (sii = 0U; sii < (size_t)(is_dual ? 2U : 1U); ++sii) {
                    size_t idx;
                    if (is_dual) {
                        idx = si * pooled_value_count + filt * 2U + sii;
                    } else {
                        idx = si * config->filter_count + filt;
                    }
                    bmean += pooled_linear_cache[idx];
                    n++;
                }
            }
            bmean /= (float)n;

            /* Pass 2: compute batch variance */
            for (si = 0U; si < config->sequence_length; ++si) {
                for (sii = 0U; sii < (size_t)(is_dual ? 2U : 1U); ++sii) {
                    size_t idx;
                    if (is_dual) {
                        idx = si * pooled_value_count + filt * 2U + sii;
                    } else {
                        idx = si * config->filter_count + filt;
                    }
                    float delta = pooled_linear_cache[idx] - bmean;
                    bvar += delta * delta;
                }
            }
            bvar /= (float)n;
            if (bvar < 0.0f) bvar = 0.0f;

            /* Store batch variance for backward pass */
            bn_spatial_var[filt] = bvar;

            /* Update running statistics with momentum */
            {
                float mom = config->bn_momentum;
                context->bn_running_mean[filt] =
                    mom * context->bn_running_mean[filt] + (1.0f - mom) * bmean;
                context->bn_running_var[filt] =
                    mom * context->bn_running_var[filt] + (1.0f - mom) * bvar;
            }

            /* Pass 3: apply BN normalization with batch statistics */
            {
                float gamma     = context->bn_gamma[filt];
                float beta      = context->bn_beta[filt];
                float inv_std = 1.0f / sqrtf(bvar + config->bn_epsilon > 0.0f
                    ? bvar + config->bn_epsilon : config->bn_epsilon);

                for (si = 0U; si < config->sequence_length; ++si) {
                    for (sii = 0U; sii < (size_t)(is_dual ? 2U : 1U); ++sii) {
                        size_t idx;
                        float x_hat, bn_out;
                        if (is_dual) {
                            idx = si * pooled_value_count + filt * 2U + sii;
                        } else {
                            idx = si * config->filter_count + filt;
                        }
                        x_hat = (pooled_linear_cache[idx] - bmean) * inv_std;
                        bn_out = gamma * x_hat + beta;
                        bn_pre_cache[idx] = x_hat;
                        pooled_linear_cache[idx] = bn_out;
                        pooled_activation_cache[idx] =
                            cnn_apply_activation(bn_out, config->pooling_activation);
                    }
                }
            }
        }

        /* Recompute pooled_values and projection output with corrected activation */
        {
            for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
                /* Refresh pooled_values buffer for current step */
                for (filt = 0U; filt < config->filter_count; ++filt) {
                    if (is_dual) {
                        size_t base = step_index * pooled_value_count + filt * 2U;
                        context->pooled_values[filt * 2U]     = pooled_activation_cache[base];
                        context->pooled_values[filt * 2U + 1U] = pooled_activation_cache[base + 1U];
                    } else {
                        size_t idx = step_index * config->filter_count + filt;
                        context->pooled_values[filt] = pooled_activation_cache[idx];
                    }
                }

                /* Re-apply stored dropout mask after pooled_values refresh */
                if (dropout_rate > 0.0f && dropout_mask != NULL) {
                    size_t drop_base = step_index * pooled_value_count;
                    size_t dpi;
                    for (dpi = 0U; dpi < pooled_value_count; ++dpi) {
                        context->pooled_values[dpi] *= dropout_mask[drop_base + dpi];
                    }
                }

                /* Re-run projection with corrected activation */
                for (feature_index = 0U; feature_index < config->feature_size; ++feature_index) {
                    float linear_value = context->projection_bias[feature_index];
                    size_t pool_idx;
                    for (pool_idx = 0U; pool_idx < pooled_value_count; ++pool_idx) {
                        linear_value += context->projection_weights[
                            (feature_index * pooled_value_count) + pool_idx
                        ] * context->pooled_values[pool_idx];
                    }
                    if (output_linear_cache != NULL) {
                        output_linear_cache[(step_index * config->feature_size) + feature_index] = linear_value;
                    }
                    output[(step_index * config->feature_size) + feature_index] =
                        cnn_apply_activation(linear_value, config->output_activation);
                }
            }
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
        return ACTION_C_ERR_NULL_POINTER;
    }

    return nn_cnn_forward_pass(
        context,
        context->input_buffer,
        context->output_buffer,
        NULL,
        NULL,
        NULL,
        NULL,
        context->bn_training_pre_cache,
        context->bn_training_spatial_var,
        0.0f,
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
        return ACTION_C_ERR_NULL_POINTER;
    }

    output_size = (context->config.pooling_mode == CNN_POOL_NONE) ?
        cnn_none_output_size(&context->config) :
        (context->config.sequence_length * context->config.feature_size);
    nn_cnn_infer_set_input(context, input, context->config.total_input_size);
    if (nn_cnn_infer_step(context) != 0) {
        return ACTION_C_ERR_INTERNAL;
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
        header.feature_size != context->config.feature_size ||
        header.stride != (uint32_t)context->config.stride) {
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
    if (projection_weight_count > 0U &&
        fread(context->projection_weights, sizeof(float), projection_weight_count, fp) != projection_weight_count) {
        return 0;
    }
    if (context->config.feature_size > 0U &&
        fread(context->projection_bias, sizeof(float), context->config.feature_size, fp) != context->config.feature_size) {
        return 0;
    }
    /* Load BN parameters if present */
    if (context->bn_gamma != NULL) {
        if (fread(context->bn_gamma, sizeof(float), context->config.filter_count, fp) != context->config.filter_count) {
            return 0;
        }
        if (fread(context->bn_beta, sizeof(float), context->config.filter_count, fp) != context->config.filter_count) {
            return 0;
        }
        if (fread(context->bn_running_mean, sizeof(float), context->config.filter_count, fp) != context->config.filter_count) {
            return 0;
        }
        if (fread(context->bn_running_var, sizeof(float), context->config.filter_count, fp) != context->config.filter_count) {
            return 0;
        }
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
    header.stride = (uint32_t)context->config.stride;

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
    if (projection_weight_count > 0U &&
        fwrite(context->projection_weights, sizeof(float), projection_weight_count, fp) != projection_weight_count) {
        return 0;
    }
    if (context->config.feature_size > 0U &&
        fwrite(context->projection_bias, sizeof(float), context->config.feature_size, fp) != context->config.feature_size) {
        return 0;
    }
    /* Save BN parameters if present */
    if (context->bn_gamma != NULL) {
        if (fwrite(context->bn_gamma, sizeof(float), context->config.filter_count, fp) != context->config.filter_count) {
            return 0;
        }
        if (fwrite(context->bn_beta, sizeof(float), context->config.filter_count, fp) != context->config.filter_count) {
            return 0;
        }
        if (fwrite(context->bn_running_mean, sizeof(float), context->config.filter_count, fp) != context->config.filter_count) {
            return 0;
        }
        if (fwrite(context->bn_running_var, sizeof(float), context->config.filter_count, fp) != context->config.filter_count) {
            return 0;
        }
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

/* ─── VTable backend ─── */

#include "../../nn_backend.h"

static uint32_t cnn_infer_abi_version(void) {
    return CNN_ABI_VERSION;
}

static uint64_t cnn_infer_layout_hash(const void* context) {
    const CnnInferContext* ctx = (const CnnInferContext*)context;
    if (ctx == NULL) return 0;
    return cnn_compute_layout_hash(&ctx->config);
}

static int cnn_infer_get_output_int(const void* context, float* out, size_t out_size) {
    nn_cnn_infer_get_output((void*)context, out, out_size);
    return 0;
}

static void* cnn_infer_create_vtable(const void* config_blob, size_t config_size,
                                      struct Arena* arena) {
    const CnnConfig* config;
    (void)arena;
    if (config_blob == NULL || config_size < sizeof(CnnConfig)) return NULL;
    config = (const CnnConfig*)config_blob;
    return nn_cnn_infer_create_with_config(config, 0);
}

static int cnn_vtable_save_weights(const void* context, FILE* fp) {
    return nn_cnn_save_weights((void*)context, fp);
}

const NNInferBackend g_cnn_infer_backend = {
    .type_name        = "cnn",
    .create           = cnn_infer_create_vtable,
    .destroy          = nn_cnn_infer_destroy,
    .step             = nn_cnn_infer_step,
    .get_output       = cnn_infer_get_output_int,
    .save_weights     = cnn_vtable_save_weights,
    .load_weights     = nn_cnn_load_weights,
    .get_network_hash = nn_cnn_get_network_hash,
    .get_layout_hash  = cnn_infer_layout_hash,
    .get_abi_version  = cnn_infer_abi_version,
};

/* ─── cnn_dual_pool VTable (forces CNN_POOL_DUAL mode) ─── */

static void* cnn_dual_pool_infer_create_vtable(const void* config_blob, size_t config_size,
                                                struct Arena* arena) {
    CnnConfig config;
    (void)arena;
    if (config_blob == NULL || config_size < sizeof(CnnConfig)) return NULL;
    config = *(const CnnConfig*)config_blob;
    config.pooling_mode = CNN_POOL_DUAL;
    return nn_cnn_infer_create_with_config(&config, 0);
}

const NNInferBackend g_cnn_dual_pool_infer_backend = {
    .type_name        = "cnn_dual_pool",
    .create           = cnn_dual_pool_infer_create_vtable,
    .destroy          = nn_cnn_infer_destroy,
    .step             = nn_cnn_infer_step,
    .get_output       = cnn_infer_get_output_int,
    .save_weights     = cnn_vtable_save_weights,
    .load_weights     = nn_cnn_load_weights,
    .get_network_hash = nn_cnn_get_network_hash,
    .get_layout_hash  = cnn_infer_layout_hash,
    .get_abi_version  = cnn_infer_abi_version,
};
