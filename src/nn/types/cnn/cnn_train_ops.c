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
#include "../../../utils/error.h"

/**
 * @brief Recover d(activation)/d(linear) from the post-activation output value.
 */
static float cnn_activation_derivative_from_output(float output, CnnActivationType activation) {
    switch (activation) {
        case CNN_ACT_RELU:
            return output > 0.0f ? 1.0f : 0.0f;
        case CNN_ACT_TANH:
            return (1.0f - output) * (1.0f + output);
        case CNN_ACT_RELU6:
            return (output > 0.0f && output < 6.0f) ? 1.0f : 0.0f;
        case CNN_ACT_NONE:
        default:
            return 1.0f;
    }
}

#include "cnn_common.h"

static void cnn_zero_gradients(CnnTrainContext* context) {
    const CnnConfig* config;
    size_t conv_weight_count;
    size_t projection_weight_count;
    size_t pooled_cache_count;

    if (context == NULL || context->infer_ctx == NULL) {
        return;
    }

    config = &context->infer_ctx->config;
    conv_weight_count = cnn_conv_weight_count(config);
    projection_weight_count = cnn_projection_weight_count(config);
    pooled_cache_count = config->sequence_length * cnn_pooled_value_count(config);

    if (context->pooled_gradient_cache != NULL) {
        (void)memset(context->pooled_gradient_cache, 0, pooled_cache_count * sizeof(float));
    }
    (void)memset(context->conv_weight_grad, 0, conv_weight_count * sizeof(float));
    (void)memset(context->conv_bias_grad, 0, config->filter_count * sizeof(float));
    if (context->projection_weight_grad != NULL) {
        (void)memset(context->projection_weight_grad, 0, projection_weight_count * sizeof(float));
    }
    if (context->projection_bias_grad != NULL) {
        (void)memset(context->projection_bias_grad, 0, config->feature_size * sizeof(float));
    }
    if (context->bn_gamma_grad != NULL) {
        (void)memset(context->bn_gamma_grad, 0, config->filter_count * sizeof(float));
    }
    if (context->bn_beta_grad != NULL) {
        (void)memset(context->bn_beta_grad, 0, config->filter_count * sizeof(float));
    }
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
    float lr = context->config.learning_rate;
    float momentum = context->config.momentum;
    float wd = context->config.weight_decay;

    infer_ctx = context->infer_ctx;
    config = &infer_ctx->config;
    conv_weight_count = cnn_conv_weight_count(config);
    projection_weight_count = cnn_projection_weight_count(config);

    /* Momentum SGD: vel = momentum * vel - lr * grad; weight += vel.
     * Weight decay is applied only to true weights, not to bias vectors.
     * When momentum == 0, this degrades to pure SGD for backward compatibility. */
    for (weight_index = 0U; weight_index < conv_weight_count; ++weight_index) {
        float grad_wd = context->conv_weight_grad[weight_index] +
                        wd * infer_ctx->conv_weights[weight_index];
        context->conv_weight_vel[weight_index] =
            momentum * context->conv_weight_vel[weight_index] - lr * grad_wd;
        infer_ctx->conv_weights[weight_index] += context->conv_weight_vel[weight_index];
    }
    for (weight_index = 0U; weight_index < config->filter_count; ++weight_index) {
        context->conv_bias_vel[weight_index] =
            momentum * context->conv_bias_vel[weight_index] -
            lr * context->conv_bias_grad[weight_index];
        infer_ctx->conv_bias[weight_index] += context->conv_bias_vel[weight_index];
    }
    if (infer_ctx->projection_weights != NULL && context->projection_weight_vel != NULL) {
        for (weight_index = 0U; weight_index < projection_weight_count; ++weight_index) {
            float grad_wd = context->projection_weight_grad[weight_index] +
                            wd * infer_ctx->projection_weights[weight_index];
            context->projection_weight_vel[weight_index] =
                momentum * context->projection_weight_vel[weight_index] - lr * grad_wd;
            infer_ctx->projection_weights[weight_index] += context->projection_weight_vel[weight_index];
        }
    }
    if (infer_ctx->projection_bias != NULL && context->projection_bias_vel != NULL) {
        for (weight_index = 0U; weight_index < config->feature_size; ++weight_index) {
            context->projection_bias_vel[weight_index] =
                momentum * context->projection_bias_vel[weight_index] -
                lr * context->projection_bias_grad[weight_index];
            infer_ctx->projection_bias[weight_index] += context->projection_bias_vel[weight_index];
        }
    }

    /* Update BN learnable parameters (with momentum, same as other params) */
    if (config->use_batch_norm && infer_ctx->bn_gamma != NULL &&
        context->bn_gamma_grad != NULL) {
        for (weight_index = 0U; weight_index < config->filter_count; ++weight_index) {
            context->bn_gamma_vel[weight_index] =
                momentum * context->bn_gamma_vel[weight_index] -
                lr * context->bn_gamma_grad[weight_index];
            infer_ctx->bn_gamma[weight_index] += context->bn_gamma_vel[weight_index];
            context->bn_beta_vel[weight_index] =
                momentum * context->bn_beta_vel[weight_index] -
                lr * context->bn_beta_grad[weight_index];
            infer_ctx->bn_beta[weight_index] += context->bn_beta_vel[weight_index];
        }
    }
}

/**
 * @brief Backpropagate one externally supplied dL/dY through the CNN leaf.
 */
static int cnn_backpropagate(
    CnnTrainContext* restrict context,
    const float* restrict input,
    const float* restrict output_gradient,
    float* restrict input_gradient
) {
    CnnInferContext* infer_ctx;
    const CnnConfig* config;
    size_t frame_stride;
    size_t output_grid_width;
    size_t output_grid_height;
    size_t output_positions;
    size_t pooled_value_count;
    size_t step_index;
    size_t filter_index;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || output_gradient == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    infer_ctx = context->infer_ctx;
    config = &infer_ctx->config;
    frame_stride = config->frame_width * config->frame_height * config->channel_count;
    output_grid_width = (config->frame_width - config->kernel_size) / config->stride + 1U;
    output_grid_height = (config->frame_height - config->kernel_size) / config->stride + 1U;
    output_positions = cnn_conv_position_count(config);
    pooled_value_count = cnn_pooled_value_count(config);
    if (output_positions == 0U) {
        return ACTION_C_ERR_DIM_MISMATCH;
    }

    cnn_zero_gradients(context);
    if (input_gradient != NULL) {
        (void)memset(input_gradient, 0, config->total_input_size * sizeof(float));
    }

    if (config->pooling_mode == CNN_POOL_NONE) {
        /* POOL_NONE backprop: output_gradient flows directly through conv2d.
         * No pooling or projection layers exist, so each output gradient
         * element corresponds to one conv2d spatial position.
         * Supports depthwise conv and BN. */
        size_t grid_plane = output_grid_height * output_grid_width;
        size_t output_stride = config->filter_count * grid_plane;
        int is_dw = (config->conv_mode == CNN_CONV_DEPTHWISE) ? 1 : 0;
        size_t kernel_plane = config->kernel_size * config->kernel_size;
        int has_bn = (config->use_batch_norm && infer_ctx->bn_gamma != NULL &&
                       context->bn_gamma_grad != NULL) ? 1 : 0;
        float bn_eps = config->bn_epsilon;

        for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
            const float* frame = input + (step_index * frame_stride);
            const float* step_out = infer_ctx->output_buffer + (step_index * output_stride);
            const float* step_grad = output_gradient + (step_index * output_stride);
            float* step_in_grad = (input_gradient != NULL) ? (input_gradient + (step_index * frame_stride)) : NULL;

            for (filter_index = 0U; filter_index < config->filter_count; ++filter_index) {
                size_t out_row;
                size_t out_column;

                /* ── Pass 1: accumulate BN correction sums over all spatial positions ── */
                float sum_dz = 0.0f;
                float sum_dz_xh = 0.0f;
                float bn_scale = 1.0f;
                float bn_const_term = 0.0f;
                float bn_xh_coeff = 0.0f;
                float N = (float)(output_grid_height * output_grid_width);

                if (has_bn) {
                    for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                        for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                            size_t flat_index = ((filter_index * output_grid_height) + out_row) * output_grid_width + out_column;
                            float out_val = step_out[flat_index];
                            float act_deriv = cnn_activation_derivative_from_output(out_val, config->output_activation);
                            float dz = step_grad[flat_index] * act_deriv;
                            if (dz != 0.0f) {
                                float x_hat = context->bn_pre_cache[step_index * config->filter_count * grid_plane + flat_index];
                                sum_dz += dz;
                                sum_dz_xh += dz * x_hat;
                            }
                        }
                    }

                    {
                        float gamma = infer_ctx->bn_gamma[filter_index];
                        float sp_var = context->bn_spatial_var[step_index * config->filter_count + filter_index];
                        float var_eps = sp_var + bn_eps;
                        float inv_std = 1.0f / sqrtf(var_eps > 0.0f ? var_eps : (bn_eps > 0.0f ? bn_eps : 1e-8f));
                        float factor = gamma * inv_std / N;
                        bn_scale = gamma * inv_std;
                        bn_const_term = -factor * sum_dz;
                        bn_xh_coeff = -factor * sum_dz_xh;
                    }
                }

                /* ── Pass 2: full BN gradient through convolution weights ── */
                for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                    for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                        size_t flat_index = ((filter_index * output_grid_height) + out_row) * output_grid_width + out_column;
                        float out_val = step_out[flat_index];
                        float act_deriv = cnn_activation_derivative_from_output(out_val, config->output_activation);
                        float dz = step_grad[flat_index] * act_deriv;
                        float dconv;
                        size_t channel_index;

                        /* Batch Normalization backward (full training formula):
                         * dL/dx_i = (gamma/(N*sqrt(var+eps))) * [N*dz_i - sum_j(dz_j) - x_hat_i*sum_j(dz_j*x_hat_j)]
                         * = dz_i*scale + const_term + xh_coeff*x_hat_i
                         * where scale=gamma/sqrt(var+eps), const_term=-factor*sum_dz, xh_coeff=-factor*sum_dz_xh */
                        if (has_bn) {
                            float x_hat = context->bn_pre_cache[step_index * config->filter_count * grid_plane + flat_index];
                            context->bn_gamma_grad[filter_index] += dz * x_hat;
                            context->bn_beta_grad[filter_index] += dz;
                            dconv = dz * bn_scale + bn_const_term + bn_xh_coeff * x_hat;
                        } else {
                            dconv = dz;
                        }

                        context->conv_bias_grad[filter_index] += dconv;

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

                                    context->conv_weight_grad[weight_index] +=
                                        dconv * frame[input_index];
                                    if (input_gradient != NULL) {
                                        step_in_grad[input_index] +=
                                            infer_ctx->conv_weights[weight_index] * dconv;
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

    /* Stage 1: output projection gradients and pooled-feature backprop signal. */
    for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
        size_t feature_index;
        for (feature_index = 0U; feature_index < config->feature_size; ++feature_index) {
            size_t output_index = (step_index * config->feature_size) + feature_index;
            float output_value = infer_ctx->output_buffer[output_index];
            float dz = output_gradient[output_index] *
                cnn_activation_derivative_from_output(output_value, config->output_activation);
            size_t pooled_index;

            context->projection_bias_grad[feature_index] += dz;
            for (pooled_index = 0U; pooled_index < pooled_value_count; ++pooled_index) {
                size_t cache_index = (step_index * pooled_value_count) + pooled_index;
                size_t weight_index = (feature_index * pooled_value_count) + pooled_index;
                context->projection_weight_grad[weight_index] +=
                    dz * context->pooled_activation_cache[cache_index];
                context->pooled_gradient_cache[cache_index] +=
                    infer_ctx->projection_weights[weight_index] * dz;
            }
        }
    }

    /* Stage 2: pooled-feature gradients back through shared convolution kernels.
     * Supports depthwise conv and BN (which is applied per-filter after pooling). */
    {
        int is_dw = (config->conv_mode == CNN_CONV_DEPTHWISE) ? 1 : 0;
        size_t kernel_plane = config->kernel_size * config->kernel_size;
        int has_bn_pool = (config->use_batch_norm && infer_ctx->bn_gamma != NULL &&
                            context->bn_gamma_grad != NULL) ? 1 : 0;
        float bn_eps = config->bn_epsilon;

        for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
            const float* frame = input + (step_index * frame_stride);

            for (filter_index = 0U; filter_index < config->filter_count; ++filter_index) {
                size_t out_row;
                size_t out_column;

                if (config->pooling_mode == CNN_POOL_AVG) {
                    /* Average pooling: gradient distributed equally to all positions. */
                    size_t pooled_index = (step_index * config->filter_count) + filter_index;
                    float pooled_output = context->pooled_activation_cache[pooled_index];
                    float dpool_act = context->pooled_gradient_cache[pooled_index] *
                        cnn_activation_derivative_from_output(pooled_output, config->pooling_activation);
                    float dpool_linear;

                    /* BN backward on pooled value */
                    if (has_bn_pool && dpool_act != 0.0f) {
                        float gamma = infer_ctx->bn_gamma[filter_index];
                        float beta  = infer_ctx->bn_beta[filter_index];
                        float bn_var = context->bn_spatial_var ? context->bn_spatial_var[filter_index] : infer_ctx->bn_running_var[filter_index];
                        float linear_val = context->pooled_linear_cache[pooled_index];
                        float x_hat = (linear_val - beta) / gamma;
                        float scale = gamma / sqrtf((bn_var + bn_eps) > 0.0f ? (bn_var + bn_eps) : (bn_eps > 0.0f ? bn_eps : 1e-8f));
                        context->bn_gamma_grad[filter_index] += dpool_act * x_hat;
                        context->bn_beta_grad[filter_index] += dpool_act;
                        dpool_linear = dpool_act * scale;
                    } else {
                        dpool_linear = dpool_act;
                    }

                    float position_scale = dpool_linear / (float)output_positions;

                    context->conv_bias_grad[filter_index] += position_scale * (float)output_positions;
                    for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                        for (out_column = 0U; out_column < output_grid_width; ++out_column) {
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
                } else if (config->pooling_mode == CNN_POOL_DUAL) {
                    /* DUAL pooling: avg gradient distributed equally, max gradient to argmax only. */
                    size_t avg_cache_index = (step_index * pooled_value_count) + (filter_index * 2U);
                    size_t max_cache_index = avg_cache_index + 1U;
                    float avg_output = context->pooled_activation_cache[avg_cache_index];
                    float max_output = context->pooled_activation_cache[max_cache_index];
                    float davg_act = context->pooled_gradient_cache[avg_cache_index] *
                        cnn_activation_derivative_from_output(avg_output, config->pooling_activation);
                    float dmax_act = context->pooled_gradient_cache[max_cache_index] *
                        cnn_activation_derivative_from_output(max_output, config->pooling_activation);
                    float davg_linear, dmax_linear;
                    float avg_position_scale;
                    size_t position_counter = 0U;
                    size_t argmax_pos = 0U;

                    if (context->max_index_cache != NULL) {
                        argmax_pos = context->max_index_cache[(step_index * config->filter_count) + filter_index];
                    }

                    /* BN backward on dual-pooled values (avg and max separately) */
                    if (has_bn_pool) {
                        float gamma = infer_ctx->bn_gamma[filter_index];
                        float beta  = infer_ctx->bn_beta[filter_index];
                        float bn_var = context->bn_spatial_var ? context->bn_spatial_var[filter_index] : infer_ctx->bn_running_var[filter_index];
                        float scale = gamma / sqrtf((bn_var + bn_eps) > 0.0f ? (bn_var + bn_eps) : (bn_eps > 0.0f ? bn_eps : 1e-8f));
                        float x_hat_avg, x_hat_max;
                        float dbn_gamma = 0.0f;
                        float dbn_beta  = 0.0f;

                        if (davg_act != 0.0f) {
                            float linear_avg = context->pooled_linear_cache[avg_cache_index];
                            x_hat_avg = (linear_avg - beta) / gamma;
                            dbn_gamma += davg_act * x_hat_avg;
                            dbn_beta  += davg_act;
                            davg_linear = davg_act * scale;
                        } else {
                            davg_linear = 0.0f;
                        }
                        if (dmax_act != 0.0f) {
                            float linear_max = context->pooled_linear_cache[max_cache_index];
                            x_hat_max = (linear_max - beta) / gamma;
                            dbn_gamma += dmax_act * x_hat_max;
                            dbn_beta  += dmax_act;
                            dmax_linear = dmax_act * scale;
                        } else {
                            dmax_linear = 0.0f;
                        }
                        context->bn_gamma_grad[filter_index] += dbn_gamma;
                        context->bn_beta_grad[filter_index]  += dbn_beta;
                    } else {
                        davg_linear = davg_act;
                        dmax_linear = dmax_act;
                    }

                    avg_position_scale = davg_linear / (float)output_positions;

                    context->conv_bias_grad[filter_index] += davg_linear + dmax_linear;
                    for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                        for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                            float position_gradient = avg_position_scale;
                            if (position_counter == argmax_pos) {
                                position_gradient += dmax_linear;
                            }
                            {
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
                                            context->conv_weight_grad[weight_index] +=
                                                position_gradient * frame[input_index];
                                            if (input_gradient != NULL) {
                                                input_gradient[(step_index * frame_stride) + input_index] +=
                                                    infer_ctx->conv_weights[weight_index] * position_gradient;
                                            }
                                        }
                                    }
                                }
                            }
                            position_counter += 1U;
                        }
                    }
                } else {
                    /* MAX pooling: gradient routed to argmax position only. */
                    size_t pooled_index = (step_index * config->filter_count) + filter_index;
                    float pooled_output = context->pooled_activation_cache[pooled_index];
                    float dpool_act = context->pooled_gradient_cache[pooled_index] *
                        cnn_activation_derivative_from_output(pooled_output, config->pooling_activation);
                    float dmax_linear;
                    size_t argmax_pos = 0U;
                    size_t position_counter = 0U;

                    if (context->max_index_cache != NULL) {
                        argmax_pos = context->max_index_cache[(step_index * config->filter_count) + filter_index];
                    }

                    /* BN backward */
                    if (has_bn_pool && dpool_act != 0.0f) {
                        float gamma = infer_ctx->bn_gamma[filter_index];
                        float beta  = infer_ctx->bn_beta[filter_index];
                        float bn_var = context->bn_spatial_var ? context->bn_spatial_var[filter_index] : infer_ctx->bn_running_var[filter_index];
                        float linear_val = context->pooled_linear_cache[pooled_index];
                        float x_hat = (linear_val - beta) / gamma;
                        float scale = gamma / sqrtf((bn_var + bn_eps) > 0.0f ? (bn_var + bn_eps) : (bn_eps > 0.0f ? bn_eps : 1e-8f));
                        context->bn_gamma_grad[filter_index] += dpool_act * x_hat;
                        context->bn_beta_grad[filter_index] += dpool_act;
                        dmax_linear = dpool_act * scale;
                    } else {
                        dmax_linear = dpool_act;
                    }

                    context->conv_bias_grad[filter_index] += dmax_linear;

                    /* Route gradient only to the argmax spatial position */
                    for (out_row = 0U; out_row < output_grid_height; ++out_row) {
                        for (out_column = 0U; out_column < output_grid_width; ++out_column) {
                            if (position_counter == argmax_pos) {
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
                                            context->conv_weight_grad[weight_index] +=
                                                dmax_linear * frame[input_index];
                                            if (input_gradient != NULL) {
                                                input_gradient[(step_index * frame_stride) + input_index] +=
                                                    infer_ctx->conv_weights[weight_index] * dmax_linear;
                                            }
                                        }
                                    }
                                }
                            }
                            position_counter += 1U;
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
    size_t pooled_value_count;
    size_t pooled_cache_count;
    size_t conv_weight_count;
    size_t projection_weight_count;

    if (infer_ctx == NULL || config == NULL) {
        return NULL;
    }

    infer_config = &infer_ctx->config;
    pooled_value_count = cnn_pooled_value_count(infer_config);
    pooled_cache_count = infer_config->sequence_length * pooled_value_count;
    conv_weight_count = cnn_conv_weight_count(infer_config);
    projection_weight_count = cnn_projection_weight_count(infer_config);

    context = (CnnTrainContext*)calloc(1U, sizeof(CnnTrainContext));
    if (context == NULL) {
        return NULL;
    }

    context->infer_ctx = infer_ctx;
    context->config = *config;

    if (infer_config->pooling_mode == CNN_POOL_NONE) {
        /* No pooling or projection — only conv weights/biases are trained. */
        context->pooled_linear_cache = NULL;
        context->pooled_activation_cache = NULL;
        context->max_index_cache = NULL;
        context->output_linear_cache = NULL;
        context->pooled_gradient_cache = NULL;
        context->projection_weight_grad = NULL;
        context->projection_bias_grad = NULL;
        context->projection_weight_vel = NULL;
        context->projection_bias_vel = NULL;
    } else {
        context->pooled_linear_cache = (float*)calloc(pooled_cache_count, sizeof(float));
        context->pooled_activation_cache = (float*)calloc(pooled_cache_count, sizeof(float));
        context->output_linear_cache = (float*)calloc(
            infer_config->sequence_length * infer_config->feature_size, sizeof(float));
        if (infer_config->pooling_mode != CNN_POOL_AVG) {
            context->max_index_cache = (size_t*)calloc(
                infer_config->sequence_length * infer_config->filter_count, sizeof(size_t));
        }
        context->pooled_gradient_cache = (float*)calloc(pooled_cache_count, sizeof(float));
        context->projection_weight_grad = (float*)calloc(projection_weight_count, sizeof(float));
        context->projection_bias_grad = (float*)calloc(infer_config->feature_size, sizeof(float));
        context->projection_weight_vel = (float*)calloc(projection_weight_count, sizeof(float));
        context->projection_bias_vel = (float*)calloc(infer_config->feature_size, sizeof(float));
    }

    /* Momentum velocity buffers */
    context->conv_weight_vel = (float*)calloc(conv_weight_count, sizeof(float));
    context->conv_bias_vel = (float*)calloc(infer_config->filter_count, sizeof(float));

    context->conv_weight_grad = (float*)calloc(conv_weight_count, sizeof(float));
    context->conv_bias_grad = (float*)calloc(infer_config->filter_count, sizeof(float));

    /* BN training buffers */
    if (infer_config->use_batch_norm) {
        context->bn_gamma_grad = (float*)calloc(infer_config->filter_count, sizeof(float));
        context->bn_beta_grad  = (float*)calloc(infer_config->filter_count, sizeof(float));
        context->bn_gamma_vel  = (float*)calloc(infer_config->filter_count, sizeof(float));
        context->bn_beta_vel   = (float*)calloc(infer_config->filter_count, sizeof(float));
        context->bn_spatial_var = (float*)calloc(infer_config->sequence_length * infer_config->filter_count, sizeof(float));
        if (infer_config->pooling_mode == CNN_POOL_NONE) {
            size_t grid_h = (infer_config->frame_height - infer_config->kernel_size) / infer_config->stride + 1U;
            size_t grid_w = (infer_config->frame_width - infer_config->kernel_size) / infer_config->stride + 1U;
            context->bn_pre_cache = (float*)calloc(
                infer_config->sequence_length * infer_config->filter_count * grid_h * grid_w,
                sizeof(float));
        } else {
            context->bn_pre_cache = (float*)calloc(pooled_cache_count, sizeof(float));
        }
    } else {
        context->bn_gamma_grad = NULL;
        context->bn_beta_grad  = NULL;
        context->bn_gamma_vel  = NULL;
        context->bn_beta_vel   = NULL;
        context->bn_pre_cache  = NULL;
        context->bn_spatial_var = NULL;
    }

    /* Propagate training BN buffer pointers to inference context
     * so that graph_run (via nn_cnn_infer_step) can use spatial BN. */
    infer_ctx->bn_training_pre_cache   = context->bn_pre_cache;
    infer_ctx->bn_training_spatial_var = context->bn_spatial_var;

    if (context->conv_weight_grad == NULL || context->conv_bias_grad == NULL ||
        context->conv_weight_vel == NULL || context->conv_bias_vel == NULL) {
        nn_cnn_train_destroy(context);
        return NULL;
    }
    if (infer_config->use_batch_norm) {
        if (context->bn_gamma_vel == NULL || context->bn_beta_vel == NULL) {
            nn_cnn_train_destroy(context);
            return NULL;
        }
    }
    if (infer_config->pooling_mode != CNN_POOL_NONE) {
        if (context->pooled_linear_cache == NULL || context->pooled_activation_cache == NULL ||
            context->output_linear_cache == NULL || context->pooled_gradient_cache == NULL ||
            context->projection_weight_grad == NULL || context->projection_bias_grad == NULL) {
            nn_cnn_train_destroy(context);
            return NULL;
        }
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
    free(context->max_index_cache);
    free(context->output_linear_cache);
    free(context->pooled_gradient_cache);
    free(context->conv_weight_grad);
    free(context->conv_bias_grad);
    free(context->conv_weight_vel);
    free(context->conv_bias_vel);
    free(context->projection_weight_grad);
    free(context->projection_bias_grad);
    free(context->projection_weight_vel);
    free(context->projection_bias_vel);
    free(context->bn_gamma_grad);
    free(context->bn_beta_grad);
    free(context->bn_gamma_vel);
    free(context->bn_beta_vel);
    free(context->bn_spatial_var);
    free(context->bn_pre_cache);
    if (context->infer_ctx != NULL) {
        context->infer_ctx->bn_training_pre_cache   = NULL;
        context->infer_ctx->bn_training_spatial_var = NULL;
    }
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
        return ACTION_C_ERR_NULL_POINTER;
    }

    infer_ctx = context->infer_ctx;
    rc = nn_cnn_forward_pass(
        infer_ctx,
        input,
        infer_ctx->output_buffer,
        context->pooled_linear_cache,
        context->pooled_activation_cache,
        context->max_index_cache,
        context->output_linear_cache,
        context->bn_pre_cache,
        context->bn_spatial_var
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
        return ACTION_C_ERR_NULL_POINTER;
    }

    infer_ctx = context->infer_ctx;
    config = &infer_ctx->config;
    output_size = (config->pooling_mode == CNN_POOL_NONE) ?
        (config->sequence_length * config->filter_count *
         ((config->frame_height - config->kernel_size) / config->stride + 1U) *
         ((config->frame_width - config->kernel_size) / config->stride + 1U)) :
        (config->sequence_length * config->feature_size);

    rc = nn_cnn_forward_pass(
        infer_ctx,
        input,
        infer_ctx->output_buffer,
        context->pooled_linear_cache,
        context->pooled_activation_cache,
        context->max_index_cache,
        context->output_linear_cache,
        context->bn_pre_cache,
        context->bn_spatial_var
    );
    if (rc != 0) {
        return rc;
    }

    output_gradient = (float*)calloc(output_size, sizeof(float));
    if (output_gradient == NULL) {
        return ACTION_C_ERR_NO_MEMORY;
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
    size_t output_size;
    int rc;

    if (context == NULL || context->infer_ctx == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    output_size = (context->infer_ctx->config.pooling_mode == CNN_POOL_NONE) ?
        (context->infer_ctx->config.sequence_length * context->infer_ctx->config.filter_count *
         ((context->infer_ctx->config.frame_height - context->infer_ctx->config.kernel_size) / context->infer_ctx->config.stride + 1U) *
         ((context->infer_ctx->config.frame_width - context->infer_ctx->config.kernel_size) / context->infer_ctx->config.stride + 1U)) :
        (context->infer_ctx->config.sequence_length * context->infer_ctx->config.feature_size);

    dummy_input = (float*)calloc(context->infer_ctx->config.total_input_size, sizeof(float));
    dummy_target = (float*)calloc(output_size, sizeof(float));
    if (dummy_input == NULL || dummy_target == NULL) {
        free(dummy_input);
        free(dummy_target);
        return ACTION_C_ERR_NULL_POINTER;
    }

    rc = nn_cnn_train_step_with_data(context, dummy_input, dummy_target);
    free(dummy_input);
    free(dummy_target);
    return rc;
}

/* ─── VTable backend ─── */

#include "../../nn_backend.h"

static void* cnn_train_create_vtable(const void* config_blob, size_t config_size,
                                      const void* infer_config_blob, size_t infer_config_size,
                                      struct Arena* arena) {
    CnnInferContext* infer_ctx;
    CnnTrainConfig train_cfg;
    const CnnConfig* infer_cfg;
    (void)arena;

    if (config_blob == NULL || config_size < sizeof(CnnTrainConfig)) return NULL;
    train_cfg = *(const CnnTrainConfig*)config_blob;

    if (infer_config_blob == NULL || infer_config_size < sizeof(CnnConfig)) return NULL;
    infer_cfg = (const CnnConfig*)infer_config_blob;
    infer_ctx = nn_cnn_infer_create_with_config(infer_cfg, train_cfg.seed);
    if (infer_ctx == NULL) return NULL;

    return nn_cnn_train_create(infer_ctx, &train_cfg);
}

static int cnn_train_step_vtable(void* context, const float* input, const float* target) {
    CnnTrainContext* ctx = (CnnTrainContext*)context;
    if (ctx == NULL || input == NULL || target == NULL) return -1;
    return nn_cnn_train_step_with_data(ctx, input, target);
}

static int cnn_train_step_with_data_vtable(void* context, const float* input,
                                            const float* target, float* out_grad) {
    CnnTrainContext* ctx = (CnnTrainContext*)context;
    (void)out_grad;
    if (ctx == NULL || input == NULL || target == NULL) return -1;
    return nn_cnn_train_step_with_data(ctx, input, target);
}

static int cnn_train_save_checkpoint_vtable(void* context, FILE* fp) {
    (void)context;
    (void)fp;
    return -1;
}

static int cnn_train_load_checkpoint_vtable(void* context, FILE* fp) {
    (void)context;
    (void)fp;
    return -1;
}

const NNTrainBackend g_cnn_train_backend = {
    .type_name        = "cnn",
    .create           = cnn_train_create_vtable,
    .destroy          = (void (*)(void*))nn_cnn_train_destroy,
    .step             = cnn_train_step_vtable,
    .step_with_data   = cnn_train_step_with_data_vtable,
    .save_checkpoint  = cnn_train_save_checkpoint_vtable,
    .load_checkpoint  = cnn_train_load_checkpoint_vtable,
};

/* ─── cnn_dual_pool VTable (forces CNN_POOL_DUAL mode) ─── */

static void* cnn_dual_pool_train_create_vtable(const void* config_blob, size_t config_size,
                                                const void* infer_config_blob, size_t infer_config_size,
                                                struct Arena* arena) {
    CnnInferContext* infer_ctx;
    CnnTrainConfig train_cfg;
    CnnConfig infer_cfg;
    (void)arena;

    if (config_blob == NULL || config_size < sizeof(CnnTrainConfig)) return NULL;
    train_cfg = *(const CnnTrainConfig*)config_blob;

    if (infer_config_blob == NULL || infer_config_size < sizeof(CnnConfig)) return NULL;
    infer_cfg = *(const CnnConfig*)infer_config_blob;
    infer_cfg.pooling_mode = CNN_POOL_DUAL;
    infer_ctx = nn_cnn_infer_create_with_config(&infer_cfg, train_cfg.seed);
    if (infer_ctx == NULL) return NULL;

    return nn_cnn_train_create(infer_ctx, &train_cfg);
}

const NNTrainBackend g_cnn_dual_pool_train_backend = {
    .type_name        = "cnn_dual_pool",
    .create           = cnn_dual_pool_train_create_vtable,
    .destroy          = (void (*)(void*))nn_cnn_train_destroy,
    .step             = cnn_train_step_vtable,
    .step_with_data   = cnn_train_step_with_data_vtable,
    .save_checkpoint  = cnn_train_save_checkpoint_vtable,
    .load_checkpoint  = cnn_train_load_checkpoint_vtable,
};
