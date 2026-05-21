/**
 * @file cnn_common.h
 * @brief Shared inline helpers used by both inference and training paths.
 *
 * These helpers compute indices and dimension counts and are identical
 * regardless of whether the caller is running a forward pass or backprop.
 * Defining them once as static inline avoids copy-paste drift between the
 * two translation units.
 */

#ifndef CNN_COMMON_H
#define CNN_COMMON_H

#include "cnn_config.h"
#include <stddef.h>

/**
 * @brief Flatten a frame coordinate into the input tensor layout.
 */
static inline size_t cnn_frame_index(const CnnConfig* config, size_t channel, size_t row, size_t column) {
    return ((channel * config->frame_height) + row) * config->frame_width + column;
}

/**
 * @brief Flatten a convolution-weight coordinate into the parameter tensor.
 */
static inline size_t cnn_kernel_index(
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
 * @brief Count valid convolution windows in one frame (stride-aware).
 */
static inline size_t cnn_conv_position_count(const CnnConfig* config) {
    return ((config->frame_height - config->kernel_size) / config->stride + 1U) *
        ((config->frame_width - config->kernel_size) / config->stride + 1U);
}

/**
 * @brief Count convolution weights — depthwise uses fewer parameters.
 */
static inline size_t cnn_conv_weight_count(const CnnConfig* config) {
    if (config->conv_mode == CNN_CONV_DEPTHWISE) {
        return config->filter_count * config->kernel_size * config->kernel_size;
    }
    return config->filter_count * config->channel_count * config->kernel_size * config->kernel_size;
}

/**
 * @brief Count per-filter pooled values (2x for DUAL, 1x for AVG/MAX, 0 for NONE).
 */
static inline size_t cnn_pooled_value_count(const CnnConfig* config) {
    if (config == NULL) return 0;
    if (config->pooling_mode == CNN_POOL_NONE) return 0;
    return (config->pooling_mode == CNN_POOL_DUAL) ?
        (config->filter_count * 2U) : config->filter_count;
}

/**
 * @brief Count projection (fully-connected) weights.
 */
static inline size_t cnn_projection_weight_count(const CnnConfig* config) {
    return config->feature_size * cnn_pooled_value_count(config);
}

#endif /* CNN_COMMON_H */
