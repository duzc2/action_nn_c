/**
 * @file cnn_dual_pool_config.h
 * @brief POD-style configuration shared by the dual-pool CNN backend and generated code.
 *
 * This variant keeps the same compact frame encoder shape as the baseline CNN,
 * but each convolution filter contributes two pooled summaries: one global
 * average response and one global maximum response. The profiler only needs
 * stable plain-old-data metadata, so the config stays self-contained and easy
 * to copy into generated code.
 */

#ifndef CNN_DUAL_POOL_CONFIG_H
#define CNN_DUAL_POOL_CONFIG_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    CNN_DUAL_POOL_ACT_NONE = 0,
    CNN_DUAL_POOL_ACT_RELU = 1,
    CNN_DUAL_POOL_ACT_TANH = 2
} CnnDualPoolActivationType;

typedef struct {
    size_t total_input_size;
    size_t sequence_length;
    size_t frame_width;
    size_t frame_height;
    size_t channel_count;
    size_t kernel_size;
    size_t filter_count;
    size_t feature_size;
    CnnDualPoolActivationType pooling_activation;
    CnnDualPoolActivationType output_activation;
    uint32_t seed;
} CnnDualPoolConfig;

typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    size_t batch_size;
    uint32_t seed;
} CnnDualPoolTrainConfig;

#endif
