/**
 * @file cnn_config.h
 * @brief POD-style configuration shared by the tiny CNN backend and generated code.
 *
 * The CNN added for the reaction demo is intentionally compact: it interprets
 * the flattened network input as a short sequence of 2D frames, applies one
 * shared convolution stage to each frame, performs global average pooling, and
 * then projects the pooled response into a small feature vector. The profiler
 * only needs stable plain-old-data metadata, so this header keeps the config
 * self-contained and serialization friendly.
 */

#ifndef CNN_CONFIG_H
#define CNN_CONFIG_H

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Activation types used by the tiny CNN backend.
 */
typedef enum {
    CNN_ACT_NONE = 0,
    CNN_ACT_RELU = 1,
    CNN_ACT_TANH = 2
} CnnActivationType;

/**
 * @brief Structural configuration required to build one CNN leaf.
 */
typedef struct {
    size_t total_input_size;              /**< Flattened input width seen by generated graph code. */
    size_t sequence_length;               /**< Number of frames packed into the input vector. */
    size_t frame_width;                   /**< Width of one frame in cells. */
    size_t frame_height;                  /**< Height of one frame in cells. */
    size_t channel_count;                 /**< Channel count for one frame. */
    size_t kernel_size;                   /**< Shared square convolution kernel size. */
    size_t filter_count;                  /**< Number of shared convolution filters. */
    size_t feature_size;                  /**< Projected per-frame feature width. */
    CnnActivationType pooling_activation; /**< Activation applied after global average pooling. */
    CnnActivationType output_activation;  /**< Activation applied to projected features. */
    uint32_t seed;                        /**< Deterministic parameter initialization seed. */
} CnnConfig;

/**
 * @brief Minimal train-time hyperparameters for the tiny CNN backend.
 */
typedef struct {
    float learning_rate;                  /**< Step size used by the simple SGD update. */
    float momentum;                       /**< Reserved for future optimizer growth. */
    float weight_decay;                   /**< L2-style decay applied to trainable weights. */
    size_t batch_size;                    /**< Batch size requested by generated wrappers. */
    uint32_t seed;                        /**< Reserved deterministic seed for future train state. */
} CnnTrainConfig;

#endif
