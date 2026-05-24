/**
 * @file cnn_config.h
 * @brief POD-style configuration shared by the tiny CNN backend and generated code.
 *
 * The CNN supports four pooling modes controlled by an enum:
 *   CNN_POOL_AVG  – average pooling (original CNN behaviour)
 *   CNN_POOL_MAX  – maximum pooling
 *   CNN_POOL_DUAL – both average and maximum pooling per filter (formerly cnn_dual_pool)
 *   CNN_POOL_NONE – no pooling, outputs 2D feature maps for multi-layer cascading
 *
 * The profiler only needs stable plain-old-data metadata, so this header keeps
 * the config self-contained and serialization friendly.
 */

#ifndef CNN_CONFIG_H
#define CNN_CONFIG_H

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Activation types used by the tiny CNN backend.
 */
typedef enum {
    CNN_ACT_NONE       = 0,
    CNN_ACT_RELU       = 1,
    CNN_ACT_TANH       = 2,
    CNN_ACT_RELU6      = 3,  /**< Clipped ReLU: min(max(0, x), 6) */
    CNN_ACT_LEAKY_RELU = 4   /**< Leaky ReLU: max(x, 0.01*x), alpha=0.01 */
} CnnActivationType;

/**
 * @brief Global pooling variant applied after convolution.
 */
typedef enum {
    CNN_POOL_AVG  = 0, /**< Global average pooling (one scalar per filter). */
    CNN_POOL_MAX  = 1, /**< Global maximum pooling (one scalar per filter). */
    CNN_POOL_DUAL = 2, /**< Both average and maximum pooling (two scalars per filter). */
    CNN_POOL_NONE = 3  /**< No pooling – outputs 2D feature maps for multi-layer cascading. */
} CnnPoolingMode;

/**
 * @brief Convolution mode.
 */
typedef enum {
    CNN_CONV_STANDARD  = 0,  /**< Standard conv: F filters, each sees all C channels */
    CNN_CONV_DEPTHWISE = 1   /**< Depthwise: C filters, each sees exactly 1 channel */
} CnnConvMode;

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
    CnnActivationType pooling_activation; /**< Activation applied after global pooling. */
    CnnActivationType output_activation;  /**< Activation applied to projected features. */
    CnnPoolingMode pooling_mode;          /**< Global pooling variant (default AVG). */
    CnnConvMode conv_mode;                /**< Convolution mode (default STANDARD). */
    size_t stride;                        /**< Convolution stride (1 = no skip, 2 = half size). */
    int use_batch_norm;                   /**< 0 = no BN, 1 = BN after conv, before activation */
    float bn_momentum;                    /**< Running mean/variance update rate (typical 0.9) */
    float bn_epsilon;                     /**< Numerical stability constant (typical 1e-5) */
    uint32_t seed;                        /**< Deterministic parameter initialization seed. */
} CnnConfig;

/**
 * @brief Minimal train-time hyperparameters for the tiny CNN backend.
 */
typedef struct {
    float learning_rate;                  /**< Step size used by the simple SGD update. */
    float momentum;                       /**< Reserved for future optimizer growth. */
    float weight_decay;                   /**< L2-style decay applied to trainable weights. */
    float bias_weight_decay;              /**< L2-style decay applied to bias vectors. */
    float dropout_rate;                   /**< Dropout rate (0.0 = disabled). Applied between GAP pooling and MLP projection. */
    uint32_t batch_size;                  /**< Batch size requested by generated wrappers. */
    uint32_t seed;                        /**< Reserved deterministic seed for future train state. */
} CnnTrainConfig;

#endif
