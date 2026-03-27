/**
 * @file rnn_config.h
 * @brief POD-style configuration shared by the tiny RNN backend and generated code.
 */

#ifndef RNN_CONFIG_H
#define RNN_CONFIG_H

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Activation types used by the tiny RNN backend.
 */
typedef enum {
    RNN_ACT_NONE = 0,
    RNN_ACT_TANH = 1
} RnnActivationType;

/**
 * @brief Structural configuration required to build one RNN leaf.
 */
typedef struct {
    size_t sequence_length;               /**< Number of time steps in one sample. */
    size_t input_feature_size;            /**< Width of one per-step input feature vector. */
    size_t hidden_size;                   /**< Width of the recurrent hidden state. */
    size_t output_size;                   /**< Width of the final control vector. */
    RnnActivationType hidden_activation;  /**< Activation applied to recurrent state updates. */
    RnnActivationType output_activation;  /**< Activation applied to final control axes. */
    uint32_t seed;                        /**< Deterministic parameter initialization seed. */
} RnnConfig;

/**
 * @brief Minimal train-time hyperparameters for the tiny RNN backend.
 */
typedef struct {
    float learning_rate;                  /**< Step size used by the simple SGD update. */
    float momentum;                       /**< Reserved for future optimizer growth. */
    float weight_decay;                   /**< L2-style decay applied to trainable weights. */
    size_t batch_size;                    /**< Batch size requested by generated wrappers. */
    uint32_t seed;                        /**< Reserved deterministic seed for future train state. */
} RnnTrainConfig;

#endif
