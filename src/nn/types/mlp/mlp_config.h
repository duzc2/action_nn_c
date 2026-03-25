/**
 * @file mlp_config.h
 * @brief MLP type-specific configuration definitions
 */

#ifndef MLP_CONFIG_H
#define MLP_CONFIG_H

#include "mlp_layers.h"

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Optimizer type
 */
typedef enum {
    MLP_OPT_SGD = 0,
    MLP_OPT_ADAM = 1
} MlpOptimizerType;

/**
 * @brief Loss function type
 */
typedef enum {
    MLP_LOSS_MSE = 0,
    MLP_LOSS_CROSS_ENTROPY = 1
} MlpLossType;

/**
 * @brief MLP network configuration
 */
typedef struct {
    size_t input_size;
    size_t hidden_layer_count;
    size_t hidden_sizes[4];
    size_t output_size;
    MlpActivationType hidden_activation;
    MlpActivationType output_activation;
} MlpConfig;

/**
 * @brief MLP training configuration
 */
typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    MlpOptimizerType optimizer;
    MlpLossType loss_func;
    size_t batch_size;
    uint32_t seed;
} MlpTrainConfig;

#endif
