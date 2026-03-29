/**
 * @file mlp_config.h
 * @brief MLP type-specific configuration definitions
 */

#ifndef MLP_CONFIG_H
#define MLP_CONFIG_H

#include "mlp_layers.h"

#include <stddef.h>
#include <stdlib.h>
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
 * @brief Variable-sized MLP network configuration blob.
 *
 * The hidden layer widths are stored immediately after the fixed header so the
 * profiler can copy one self-contained byte blob without imposing a builtin
 * cap on network depth.
 */
typedef struct {
    size_t input_size;
    size_t hidden_layer_count;
    size_t output_size;
    MlpActivationType hidden_activation;
    MlpActivationType output_activation;
} MlpConfig;

/**
 * @brief Return the exact byte size required for one MLP config blob.
 */
static inline size_t mlp_config_size_for_hidden_layers(size_t hidden_layer_count) {
    return sizeof(MlpConfig) + (hidden_layer_count * sizeof(size_t));
}

/**
 * @brief Return a writable pointer to the trailing hidden-size array.
 */
static inline size_t* mlp_config_hidden_sizes_mut(MlpConfig* config) {
    return config == NULL
        ? NULL
        : (size_t*)((unsigned char*)config + sizeof(MlpConfig));
}

/**
 * @brief Return a read-only pointer to the trailing hidden-size array.
 */
static inline const size_t* mlp_config_hidden_sizes_view(const MlpConfig* config) {
    return config == NULL
        ? NULL
        : (const size_t*)((const unsigned char*)config + sizeof(MlpConfig));
}

/**
 * @brief Allocate one profiler-ready MLP config blob on the heap.
 */
static inline MlpConfig* mlp_config_create(size_t hidden_layer_count) {
    size_t total_size = mlp_config_size_for_hidden_layers(hidden_layer_count);
    return (MlpConfig*)calloc(1U, total_size);
}

/**
 * @brief Initialize one allocated MLP config blob from caller parameters.
 */
static inline int mlp_config_init(
    MlpConfig* config,
    size_t input_size,
    size_t hidden_layer_count,
    const size_t* hidden_sizes,
    size_t output_size,
    MlpActivationType hidden_activation,
    MlpActivationType output_activation
) {
    size_t hidden_index;
    size_t* stored_hidden_sizes;

    if (config == NULL) {
        return -1;
    }
    if (hidden_layer_count > 0U && hidden_sizes == NULL) {
        return -1;
    }

    config->input_size = input_size;
    config->hidden_layer_count = hidden_layer_count;
    config->output_size = output_size;
    config->hidden_activation = hidden_activation;
    config->output_activation = output_activation;
    stored_hidden_sizes = mlp_config_hidden_sizes_mut(config);

    for (hidden_index = 0U; hidden_index < hidden_layer_count; ++hidden_index) {
        stored_hidden_sizes[hidden_index] = hidden_sizes[hidden_index];
    }

    return 0;
}

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
