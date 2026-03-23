/**
 * @file mlp_layers.h
 * @brief MLP layer mathematics implementation
 *
 * Implements:
 * - Dense layer forward propagation
 * - Activation functions (ReLU, Sigmoid, Tanh, Softmax)
 * - Xavier weight initialization
 */

#ifndef MLP_LAYERS_H
#define MLP_LAYERS_H

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Activation function types
 */
typedef enum {
    MLP_ACT_NONE = 0,
    MLP_ACT_RELU = 1,
    MLP_ACT_SIGMOID = 2,
    MLP_ACT_TANH = 3,
    MLP_ACT_SOFTMAX = 4,
    MLP_ACT_LEAKY_RELU = 5
} MlpActivationType;

/**
 * @brief Dense layer parameters
 */
typedef struct {
    size_t input_size;
    size_t output_size;
    float* weights;
    float* bias;
    MlpActivationType activation;
} MlpDenseLayer;

/**
 * @brief Activation function pointer type
 */
typedef float (*ActivationFn)(float x);

/**
 * @brief Apply element-wise activation function
 *
 * @param output Output array
 * @param input Input array
 * @param size Number of elements
 * @param act_type Activation type
 */
void mlp_activation(float* output, const float* input, size_t size, MlpActivationType act_type);

/**
 * @brief Dense layer forward propagation: output = activation(W * input + b)
 *
 * @param layer Layer parameters
 * @param output Output array (size: layer->output_size)
 * @param input Input array (size: layer->input_size)
 */
void mlp_dense_forward(const MlpDenseLayer* layer, float* output, const float* input);

/**
 * @brief Initialize layer with Xavier/Glorot initialization
 *
 * @param layer Layer to initialize
 * @param seed Random seed for reproducibility
 */
void mlp_dense_init(MlpDenseLayer* layer, uint32_t seed);

/**
 * @brief Allocate and create a dense layer
 *
 * @param input_size Number of input features
 * @param output_size Number of output features
 * @param activation Activation function type
 * @param seed Random seed
 * @return New layer, or NULL on allocation failure
 */
MlpDenseLayer* mlp_dense_create(
    size_t input_size,
    size_t output_size,
    MlpActivationType activation,
    uint32_t seed
);

/**
 * @brief Free dense layer
 *
 * @param layer Layer to free
 */
void mlp_dense_free(MlpDenseLayer* layer);

/**
 * @brief Get activation function by type
 *
 * @param act_type Activation type
 * @return Activation function pointer
 */
ActivationFn mlp_get_activation(MlpActivationType act_type);

#endif
