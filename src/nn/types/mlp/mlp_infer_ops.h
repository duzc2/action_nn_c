/**
 * @file mlp_infer_ops.h
 * @brief MLP inference operations interface
 *
 * Real MLP inference implementation with:
 * - Configurable network structure (layers, sizes, activation)
 * - Weight loading/saving with hash validation
 * - Forward propagation
 */

#ifndef MLP_INFER_OPS_H
#define MLP_INFER_OPS_H

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include "mlp_layers.h"

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
 * @brief MLP inference context
 */
typedef struct {
    MlpConfig config;
    MlpDenseLayer** layers;
    size_t layer_count;
    size_t max_buffer_size;
    uint64_t expected_network_hash;
    uint64_t expected_layout_hash;
    float* input_buffer;
    float* output_buffer;
    float* work_buffer_a;
    float* work_buffer_b;
    uint32_t seed;
} MlpInferContext;

/**
 * @brief Create MLP inference context without explicit configuration
 *
 * This framework now requires user-side code to provide a concrete
 * type configuration through the profiler pipeline. Calling this helper
 * without an explicit configuration returns NULL.
 *
 * @return NULL because explicit configuration is required
 */
MlpInferContext* nn_mlp_infer_create(void);

/**
 * @brief Create MLP with custom configuration
 *
 * @param config Network configuration
 * @param seed Random seed for weight initialization
 * @return New context, or NULL on failure
 */
MlpInferContext* nn_mlp_infer_create_with_config(const MlpConfig* config, uint32_t seed);

/**
 * @brief Free MLP inference context
 *
 * @param context Context to free
 */
void nn_mlp_infer_destroy(void* context);

/**
 * @brief Set input values for inference
 *
 * @param context MLP context
 * @param input Input array
 * @param size Input size (must match config.input_size)
 */
void nn_mlp_infer_set_input(void* context, const float* input, size_t size);

/**
 * @brief Get output values after inference
 *
 * @param context MLP context
 * @param output Output array (caller must provide buffer)
 * @param size Output size (must match config.output_size)
 */
void nn_mlp_infer_get_output(void* context, float* output, size_t size);

/**
 * @brief Run one inference step
 *
 * @param context MLP context
 * @return 0 on success, -1 on failure
 */
int nn_mlp_infer_step(void* context);

/**
 * @brief Run inference with auto loop (for testing)
 *
 * @param context MLP context
 * @param input Input array
 * @param output Output array
 * @return 0 on success, -1 on failure
 */
int nn_mlp_infer_auto_run(void* context, const float* input, float* output);

/**
 * @brief Load weights from file
 *
 * File format:
 * - Header: network_hash, layout_hash, abi_version
 * - Weights: layer weights and biases
 *
 * @param context MLP context
 * @param fp File pointer (opened by caller)
 * @return 1 on success, 0 on failure
 */
int nn_mlp_load_weights(void* context, FILE* fp);

/**
 * @brief Save weights to file
 *
 * @param context MLP context
 * @param fp File pointer (opened by caller)
 * @return 1 on success, 0 on failure
 */
int nn_mlp_save_weights(void* context, FILE* fp);

/**
 * @brief Get network hash for weight validation
 *
 * @param context MLP context
 * @return Network hash value
 */
uint64_t nn_mlp_get_network_hash(const void* context);

#endif
