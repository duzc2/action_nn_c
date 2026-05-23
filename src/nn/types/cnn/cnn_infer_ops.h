/**
 * @file cnn_infer_ops.h
 * @brief Public inference-side API for the tiny CNN backend.
 */

#ifndef CNN_INFER_OPS_H
#define CNN_INFER_OPS_H

#include "cnn_config.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @brief Inference context for the tiny CNN leaf.
 *
 * The context owns trainable parameters, reusable I/O buffers, and the hash
 * metadata needed to validate weight files emitted by generated wrappers.
 */
typedef struct {
    CnnConfig config;               /**< Reconstructed type-specific configuration blob. */
    uint64_t expected_network_hash; /**< Hash guard for cross-network weight reuse. */
    uint64_t expected_layout_hash;  /**< Hash guard for parameter layout compatibility. */
    uint32_t rng_state;             /**< Deterministic RNG state for parameter init. */
    float* conv_weights;            /**< [filter][channel][ky][kx] flattened tensor. */
    float* conv_bias;               /**< One scalar bias per convolution filter. */
    float* projection_weights;      /**< [feature][pooled_dim] projection tensor. */
    float* projection_bias;         /**< One scalar bias per projected feature. */
    float* input_buffer;            /**< Owned copy of the latest flattened input. */
    float* output_buffer;           /**< Owned copy of the latest flattened output. */
    float* pooled_values;           /**< Reusable scratch buffer for pooled filter responses. */
    size_t* max_index_cache;        /**< Argmax positions per filter (used by dual/max pool backprop). */
    float* bn_running_mean;         /**< Per-filter running mean [filter_count] */
    float* bn_running_var;          /**< Per-filter running variance [filter_count] */
    float* bn_gamma;                /**< Per-filter learnable scale [filter_count] */
    float* bn_beta;                 /**< Per-filter learnable shift [filter_count] */
    float* bn_training_pre_cache;   /**< Set by train_create: points to spatial BN x_hat buffer for graph_run */
    float* bn_training_spatial_var; /**< Set by train_create: points to per-filter spatial variance values */
} CnnInferContext;

CnnInferContext* nn_cnn_infer_create_with_config(const CnnConfig* config, uint32_t seed);
void nn_cnn_infer_destroy(void* context);
void nn_cnn_infer_set_input(void* context, const float* input, size_t size);
void nn_cnn_infer_get_output(void* context, float* output, size_t size);
int nn_cnn_infer_step(void* context);
int nn_cnn_infer_auto_run(void* context, const float* input, float* output);

/**
 * @brief Shared forward helper reused by both inference and training code.
 *
 * Any cache pointer may be NULL when the caller only needs the final output.
 */
int nn_cnn_forward_pass(
    CnnInferContext* restrict context,
    const float* restrict input,
    float* restrict output,
    float* restrict pooled_linear_cache,
    float* restrict pooled_activation_cache,
    size_t* restrict max_index_cache,
    float* restrict output_linear_cache,
    float* restrict bn_pre_cache,
    float* restrict bn_spatial_var,
    float dropout_rate,
    float* restrict dropout_mask
);

int nn_cnn_load_weights(void* context, FILE* fp);
int nn_cnn_save_weights(void* context, FILE* fp);
uint64_t nn_cnn_get_network_hash(const void* context);

#endif
