/**
 * @file cnn_dual_pool_infer_ops.h
 * @brief Public inference-side API for the dual-pool CNN backend.
 */

#ifndef CNN_DUAL_POOL_INFER_OPS_H
#define CNN_DUAL_POOL_INFER_OPS_H

#include "cnn_dual_pool_config.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    CnnDualPoolConfig config;
    uint64_t expected_network_hash;
    uint64_t expected_layout_hash;
    uint32_t rng_state;
    float* conv_weights;
    float* conv_bias;
    float* projection_weights;
    float* projection_bias;
    float* input_buffer;
    float* output_buffer;
    float* pooled_values;
} CnnDualPoolInferContext;

CnnDualPoolInferContext* nn_cnn_dual_pool_infer_create(void);
CnnDualPoolInferContext* nn_cnn_dual_pool_infer_create_with_config(const CnnDualPoolConfig* config, uint32_t seed);
void nn_cnn_dual_pool_infer_destroy(void* context);
void nn_cnn_dual_pool_infer_set_input(void* context, const float* input, size_t size);
void nn_cnn_dual_pool_infer_get_output(void* context, float* output, size_t size);
int nn_cnn_dual_pool_infer_step(void* context);
int nn_cnn_dual_pool_infer_auto_run(void* context, const float* input, float* output);
int nn_cnn_dual_pool_forward_pass(CnnDualPoolInferContext* context, const float* input, float* output, float* pooled_linear_cache, float* pooled_activation_cache, size_t* max_index_cache, float* output_linear_cache);
int nn_cnn_dual_pool_load_weights(void* context, FILE* fp);
int nn_cnn_dual_pool_save_weights(void* context, FILE* fp);
uint64_t nn_cnn_dual_pool_get_network_hash(const void* context);

#endif
