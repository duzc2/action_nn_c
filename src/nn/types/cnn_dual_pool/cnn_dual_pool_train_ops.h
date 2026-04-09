/**
 * @file cnn_dual_pool_train_ops.h
 * @brief Training-side API for the dual-pool CNN backend.
 */

#ifndef CNN_DUAL_POOL_TRAIN_OPS_H
#define CNN_DUAL_POOL_TRAIN_OPS_H

#include "cnn_dual_pool_infer_ops.h"

#include <stddef.h>

typedef struct {
    CnnDualPoolInferContext* infer_ctx;
    CnnDualPoolTrainConfig config;
    float* pooled_linear_cache;
    float* pooled_activation_cache;
    size_t* max_index_cache;
    float* output_linear_cache;
    float* pooled_gradient_cache;
    float* conv_weight_grad;
    float* conv_bias_grad;
    float* projection_weight_grad;
    float* projection_bias_grad;
    size_t total_steps;
    size_t total_epochs;
    float cumulative_loss;
    float average_loss;
    float last_loss;
} CnnDualPoolTrainContext;

CnnDualPoolTrainContext* nn_cnn_dual_pool_train_create(void* infer_ctx, const CnnDualPoolTrainConfig* config);
void nn_cnn_dual_pool_train_destroy(CnnDualPoolTrainContext* context);
int nn_cnn_dual_pool_train_step_with_data(CnnDualPoolTrainContext* context, const float* input, const float* target);
int nn_cnn_dual_pool_train_step_with_output_gradient(CnnDualPoolTrainContext* context, const float* input, const float* output_gradient, float* input_gradient);
void nn_cnn_dual_pool_train_get_stats(CnnDualPoolTrainContext* context, size_t* out_epochs, size_t* out_steps, float* out_avg_loss);
int nn_cnn_dual_pool_train_step(void* context);

#endif
