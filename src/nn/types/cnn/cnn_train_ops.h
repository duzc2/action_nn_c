/**
 * @file cnn_train_ops.h
 * @brief Training-side API for the tiny CNN backend.
 */

#ifndef CNN_TRAIN_OPS_H
#define CNN_TRAIN_OPS_H

#include "cnn_infer_ops.h"

#include <stddef.h>

/**
 * @brief Training context wrapped around the inference-side CNN parameters.
 */
typedef struct {
    CnnInferContext* infer_ctx;     /**< Borrowed inference context holding live parameters. */
    CnnTrainConfig config;          /**< Stored hyperparameters chosen by generate_main.c. */
    float* pooled_linear_cache;     /**< Per-step pooled linear responses before activation. */
    float* pooled_activation_cache; /**< Per-step pooled responses after activation. */
    float* output_linear_cache;     /**< Per-step projected feature logits. */
    float* pooled_gradient_cache;   /**< Scratch dL/d(pool activation) buffer. */
    float* conv_weight_grad;        /**< Scratch convolution gradient tensor. */
    float* conv_bias_grad;          /**< Scratch convolution bias gradient tensor. */
    float* projection_weight_grad;  /**< Scratch projection gradient tensor. */
    float* projection_bias_grad;    /**< Scratch projection bias gradient vector. */
    size_t total_steps;             /**< Count of successful training updates. */
    size_t total_epochs;            /**< Count reported through train_get_stats. */
    float cumulative_loss;          /**< Running loss accumulator for averages. */
    float average_loss;             /**< Average loss seen so far. */
    float last_loss;                /**< Loss from the most recent supervised step. */
} CnnTrainContext;

CnnTrainContext* nn_cnn_train_create(void* infer_ctx, const CnnTrainConfig* config);
void nn_cnn_train_destroy(CnnTrainContext* context);
int nn_cnn_train_step_with_data(CnnTrainContext* context, const float* input, const float* target);
int nn_cnn_train_step_with_output_gradient(
    CnnTrainContext* context,
    const float* input,
    const float* output_gradient,
    float* input_gradient
);
void nn_cnn_train_get_stats(
    CnnTrainContext* context,
    size_t* out_epochs,
    size_t* out_steps,
    float* out_avg_loss
);
int nn_cnn_train_step(void* context);

#endif
