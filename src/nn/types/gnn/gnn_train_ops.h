/**
 * @file gnn_train_ops.h
 * @brief Training-side API for the tiny GNN backend.
 */

#ifndef GNN_TRAIN_OPS_H
#define GNN_TRAIN_OPS_H

#include "gnn_infer_ops.h"

#include <stddef.h>

/**
 * @brief Training context wrapped around the inference-side GNN parameters.
 */
typedef struct {
    GnnInferContext* infer_ctx;     /**< Borrowed inference context holding live parameters. */
    GnnTrainConfig config;          /**< Stored hyperparameters chosen by generate_main.c. */
    float* hidden_cache;            /**< Cached node states for all message-passing stages. */
    float* input_weight_grad;       /**< Scratch gradient tensor for input projection weights. */
    float* input_bias_grad;         /**< Scratch gradient vector for input projection bias. */
    float* self_weight_grad;        /**< Scratch gradient tensor for self-transition weights. */
    float* message_weight_grad;     /**< Scratch gradient tensor for message weights. */
    float* message_bias_grad;       /**< Scratch gradient vector for message bias. */
    float* readout_primary_grad;    /**< Scratch gradient tensor for primary-anchor readout weights. */
    float* readout_secondary_grad;  /**< Scratch gradient tensor for secondary-anchor readout weights. */
    float* readout_neighbor_grad;   /**< Scratch gradient tensor for slot-neighbor readout weights. */
    float* output_bias_grad;        /**< Scratch gradient vector for readout bias. */
    size_t total_steps;             /**< Count of successful training updates. */
    size_t total_epochs;            /**< Count reported through train_get_stats. */
    float cumulative_loss;          /**< Running loss accumulator for averages. */
    float average_loss;             /**< Average loss seen so far. */
    float last_loss;                /**< Loss from the most recent update. */
} GnnTrainContext;

GnnTrainContext* nn_gnn_train_create(void* infer_ctx, const GnnTrainConfig* config);
void nn_gnn_train_destroy(GnnTrainContext* context);
int nn_gnn_train_step_with_data(GnnTrainContext* context, const float* input, const float* target);
int nn_gnn_train_step_with_output_gradient(
    GnnTrainContext* context,
    const float* input,
    const float* output_gradient,
    float* input_gradient
);
void nn_gnn_train_get_stats(
    GnnTrainContext* context,
    size_t* out_epochs,
    size_t* out_steps,
    float* out_avg_loss
);
int nn_gnn_train_step(void* context);

#endif
