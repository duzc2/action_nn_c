/**
 * @file rnn_train_ops.h
 * @brief Training-side API for the tiny RNN backend.
 */

#ifndef RNN_TRAIN_OPS_H
#define RNN_TRAIN_OPS_H

#include "rnn_infer_ops.h"

#include <stddef.h>

/**
 * @brief Training context wrapped around the inference-side RNN parameters.
 */
typedef struct {
    RnnInferContext* infer_ctx;     /**< Borrowed inference context holding live parameters. */
    RnnTrainConfig config;          /**< Stored hyperparameters chosen by generate_main.c. */
    float* hidden_cache;            /**< Cached hidden states for all time steps. */
    float* output_linear_cache;     /**< Cached readout logits before output activation. */
    float* input_to_hidden_grad;    /**< Scratch gradient tensor for input weights. */
    float* hidden_to_hidden_grad;   /**< Scratch gradient tensor for recurrent weights. */
    float* hidden_bias_grad;        /**< Scratch gradient vector for hidden bias. */
    float* hidden_to_output_grad;   /**< Scratch gradient tensor for output weights. */
    float* output_bias_grad;        /**< Scratch gradient vector for output bias. */
    size_t total_steps;             /**< Count of successful training updates. */
    size_t total_epochs;            /**< Count reported through train_get_stats. */
    float cumulative_loss;          /**< Running loss accumulator for averages. */
    float average_loss;             /**< Average loss seen so far. */
    float last_loss;                /**< Loss from the most recent supervised step. */
} RnnTrainContext;

RnnTrainContext* nn_rnn_train_create(void* infer_ctx, const RnnTrainConfig* config);
void nn_rnn_train_destroy(RnnTrainContext* context);
int nn_rnn_train_step_with_data(RnnTrainContext* context, const float* input, const float* target);
int nn_rnn_train_step_with_output_gradient(
    RnnTrainContext* context,
    const float* input,
    const float* output_gradient,
    float* input_gradient
);
void nn_rnn_train_get_stats(
    RnnTrainContext* context,
    size_t* out_epochs,
    size_t* out_steps,
    float* out_avg_loss
);
int nn_rnn_train_step(void* context);

#endif
