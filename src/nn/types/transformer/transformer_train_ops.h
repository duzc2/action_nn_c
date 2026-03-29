/**
 * @file transformer_train_ops.h
 * @brief Training-side API for the dynamically sized transformer backend.
 */

#ifndef TRANSFORMER_TRAIN_OPS_H
#define TRANSFORMER_TRAIN_OPS_H

#include "transformer_config.h"
#include "transformer_infer_ops.h"

#include <stddef.h>

/**
 * @brief Minimal training context wrapped around the inference parameters.
 *
 * The transformer keeps training state intentionally compact so generated code
 * can reuse the same backend in standalone and composed-graph scenarios.
 */
typedef struct {
    TransformerInferContext* infer_ctx; /**< Borrowed inference context holding parameters. */
    const char* current_question;       /**< Current training question string. */
    const char* current_answer;         /**< Current target answer string. */
    float learning_rate;                /**< Scalar learning rate for gradient updates. */
    size_t total_epochs;                /**< Cumulative epoch counter for reporting. */
    size_t total_steps;                 /**< Cumulative step counter for reporting. */
    float cumulative_loss;              /**< Running loss sum for average tracking. */
    float average_loss;                 /**< Current average loss estimate. */
    float last_loss;                    /**< Most recent step loss for debugging/reporting. */
} TransformerTrainContext;

int nn_transformer_train_step(void* context);
int nn_transformer_train_step_with_output_gradient(
    TransformerTrainContext* context,
    const float* input,
    const float* output_gradient,
    float* input_gradient
);

#endif
