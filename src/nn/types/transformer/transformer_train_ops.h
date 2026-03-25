#ifndef TRANSFORMER_TRAIN_OPS_H
#define TRANSFORMER_TRAIN_OPS_H

#include "transformer_infer_ops.h"

#include <stddef.h>

typedef struct {
    TransformerInferContext* infer_ctx;
    const char* current_question;
    const char* current_answer;
    float learning_rate;
    size_t total_epochs;
    size_t total_steps;
    float cumulative_loss;
    float average_loss;
    float last_loss;
} TransformerTrainContext;

int nn_transformer_train_step(void* context);

#endif
