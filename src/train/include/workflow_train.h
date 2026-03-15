#ifndef WORKFLOW_TRAIN_H
#define WORKFLOW_TRAIN_H

#include <stddef.h>

#include "../../include/workflow_status.h"

typedef struct WorkflowTrainOptions {
    const char* csv_path;
    const char* vocab_path;
    const char* out_weights_bin;
    const char* out_weights_c;
    const char* out_symbol;
    size_t epochs;
    float learning_rate;
} WorkflowTrainOptions;

int workflow_train_from_csv(const WorkflowTrainOptions* options);

typedef struct WorkflowTrainSample {
    const char* command;
    const float* state;
    const float* target;
} WorkflowTrainSample;

typedef struct WorkflowTrainMemoryOptions {
    const WorkflowTrainSample* samples;
    size_t sample_count;
    const char* vocab_path;
    const char* out_weights_bin;
    const char* out_weights_c;
    const char* out_symbol;
    size_t epochs;
    float learning_rate;
} WorkflowTrainMemoryOptions;

int workflow_train_from_memory(const WorkflowTrainMemoryOptions* options);

#endif
