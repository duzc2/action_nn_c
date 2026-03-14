#ifndef WORKFLOW_H
#define WORKFLOW_H

#include <stddef.h>

#include "tokenizer.h"

typedef enum WorkflowStatus {
    WORKFLOW_STATUS_OK = 0,
    WORKFLOW_STATUS_INVALID_ARGUMENT = -1,
    WORKFLOW_STATUS_IO_ERROR = -2,
    WORKFLOW_STATUS_DATA_ERROR = -3,
    WORKFLOW_STATUS_INTERNAL_ERROR = -4
} WorkflowStatus;

typedef struct WorkflowTrainOptions {
    const char* csv_path;
    const char* vocab_path;
    const char* out_weights_bin;
    const char* out_weights_c;
    const char* out_symbol;
    size_t epochs;
    float learning_rate;
} WorkflowTrainOptions;

typedef struct WorkflowLoopOptions {
    const char* vocab_path;
    const char* weights_bin_path;
    size_t max_frames;
    int (*build_input_callback)(size_t frame,
                                char* out_command,
                                size_t command_capacity,
                                float* out_state,
                                size_t state_dim,
                                void* user_data);
    int (*action_callback)(const float* action_values, size_t action_count, void* user_data);
    void* user_data;
} WorkflowLoopOptions;

int workflow_prepare_tokenizer(const char* vocab_path, Vocabulary* vocab, Tokenizer* tokenizer);
size_t workflow_weights_count(void);
int workflow_train_from_csv(const WorkflowTrainOptions* options);
int workflow_run_goal_loop(const WorkflowLoopOptions* options);

#endif
