#ifndef WORKFLOW_INFER_H
#define WORKFLOW_INFER_H

#include <stddef.h>

#include "../../include/tokenizer.h"
#include "../../include/workflow_status.h"

typedef struct WorkflowRuntime {
    Vocabulary vocab;
    Tokenizer tokenizer;
    float* weights;
    size_t weight_count;
    int ready;
} WorkflowRuntime;

int workflow_prepare_tokenizer(const char* vocab_path, Vocabulary* vocab, Tokenizer* tokenizer);
size_t workflow_weights_count(void);
int workflow_runtime_init(WorkflowRuntime* runtime, const char* vocab_path, const char* weights_bin_path);
void workflow_runtime_shutdown(WorkflowRuntime* runtime);
int workflow_run_step(WorkflowRuntime* runtime, const char* command, const float* state, float* out_action);

#endif
