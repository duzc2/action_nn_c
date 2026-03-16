#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../include/config_user.h"
#include "../include/ops.h"
#include "../include/tensor.h"
#include "../include/tokenizer.h"
#include "../include/weights_io.h"
#include "./include/workflow_infer.h"

enum {
    TOKEN_WEIGHT_COUNT = VOCAB_SIZE * OUTPUT_DIM,
    STATE_WEIGHT_COUNT = STATE_DIM * OUTPUT_DIM,
    BIAS_COUNT = OUTPUT_DIM,
    TOTAL_WEIGHT_COUNT = TOKEN_WEIGHT_COUNT + STATE_WEIGHT_COUNT + BIAS_COUNT
};

static int activate_output(const float* logits, float* out_act) {
    const int activations[OUTPUT_DIM] = IO_MAPPING_ACTIVATIONS;
    Tensor in_t;
    Tensor out_t;
    size_t shape[1] = { OUTPUT_DIM };
    int rc = tensor_init_view(&in_t, (float*)logits, 1U, shape);
    if (rc != TENSOR_STATUS_OK) {
        return WORKFLOW_STATUS_INTERNAL_ERROR;
    }
    rc = tensor_init_view(&out_t, out_act, 1U, shape);
    if (rc != TENSOR_STATUS_OK) {
        return WORKFLOW_STATUS_INTERNAL_ERROR;
    }
    rc = op_actuator(&in_t, activations, &out_t);
    return (rc == TENSOR_STATUS_OK) ? WORKFLOW_STATUS_OK : WORKFLOW_STATUS_INTERNAL_ERROR;
}

static void predict_logits(const float* w,
                           const int* ids,
                           size_t token_count,
                           const float* state,
                           float* logits) {
    size_t i = 0U;
    size_t j = 0U;
    size_t token_base = 0U;
    size_t state_base = TOKEN_WEIGHT_COUNT;
    size_t bias_base = TOKEN_WEIGHT_COUNT + STATE_WEIGHT_COUNT;
    for (j = 0U; j < OUTPUT_DIM; ++j) {
        logits[j] = w[bias_base + j];
    }
    for (i = 0U; i < token_count; ++i) {
        size_t id = (ids[i] >= 0) ? (size_t)ids[i] : 0U;
        if (id >= VOCAB_SIZE) {
            id = 0U;
        }
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            logits[j] += w[token_base + id * OUTPUT_DIM + j] / (float)token_count;
        }
    }
    for (i = 0U; i < STATE_DIM; ++i) {
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            logits[j] += state[i] * w[state_base + i * OUTPUT_DIM + j];
        }
    }
}

int workflow_prepare_tokenizer(const char* vocab_path, Vocabulary* vocab, Tokenizer* tokenizer) {
    int rc = 0;
    if (vocab_path == NULL || vocab == NULL || tokenizer == NULL) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    memset(vocab, 0, sizeof(*vocab));
    memset(tokenizer, 0, sizeof(*tokenizer));
    rc = vocab_load_text(vocab_path, vocab);
    if (rc != TOKENIZER_STATUS_OK) {
        return WORKFLOW_STATUS_IO_ERROR;
    }
    rc = tokenizer_init(tokenizer, vocab, 0);
    if (rc != TOKENIZER_STATUS_OK) {
        vocab_free(vocab);
        return WORKFLOW_STATUS_DATA_ERROR;
    }
    return WORKFLOW_STATUS_OK;
}

size_t workflow_weights_count(void) {
    return TOTAL_WEIGHT_COUNT;
}

int workflow_runtime_init(WorkflowRuntime* runtime, const char* vocab_path, const char* weights_bin_path) {
    int rc = 0;
    if (runtime == NULL || vocab_path == NULL || weights_bin_path == NULL) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    memset(runtime, 0, sizeof(*runtime));
    rc = weights_load_binary(weights_bin_path, &runtime->weights, &runtime->weight_count);
    if (rc != WEIGHTS_IO_STATUS_OK || runtime->weights == NULL || runtime->weight_count != TOTAL_WEIGHT_COUNT) {
        free(runtime->weights);
        runtime->weights = NULL;
        runtime->weight_count = 0U;
        return WORKFLOW_STATUS_IO_ERROR;
    }
    rc = workflow_prepare_tokenizer(vocab_path, &runtime->vocab, &runtime->tokenizer);
    if (rc != WORKFLOW_STATUS_OK) {
        free(runtime->weights);
        runtime->weights = NULL;
        runtime->weight_count = 0U;
        return rc;
    }
    runtime->ready = 1;
    return WORKFLOW_STATUS_OK;
}

void workflow_runtime_shutdown(WorkflowRuntime* runtime) {
    if (runtime == NULL) {
        return;
    }
    vocab_free(&runtime->vocab);
    free(runtime->weights);
    memset(runtime, 0, sizeof(*runtime));
}

int workflow_run_step(WorkflowRuntime* runtime, const char* command, const float* state, float* out_action) {
    int ids[MAX_SEQ_LEN];
    size_t count = 0U;
    float logits[OUTPUT_DIM];
    int rc = 0;
    if (runtime == NULL || command == NULL || state == NULL || out_action == NULL || runtime->ready != 1) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    if (command[0] == '\0') {
        return WORKFLOW_STATUS_DATA_ERROR;
    }
    memset(ids, 0, sizeof(ids));
    memset(logits, 0, sizeof(logits));
    rc = tokenizer_encode(&runtime->tokenizer, command, ids, MAX_SEQ_LEN, &count);
    if (rc != TOKENIZER_STATUS_OK || count == 0U) {
        return WORKFLOW_STATUS_DATA_ERROR;
    }
    predict_logits(runtime->weights, ids, count, state, logits);
    rc = activate_output(logits, out_action);
    return (rc == WORKFLOW_STATUS_OK) ? WORKFLOW_STATUS_OK : WORKFLOW_STATUS_INTERNAL_ERROR;
}

int workflow_run_step_goal(WorkflowRuntime* runtime, const float* state, float* out_action) {
    float logits[OUTPUT_DIM];
    int rc = 0;
    if (runtime == NULL || state == NULL || out_action == NULL || runtime->ready != 1) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    memset(logits, 0, sizeof(logits));
    predict_logits(runtime->weights, NULL, 0U, state, logits);
    rc = activate_output(logits, out_action);
    return (rc == WORKFLOW_STATUS_OK) ? WORKFLOW_STATUS_OK : WORKFLOW_STATUS_INTERNAL_ERROR;
}
