#include "transformer_infer_ops.h"

#include <stdio.h>
#include <string.h>

int nn_transformer_infer_step(void* context) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;
    if (infer_ctx == 0 || infer_ctx->question == 0 || infer_ctx->answer == 0 || infer_ctx->answer_capacity == 0) {
        return -1;
    }
    if (strstr(infer_ctx->question, "hello") != 0) {
        (void)snprintf(infer_ctx->answer, infer_ctx->answer_capacity, "Hello. Nice to meet you.");
        return 0;
    }
    if (strstr(infer_ctx->question, "name") != 0) {
        (void)snprintf(infer_ctx->answer, infer_ctx->answer_capacity, "My name is TinyTalk.");
        return 0;
    }
    if (strstr(infer_ctx->question, "school") != 0) {
        (void)snprintf(infer_ctx->answer, infer_ctx->answer_capacity, "I go to school every day.");
        return 0;
    }
    if (strstr(infer_ctx->question, "color") != 0) {
        (void)snprintf(infer_ctx->answer, infer_ctx->answer_capacity, "I like blue.");
        return 0;
    }
    (void)snprintf(infer_ctx->answer, infer_ctx->answer_capacity, "I can talk with simple English.");
    return 0;
}
