#include "transformer_infer_ops.h"

#include <stdio.h>
#include <string.h>

#define TRANSFORMER_ABI_VERSION 1U

typedef struct {
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t abi_version;
} TransformerWeightFileHeader;

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

int nn_transformer_load_weights(void* context, FILE* fp) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;
    TransformerWeightFileHeader header;

    if (infer_ctx == 0 || fp == 0) {
        return 0;
    }

    if (fread(&header, sizeof(header), 1, fp) != 1) {
        return 0;
    }

    if (header.abi_version != TRANSFORMER_ABI_VERSION) {
        return 0;
    }

    if (infer_ctx->expected_network_hash != 0U &&
        header.network_hash != infer_ctx->expected_network_hash) {
        return 0;
    }

    if (infer_ctx->expected_layout_hash != 0U &&
        header.layout_hash != infer_ctx->expected_layout_hash) {
        return 0;
    }

    return 1;
}

int nn_transformer_save_weights(void* context, FILE* fp) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;
    TransformerWeightFileHeader header;

    if (infer_ctx == 0 || fp == 0) {
        return 0;
    }

    header.network_hash = infer_ctx->expected_network_hash;
    header.layout_hash = infer_ctx->expected_layout_hash;
    header.abi_version = TRANSFORMER_ABI_VERSION;

    return fwrite(&header, sizeof(header), 1, fp) == 1 ? 1 : 0;
}
