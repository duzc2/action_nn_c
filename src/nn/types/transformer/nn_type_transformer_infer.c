#include "nn_infer_registry.h"
#include "transformer_infer_ops.h"

#include <stdlib.h>

static void* nn_type_transformer_infer_create_codegen(const NNCodegenInferConfig* config) {
    TransformerInferContext* context;
    (void)config;

    context = (TransformerInferContext*)calloc(1U, sizeof(TransformerInferContext));
    return context;
}

static void nn_type_transformer_infer_destroy_codegen(void* context) {
    free(context);
}

static int nn_type_transformer_infer_auto_run_codegen(void* context, const void* input, void* output) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;

    if (infer_ctx == 0 || input == 0 || output == 0) {
        return -1;
    }

    infer_ctx->question = (const char*)input;
    infer_ctx->answer = (char*)output;
    infer_ctx->answer_capacity = 256U;
    return nn_transformer_infer_step(infer_ctx);
}

const NNInferRegistryEntry nn_type_transformer_infer_entry = {
    "transformer",
    nn_transformer_infer_step,
    nn_type_transformer_infer_create_codegen,
    nn_type_transformer_infer_destroy_codegen,
    nn_type_transformer_infer_auto_run_codegen
};
