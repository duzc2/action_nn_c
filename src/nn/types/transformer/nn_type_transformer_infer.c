#include "nn_infer_registry.h"
#include "transformer_infer_ops.h"

#include <stdlib.h>

static void* nn_type_transformer_infer_create_codegen(const NNCodegenInferConfig* config) {
    TransformerInferContext* context;
    size_t model_dim = 24U;
    uint32_t seed = 42U;

    if (config != 0) {
        if (config->input_size > 0U && config->input_size <= TRANSFORMER_MAX_MODEL_DIM) {
            model_dim = config->input_size;
        }
        if (config->seed != 0U) {
            seed = config->seed;
        }
    }

    context = (TransformerInferContext*)calloc(1U, sizeof(TransformerInferContext));
    if (context == 0) {
        return 0;
    }
    if (nn_transformer_init_parameters(context, model_dim, seed) != 0) {
        free(context);
        return 0;
    }

    if (config != 0) {
        context->expected_network_hash = config->network_hash;
        context->expected_layout_hash = config->layout_hash;
    }

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
    .type_name = "transformer",
    .infer_step = nn_transformer_infer_step,
    .create = nn_type_transformer_infer_create_codegen,
    .destroy = nn_type_transformer_infer_destroy_codegen,
    .auto_run = nn_type_transformer_infer_auto_run_codegen,
    .load_weights = nn_transformer_load_weights,
    .save_weights = nn_transformer_save_weights
};
