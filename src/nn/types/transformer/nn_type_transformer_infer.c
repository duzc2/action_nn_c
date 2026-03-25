#include "nn_infer_registry.h"
#include "transformer_infer_ops.h"

#include <stdlib.h>
#include <string.h>

static void* nn_type_transformer_infer_create_codegen(const NNCodegenInferConfig* config) {
    TransformerInferContext* context;
    TransformerModelConfig model_config;

    if (config == 0 ||
        config->type_config == 0 ||
        config->type_config_size != sizeof(TransformerModelConfig) ||
        config->type_config_type_name == 0 ||
        strcmp(config->type_config_type_name, "TransformerModelConfig") != 0) {
        return 0;
    }

    model_config = *(const TransformerModelConfig*)config->type_config;
    context = (TransformerInferContext*)calloc(1U, sizeof(TransformerInferContext));
    if (context == 0) {
        return 0;
    }
    if (nn_transformer_init_parameters(context, model_config.model_dim, model_config.seed) != 0) {
        free(context);
        return 0;
    }

    context->expected_network_hash = config->network_hash;
    context->expected_layout_hash = config->layout_hash;
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
