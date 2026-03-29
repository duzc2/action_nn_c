/**
 * @file nn_type_transformer_infer.c
 * @brief Registry bridge that exposes the tiny transformer inference backend.
 */

#include "nn_infer_registry.h"
#include "transformer_infer_ops.h"

#include <stdlib.h>
#include <string.h>

/**
 * @brief Reconstruct a transformer inference context from codegen metadata.
 */
static void* nn_type_transformer_infer_create_codegen(const NNCodegenInferConfig* config) {
    TransformerInferContext* context;
    TransformerModelConfig model_config;

    /* Accept only the exact typed config expected by this backend bridge. */
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
    if (nn_transformer_init_parameters(
            context,
            &model_config,
            config->input_size,
            config->output_size
        ) != 0) {
        nn_transformer_infer_destroy(context);
        return 0;
    }

    context->expected_network_hash = config->network_hash;
    context->expected_layout_hash = config->layout_hash;
    context->graph_input_size = config->input_size;
    context->graph_output_size = config->output_size;
    return context;
}

/**
 * @brief Release a transformer inference context created for generated code.
 */
static void nn_type_transformer_infer_destroy_codegen(void* context) {
    nn_transformer_infer_destroy(context);
}

/**
 * @brief Adapt raw codegen buffers to the text-oriented transformer infer API.
 */
static int nn_type_transformer_infer_auto_run_codegen(void* context, const void* input, void* output) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;

    if (infer_ctx == 0 || input == 0 || output == 0) {
        return -1;
    }

    infer_ctx->question = (const char*)input;
    infer_ctx->answer = (char*)output;
    infer_ctx->answer_capacity = infer_ctx->max_text_length;
    return nn_transformer_infer_step(infer_ctx);
}

/**
 * @brief Builtin registry entry published when the transformer backend is enabled.
 */
const NNInferRegistryEntry nn_type_transformer_infer_entry = {
    .type_name = "transformer",
    .infer_step = nn_transformer_infer_step,
    .create = nn_type_transformer_infer_create_codegen,
    .destroy = nn_type_transformer_infer_destroy_codegen,
    .auto_run = nn_type_transformer_infer_auto_run_codegen,
    .graph_run = nn_transformer_graph_run,
    .load_weights = nn_transformer_load_weights,
    .save_weights = nn_transformer_save_weights
};
