/**
 * @file nn_type_cnn_infer.c
 * @brief Registry bridge that exposes the CNN inference backend to codegen.
 */

#include "nn_infer_registry.h"
#include "cnn_infer_ops.h"

#include <string.h>

/**
 * @brief Reconstruct a typed CNN inference context from generated metadata.
 */
static void* nn_type_cnn_infer_create_codegen(const NNCodegenInferConfig* config) {
    CnnConfig typed_config;
    CnnInferContext* context;
    uint32_t seed;

    if (config == 0 ||
        config->type_config == 0 ||
        config->type_config_size != sizeof(CnnConfig) ||
        config->type_config_type_name == 0 ||
        strcmp(config->type_config_type_name, "CnnConfig") != 0) {
        return 0;
    }

    typed_config = *(const CnnConfig*)config->type_config;
    seed = config->seed != 0U ? config->seed : typed_config.seed;
    context = nn_cnn_infer_create_with_config(&typed_config, seed);
    if (context != 0) {
        context->expected_network_hash = config->network_hash;
        context->expected_layout_hash = config->layout_hash;
    }
    return context;
}

/**
 * @brief Adapt raw graph buffers to the float-based CNN helper.
 */
static int nn_type_cnn_infer_auto_run_codegen(void* context, const void* input, void* output) {
    return nn_cnn_infer_auto_run(context, (const float*)input, (float*)output);
}

/**
 * @brief Builtin registry entry published when the CNN backend is enabled.
 */
const NNInferRegistryEntry nn_type_cnn_infer_entry = {
    .type_name = "cnn",
    .infer_step = nn_cnn_infer_step,
    .create = nn_type_cnn_infer_create_codegen,
    .destroy = nn_cnn_infer_destroy,
    .auto_run = nn_type_cnn_infer_auto_run_codegen,
    .graph_run = nn_type_cnn_infer_auto_run_codegen,
    .load_weights = nn_cnn_load_weights,
    .save_weights = nn_cnn_save_weights
};
