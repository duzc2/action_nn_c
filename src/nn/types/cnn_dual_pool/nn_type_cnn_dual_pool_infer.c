/**
 * @file nn_type_cnn_dual_pool_infer.c
 * @brief Registry bridge that exposes the dual-pool CNN inference backend to codegen.
 */

#include "nn_infer_registry.h"
#include "cnn_dual_pool_infer_ops.h"

#include <string.h>

static void* nn_type_cnn_dual_pool_infer_create_codegen(const NNCodegenInferConfig* config) {
    CnnDualPoolConfig typed_config;
    CnnDualPoolInferContext* context;
    uint32_t seed;

    if (config == 0 || config->type_config == 0 ||
        config->type_config_size != sizeof(CnnDualPoolConfig) ||
        config->type_config_type_name == 0 ||
        strcmp(config->type_config_type_name, "CnnDualPoolConfig") != 0) {
        return 0;
    }

    typed_config = *(const CnnDualPoolConfig*)config->type_config;
    seed = config->seed != 0U ? config->seed : typed_config.seed;
    context = nn_cnn_dual_pool_infer_create_with_config(&typed_config, seed);
    if (context != 0) {
        context->expected_network_hash = config->network_hash;
        context->expected_layout_hash = config->layout_hash;
    }
    return context;
}

static int nn_type_cnn_dual_pool_infer_auto_run_codegen(void* context, const void* input, void* output) {
    return nn_cnn_dual_pool_infer_auto_run(context, (const float*)input, (float*)output);
}

const NNInferRegistryEntry nn_type_cnn_dual_pool_infer_entry = {
    .type_name = "cnn_dual_pool",
    .infer_step = nn_cnn_dual_pool_infer_step,
    .create = nn_type_cnn_dual_pool_infer_create_codegen,
    .destroy = nn_cnn_dual_pool_infer_destroy,
    .auto_run = nn_type_cnn_dual_pool_infer_auto_run_codegen,
    .graph_run = nn_type_cnn_dual_pool_infer_auto_run_codegen,
    .load_weights = nn_cnn_dual_pool_load_weights,
    .save_weights = nn_cnn_dual_pool_save_weights
};
