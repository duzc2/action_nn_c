/**
 * @file nn_type_gnn_infer.c
 * @brief Registry bridge that exposes the GNN inference backend to codegen.
 */

#include "nn_infer_registry.h"
#include "gnn_infer_ops.h"

#include <string.h>

/**
 * @brief Reconstruct a typed GNN inference context from codegen metadata.
 */
static void* nn_type_gnn_infer_create_codegen(const NNCodegenInferConfig* config) {
    GnnInferContext* context;

    if (config == 0 ||
        config->type_config == 0 ||
        config->type_config_size < sizeof(GnnConfig) ||
        config->type_config_type_name == 0 ||
        strcmp(config->type_config_type_name, "GnnConfig") != 0) {
        return 0;
    }

    context = nn_gnn_infer_create_with_config_blob(
        config->type_config,
        config->type_config_size,
        config->seed
    );
    if (context != 0) {
        context->expected_network_hash = config->network_hash;
        context->expected_layout_hash = config->layout_hash;
    }

    return context;
}

/**
 * @brief Adapt raw codegen buffers to the float-based GNN inference helper.
 */
static int nn_type_gnn_infer_auto_run_codegen(void* context, const void* input, void* output) {
    return nn_gnn_infer_auto_run(context, (const float*)input, (float*)output);
}

/**
 * @brief Builtin registry entry published when the GNN backend is enabled.
 */
const NNInferRegistryEntry nn_type_gnn_infer_entry = {
    .type_name = "gnn",
    .infer_step = nn_gnn_infer_step,
    .create = nn_type_gnn_infer_create_codegen,
    .destroy = nn_gnn_infer_destroy,
    .auto_run = nn_type_gnn_infer_auto_run_codegen,
    .graph_run = nn_type_gnn_infer_auto_run_codegen,
    .load_weights = nn_gnn_load_weights,
    .save_weights = nn_gnn_save_weights
};
