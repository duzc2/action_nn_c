/**
 * @file nn_type_gnn_train.c
 * @brief Registry bridge that exposes the GNN training backend to codegen.
 */

#include "nn_train_registry.h"
#include "gnn_train_ops.h"

#include <string.h>

/**
 * @brief Reconstruct a typed GNN training context from codegen metadata.
 */
static void* nn_type_gnn_train_create_codegen(void* infer_ctx, const NNCodegenTrainConfig* config) {
    GnnTrainConfig train_config;

    if (infer_ctx == 0 ||
        config == 0 ||
        config->type_config == 0 ||
        config->type_config_size != sizeof(GnnTrainConfig) ||
        config->type_config_type_name == 0 ||
        strcmp(config->type_config_type_name, "GnnTrainConfig") != 0) {
        return 0;
    }

    train_config = *(const GnnTrainConfig*)config->type_config;
    return nn_gnn_train_create(infer_ctx, &train_config);
}

/**
 * @brief Adapt raw graph buffers to the typed GNN backprop entry point.
 */
static int nn_type_gnn_train_step_with_output_gradient_codegen(
    void* context,
    const void* input,
    const void* output_gradient,
    void* input_gradient
) {
    return nn_gnn_train_step_with_output_gradient(
        (GnnTrainContext*)context,
        (const float*)input,
        (const float*)output_gradient,
        (float*)input_gradient
    );
}

/**
 * @brief Builtin registry entry published when the GNN training backend is enabled.
 */
const NNTrainRegistryEntry nn_type_gnn_train_entry = {
    "gnn",
    nn_gnn_train_step,
    nn_type_gnn_train_create_codegen,
    (void (*)(void*))nn_gnn_train_destroy,
    (NNTrainStepWithDataFn)nn_gnn_train_step_with_data,
    nn_type_gnn_train_step_with_output_gradient_codegen,
    (NNTrainGetStatsFn)nn_gnn_train_get_stats
};
