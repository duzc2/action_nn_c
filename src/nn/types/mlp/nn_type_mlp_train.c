/**
 * @file nn_type_mlp_train.c
 * @brief Registry bridge that exposes the MLP training backend to codegen.
 */

#include "nn_train_registry.h"
#include "mlp_train_ops.h"

#include <string.h>

/**
 * @brief Reconstruct a typed MLP training context from codegen metadata.
 */
static void* nn_type_mlp_train_create_codegen(void* infer_ctx, const NNCodegenTrainConfig* config) {
    MlpTrainConfig train_config;

    /* Training must wrap an existing MLP infer context plus a typed train config. */
    if (infer_ctx == 0 ||
        config == 0 ||
        config->type_config == 0 ||
        config->type_config_size != sizeof(MlpTrainConfig) ||
        config->type_config_type_name == 0 ||
        strcmp(config->type_config_type_name, "MlpTrainConfig") != 0) {
        return 0;
    }

    train_config = *(const MlpTrainConfig*)config->type_config;
    return nn_mlp_train_create(infer_ctx, &train_config);
}

/**
 * @brief Adapt raw graph buffers to the typed MLP backprop entry point.
 */
static int nn_type_mlp_train_step_with_output_gradient_codegen(
    void* context,
    const void* input,
    const void* output_gradient,
    void* input_gradient
) {
    return nn_mlp_train_step_with_output_gradient(
        (MlpTrainContext*)context,
        (const float*)input,
        (const float*)output_gradient,
        (float*)input_gradient
    );
}

/**
 * @brief Builtin registry entry published when the MLP training backend is enabled.
 */
const NNTrainRegistryEntry nn_type_mlp_train_entry = {
    "mlp",
    nn_mlp_train_step,
    nn_type_mlp_train_create_codegen,
    (void (*)(void*))nn_mlp_train_destroy,
    (NNTrainStepWithDataFn)nn_mlp_train_step_with_data,
    nn_type_mlp_train_step_with_output_gradient_codegen,
    (NNTrainGetStatsFn)nn_mlp_train_get_stats
};
