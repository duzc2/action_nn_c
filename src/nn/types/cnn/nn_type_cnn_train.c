/**
 * @file nn_type_cnn_train.c
 * @brief Registry bridge that exposes the CNN training backend to codegen.
 */

#include "nn_train_registry.h"
#include "cnn_train_ops.h"

#include <string.h>

/**
 * @brief Reconstruct a typed CNN training context from generated metadata.
 */
static void* nn_type_cnn_train_create_codegen(void* infer_ctx, const NNCodegenTrainConfig* config) {
    CnnTrainConfig typed_config;

    if (infer_ctx == 0 ||
        config == 0 ||
        config->type_config == 0 ||
        config->type_config_size != sizeof(CnnTrainConfig) ||
        config->type_config_type_name == 0 ||
        strcmp(config->type_config_type_name, "CnnTrainConfig") != 0) {
        return 0;
    }

    typed_config = *(const CnnTrainConfig*)config->type_config;
    return nn_cnn_train_create(infer_ctx, &typed_config);
}

/**
 * @brief Adapt raw graph buffers to the typed CNN backprop entry point.
 */
static int nn_type_cnn_train_step_with_output_gradient_codegen(
    void* context,
    const void* input,
    const void* output_gradient,
    void* input_gradient
) {
    return nn_cnn_train_step_with_output_gradient(
        (CnnTrainContext*)context,
        (const float*)input,
        (const float*)output_gradient,
        (float*)input_gradient
    );
}

/**
 * @brief Builtin registry entry published when the CNN training backend is enabled.
 */
const NNTrainRegistryEntry nn_type_cnn_train_entry = {
    "cnn",
    nn_cnn_train_step,
    nn_type_cnn_train_create_codegen,
    (void (*)(void*))nn_cnn_train_destroy,
    (NNTrainStepWithDataFn)nn_cnn_train_step_with_data,
    nn_type_cnn_train_step_with_output_gradient_codegen,
    (NNTrainGetStatsFn)nn_cnn_train_get_stats
};
