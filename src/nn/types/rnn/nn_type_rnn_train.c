/**
 * @file nn_type_rnn_train.c
 * @brief Registry bridge that exposes the RNN training backend to codegen.
 */

#include "nn_train_registry.h"
#include "rnn_train_ops.h"

#include <string.h>

/**
 * @brief Reconstruct a typed RNN training context from generated metadata.
 */
static void* nn_type_rnn_train_create_codegen(void* infer_ctx, const NNCodegenTrainConfig* config) {
    RnnTrainConfig typed_config;

    if (infer_ctx == 0 ||
        config == 0 ||
        config->type_config == 0 ||
        config->type_config_size != sizeof(RnnTrainConfig) ||
        config->type_config_type_name == 0 ||
        strcmp(config->type_config_type_name, "RnnTrainConfig") != 0) {
        return 0;
    }

    typed_config = *(const RnnTrainConfig*)config->type_config;
    return nn_rnn_train_create(infer_ctx, &typed_config);
}

/**
 * @brief Adapt raw graph buffers to the typed RNN backprop entry point.
 */
static int nn_type_rnn_train_step_with_output_gradient_codegen(
    void* context,
    const void* input,
    const void* output_gradient,
    void* input_gradient
) {
    return nn_rnn_train_step_with_output_gradient(
        (RnnTrainContext*)context,
        (const float*)input,
        (const float*)output_gradient,
        (float*)input_gradient
    );
}

/**
 * @brief Builtin registry entry published when the RNN training backend is enabled.
 */
const NNTrainRegistryEntry nn_type_rnn_train_entry = {
    "rnn",
    nn_rnn_train_step,
    nn_type_rnn_train_create_codegen,
    (void (*)(void*))nn_rnn_train_destroy,
    (NNTrainStepWithDataFn)nn_rnn_train_step_with_data,
    nn_type_rnn_train_step_with_output_gradient_codegen,
    (NNTrainGetStatsFn)nn_rnn_train_get_stats
};
