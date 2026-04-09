/**
 * @file nn_type_cnn_dual_pool_train.c
 * @brief Registry bridge that exposes the dual-pool CNN training backend to codegen.
 */

#include "nn_train_registry.h"
#include "cnn_dual_pool_train_ops.h"

#include <string.h>

static void* nn_type_cnn_dual_pool_train_create_codegen(void* infer_ctx, const NNCodegenTrainConfig* config) {
    CnnDualPoolTrainConfig typed_config;

    if (infer_ctx == 0 || config == 0 || config->type_config == 0 ||
        config->type_config_size != sizeof(CnnDualPoolTrainConfig) ||
        config->type_config_type_name == 0 ||
        strcmp(config->type_config_type_name, "CnnDualPoolTrainConfig") != 0) {
        return 0;
    }

    typed_config = *(const CnnDualPoolTrainConfig*)config->type_config;
    return nn_cnn_dual_pool_train_create(infer_ctx, &typed_config);
}

static int nn_type_cnn_dual_pool_train_step_with_output_gradient_codegen(void* context, const void* input, const void* output_gradient, void* input_gradient) {
    return nn_cnn_dual_pool_train_step_with_output_gradient((CnnDualPoolTrainContext*)context, (const float*)input, (const float*)output_gradient, (float*)input_gradient);
}

const NNTrainRegistryEntry nn_type_cnn_dual_pool_train_entry = {
    "cnn_dual_pool",
    nn_cnn_dual_pool_train_step,
    nn_type_cnn_dual_pool_train_create_codegen,
    (void (*)(void*))nn_cnn_dual_pool_train_destroy,
    (NNTrainStepWithDataFn)nn_cnn_dual_pool_train_step_with_data,
    nn_type_cnn_dual_pool_train_step_with_output_gradient_codegen,
    (NNTrainGetStatsFn)nn_cnn_dual_pool_train_get_stats
};
