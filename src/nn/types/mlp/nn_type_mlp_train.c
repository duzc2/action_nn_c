#include "nn_train_registry.h"
#include "mlp_train_ops.h"

#include <string.h>

static void* nn_type_mlp_train_create_codegen(void* infer_ctx, const NNCodegenTrainConfig* config) {
    MlpTrainConfig train_config;

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

const NNTrainRegistryEntry nn_type_mlp_train_entry = {
    "mlp",
    nn_mlp_train_step,
    nn_type_mlp_train_create_codegen,
    (void (*)(void*))nn_mlp_train_destroy,
    (NNTrainStepWithDataFn)nn_mlp_train_step_with_data,
    (NNTrainGetStatsFn)nn_mlp_train_get_stats
};
