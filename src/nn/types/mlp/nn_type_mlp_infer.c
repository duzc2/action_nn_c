#include "nn_infer_registry.h"
#include "mlp_infer_ops.h"

#include <string.h>

static void* nn_type_mlp_infer_create_codegen(const NNCodegenInferConfig* config) {
    MlpConfig mlp_config;
    MlpInferContext* context;

    if (config == 0 ||
        config->type_config == 0 ||
        config->type_config_size != sizeof(MlpConfig) ||
        config->type_config_type_name == 0 ||
        strcmp(config->type_config_type_name, "MlpConfig") != 0) {
        return 0;
    }

    mlp_config = *(const MlpConfig*)config->type_config;
    context = nn_mlp_infer_create_with_config(&mlp_config, config->seed);
    if (context != 0) {
        context->expected_network_hash = config->network_hash;
        context->expected_layout_hash = config->layout_hash;
    }

    return context;
}

static int nn_type_mlp_infer_auto_run_codegen(void* context, const void* input, void* output) {
    return nn_mlp_infer_auto_run(context, (const float*)input, (float*)output);
}

const NNInferRegistryEntry nn_type_mlp_infer_entry = {
    .type_name = "mlp",
    .infer_step = nn_mlp_infer_step,
    .create = nn_type_mlp_infer_create_codegen,
    .destroy = nn_mlp_infer_destroy,
    .auto_run = nn_type_mlp_infer_auto_run_codegen,
    .load_weights = nn_mlp_load_weights,
    .save_weights = nn_mlp_save_weights
};
