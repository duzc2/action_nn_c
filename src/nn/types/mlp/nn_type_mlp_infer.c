#include "nn_infer_registry.h"
#include "mlp_infer_ops.h"

static void* nn_type_mlp_infer_create_codegen(const NNCodegenInferConfig* config) {
    MlpConfig mlp_config;
    MlpInferContext* context;
    size_t i;

    if (config == 0) {
        return 0;
    }

    mlp_config.input_size = config->input_size;
    mlp_config.hidden_layer_count = config->hidden_layer_count;
    if (mlp_config.hidden_layer_count > (sizeof(mlp_config.hidden_sizes) / sizeof(mlp_config.hidden_sizes[0]))) {
        mlp_config.hidden_layer_count = sizeof(mlp_config.hidden_sizes) / sizeof(mlp_config.hidden_sizes[0]);
    }
    for (i = 0; i < mlp_config.hidden_layer_count; ++i) {
        mlp_config.hidden_sizes[i] = config->hidden_sizes[i];
    }
    for (; i < (sizeof(mlp_config.hidden_sizes) / sizeof(mlp_config.hidden_sizes[0])); ++i) {
        mlp_config.hidden_sizes[i] = 0U;
    }
    mlp_config.output_size = config->output_size;

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
