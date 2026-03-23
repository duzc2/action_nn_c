#include "nn_infer_registry.h"
#include "mlp_infer_ops.h"

static void* nn_type_mlp_infer_create_codegen(const NNCodegenInferConfig* config) {
    MlpConfig mlp_config;
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

    return nn_mlp_infer_create_with_config(&mlp_config, config->seed);
}

static int nn_type_mlp_infer_auto_run_codegen(void* context, const void* input, void* output) {
    return nn_mlp_infer_auto_run(context, (const float*)input, (float*)output);
}

const NNInferRegistryEntry nn_type_mlp_infer_entry = {
    "mlp",
    nn_mlp_infer_step,
    nn_type_mlp_infer_create_codegen,
    nn_mlp_infer_destroy,
    nn_type_mlp_infer_auto_run_codegen
};
