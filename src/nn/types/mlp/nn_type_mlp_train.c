#include "nn_train_registry.h"
#include "mlp_train_ops.h"

static void* nn_type_mlp_train_create_codegen(void* infer_ctx, const NNCodegenTrainConfig* config) {
    MlpTrainConfig train_config;

    if (config == 0) {
        return 0;
    }

    train_config.learning_rate = config->learning_rate;
    train_config.momentum = config->momentum;
    train_config.weight_decay = config->weight_decay;
    train_config.optimizer = MLP_OPT_SGD;
    train_config.loss_func = MLP_LOSS_MSE;
    train_config.batch_size = config->batch_size;
    train_config.seed = config->seed;

    return nn_mlp_train_create(infer_ctx, &train_config);
}

static void nn_type_mlp_train_destroy_codegen(void* context) {
    nn_mlp_train_destroy((MlpTrainContext*)context);
}

static int nn_type_mlp_train_step_with_data_codegen(void* context, const void* input, const void* target) {
    return nn_mlp_train_step_with_data((MlpTrainContext*)context, (const float*)input, (const float*)target);
}

static void nn_type_mlp_train_get_stats_codegen(void* context, size_t* out_epochs, size_t* out_steps, float* out_avg_loss) {
    nn_mlp_train_get_stats((MlpTrainContext*)context, out_epochs, out_steps, out_avg_loss);
}

const NNTrainRegistryEntry nn_type_mlp_train_entry = {
    "mlp",
    nn_mlp_train_step,
    nn_type_mlp_train_create_codegen,
    nn_type_mlp_train_destroy_codegen,
    nn_type_mlp_train_step_with_data_codegen,
    nn_type_mlp_train_get_stats_codegen
};
