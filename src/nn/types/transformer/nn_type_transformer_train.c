#include "nn_train_registry.h"
#include "transformer_train_ops.h"

#include <stdlib.h>
#include <string.h>

static void* nn_type_transformer_train_create_codegen(
    void* infer_ctx,
    const NNCodegenTrainConfig* config
) {
    TransformerTrainContext* context;
    TransformerTrainConfig typed_config;

    if (infer_ctx == 0 ||
        config == 0 ||
        config->type_config == 0 ||
        config->type_config_size != sizeof(TransformerTrainConfig) ||
        config->type_config_type_name == 0 ||
        strcmp(config->type_config_type_name, "TransformerTrainConfig") != 0) {
        return 0;
    }

    context = (TransformerTrainContext*)calloc(1U, sizeof(TransformerTrainContext));
    if (context == 0) {
        return 0;
    }

    typed_config = *(const TransformerTrainConfig*)config->type_config;
    context->infer_ctx = (TransformerInferContext*)infer_ctx;
    context->learning_rate = typed_config.learning_rate;

    return context;
}

static void nn_type_transformer_train_destroy_codegen(void* context) {
    free(context);
}

static int nn_type_transformer_train_step_with_data_codegen(
    void* context,
    const void* input,
    const void* target
) {
    TransformerTrainContext* train_ctx = (TransformerTrainContext*)context;

    if (train_ctx == 0 || input == 0 || target == 0) {
        return -1;
    }

    train_ctx->current_question = (const char*)input;
    train_ctx->current_answer = (const char*)target;
    return nn_transformer_train_step(train_ctx);
}

static int nn_type_transformer_train_step_with_output_gradient_codegen(
    void* context,
    const void* input,
    const void* output_gradient,
    void* input_gradient
) {
    return nn_transformer_train_step_with_output_gradient(
        (TransformerTrainContext*)context,
        (const float*)input,
        (const float*)output_gradient,
        (float*)input_gradient
    );
}

static void nn_type_transformer_train_get_stats_codegen(
    void* context,
    size_t* out_epochs,
    size_t* out_steps,
    float* out_avg_loss
) {
    TransformerTrainContext* train_ctx = (TransformerTrainContext*)context;

    if (train_ctx == 0) {
        return;
    }
    if (out_epochs != 0) {
        *out_epochs = train_ctx->total_epochs;
    }
    if (out_steps != 0) {
        *out_steps = train_ctx->total_steps;
    }
    if (out_avg_loss != 0) {
        *out_avg_loss = train_ctx->average_loss;
    }
}

const NNTrainRegistryEntry nn_type_transformer_train_entry = {
    "transformer",
    nn_transformer_train_step,
    nn_type_transformer_train_create_codegen,
    nn_type_transformer_train_destroy_codegen,
    nn_type_transformer_train_step_with_data_codegen,
    nn_type_transformer_train_step_with_output_gradient_codegen,
    nn_type_transformer_train_get_stats_codegen
};
