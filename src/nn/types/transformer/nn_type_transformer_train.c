#include "nn_train_registry.h"
#include "transformer_train_ops.h"

#include <stdlib.h>

typedef struct {
    size_t total_epochs;
    size_t total_steps;
    float average_loss;
} TransformerTrainContext;

static void* nn_type_transformer_train_create_codegen(void* infer_ctx, const NNCodegenTrainConfig* config) {
    TransformerTrainContext* context;
    (void)infer_ctx;
    (void)config;

    context = (TransformerTrainContext*)calloc(1U, sizeof(TransformerTrainContext));
    return context;
}

static void nn_type_transformer_train_destroy_codegen(void* context) {
    free(context);
}

static int nn_type_transformer_train_step_with_data_codegen(void* context, const void* input, const void* target) {
    TransformerTrainContext* train_ctx = (TransformerTrainContext*)context;
    (void)input;
    (void)target;

    if (train_ctx == 0) {
        return -1;
    }

    train_ctx->total_steps += 1U;
    train_ctx->average_loss = 0.0f;
    return nn_transformer_train_step(context);
}

static void nn_type_transformer_train_get_stats_codegen(void* context, size_t* out_epochs, size_t* out_steps, float* out_avg_loss) {
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
    nn_type_transformer_train_get_stats_codegen
};
