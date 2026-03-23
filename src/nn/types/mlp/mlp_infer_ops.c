/**
 * @file mlp_infer_ops.c
 * @brief MLP inference operations implementation
 */

#include "mlp_infer_ops.h"

#include <stdlib.h>
#include <string.h>

#define ABI_VERSION 1

/**
 * @brief Default MLP config for move network
 */
static const MlpConfig g_default_config = {
    .input_size = 3,
    .hidden_layer_count = 2,
    .hidden_sizes = { 16, 8 },
    .output_size = 2
};

/**
 * @brief FNV-1a hash for network structure
 */
static uint64_t compute_layout_hash(const MlpConfig* config) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    size_t i;

    if (config == NULL) {
        return hash;
    }

    hash ^= (uint64_t)config->input_size;
    hash *= 0x100000001b3ULL;

    hash ^= (uint64_t)config->output_size;
    hash *= 0x100000001b3ULL;

    hash ^= (uint64_t)config->hidden_layer_count;
    hash *= 0x100000001b3ULL;

    for (i = 0; i < config->hidden_layer_count && i < 4; i++) {
        hash ^= (uint64_t)config->hidden_sizes[i];
        hash *= 0x100000001b3ULL;
    }

    return hash;
}

MlpInferContext* nn_mlp_infer_create(void) {
    return nn_mlp_infer_create_with_config(&g_default_config, 42);
}

MlpInferContext* nn_mlp_infer_create_with_config(const MlpConfig* config, uint32_t seed) {
    MlpInferContext* ctx;
    size_t i;
    size_t prev_size;
    size_t max_buffer_size;

    if (config == NULL) {
        return NULL;
    }

    ctx = (MlpInferContext*)malloc(sizeof(MlpInferContext));
    if (ctx == NULL) {
        return NULL;
    }

    memset(ctx, 0, sizeof(MlpInferContext));

    ctx->config = *config;
    ctx->seed = seed;
    ctx->max_buffer_size = 0U;

    ctx->layer_count = config->hidden_layer_count + 1;
    ctx->layers = (MlpDenseLayer**)malloc(ctx->layer_count * sizeof(MlpDenseLayer*));

    if (ctx->layers == NULL) {
        free(ctx);
        return NULL;
    }

    prev_size = config->input_size;

    for (i = 0; i < config->hidden_layer_count; i++) {
        MlpActivationType act = (i < config->hidden_layer_count - 1) ?
            MLP_ACT_RELU : MLP_ACT_RELU;
        ctx->layers[i] = mlp_dense_create(prev_size, config->hidden_sizes[i], act, seed + (uint32_t)i);
        if (ctx->layers[i] == NULL) {
            for (size_t j = 0; j < i; j++) {
                mlp_dense_free(ctx->layers[j]);
            }
            free(ctx->layers);
            free(ctx);
            return NULL;
        }
        prev_size = config->hidden_sizes[i];
    }

    ctx->layers[ctx->layer_count - 1] = mlp_dense_create(
        prev_size,
        config->output_size,
        MLP_ACT_NONE,
        seed + (uint32_t)ctx->layer_count
    );

    if (ctx->layers[ctx->layer_count - 1] == NULL) {
        for (i = 0; i < ctx->layer_count - 1; i++) {
            mlp_dense_free(ctx->layers[i]);
        }
        free(ctx->layers);
        free(ctx);
        return NULL;
    }

    max_buffer_size = config->input_size;
    if (config->output_size > max_buffer_size) {
        max_buffer_size = config->output_size;
    }
    for (i = 0; i < config->hidden_layer_count && i < 4; i++) {
        if (config->hidden_sizes[i] > max_buffer_size) {
            max_buffer_size = config->hidden_sizes[i];
        }
    }

    ctx->input_buffer = (float*)malloc(config->input_size * sizeof(float));
    ctx->output_buffer = (float*)malloc(config->output_size * sizeof(float));
    ctx->work_buffer_a = (float*)malloc(max_buffer_size * sizeof(float));
    ctx->work_buffer_b = (float*)malloc(max_buffer_size * sizeof(float));
    ctx->max_buffer_size = max_buffer_size;

    if (ctx->input_buffer == NULL || ctx->output_buffer == NULL ||
        ctx->work_buffer_a == NULL || ctx->work_buffer_b == NULL) {
        free(ctx->input_buffer);
        free(ctx->output_buffer);
        free(ctx->work_buffer_a);
        free(ctx->work_buffer_b);
        for (i = 0; i < ctx->layer_count; i++) {
            mlp_dense_free(ctx->layers[i]);
        }
        free(ctx->layers);
        free(ctx);
        return NULL;
    }

    memset(ctx->input_buffer, 0, config->input_size * sizeof(float));
    memset(ctx->output_buffer, 0, config->output_size * sizeof(float));
    memset(ctx->work_buffer_a, 0, max_buffer_size * sizeof(float));
    memset(ctx->work_buffer_b, 0, max_buffer_size * sizeof(float));

    return ctx;
}

void nn_mlp_infer_destroy(void* context) {
    MlpInferContext* ctx = (MlpInferContext*)context;
    size_t i;

    if (ctx == NULL) {
        return;
    }

    free(ctx->input_buffer);
    free(ctx->output_buffer);
    free(ctx->work_buffer_a);
    free(ctx->work_buffer_b);

    for (i = 0; i < ctx->layer_count; i++) {
        if (ctx->layers[i] != NULL) {
            mlp_dense_free(ctx->layers[i]);
        }
    }

    free(ctx->layers);
    free(ctx);
}

void nn_mlp_infer_set_input(void* context, const float* input, size_t size) {
    MlpInferContext* ctx = (MlpInferContext*)context;
    size_t i;

    if (ctx == NULL || input == NULL) {
        return;
    }

    if (size != ctx->config.input_size) {
        return;
    }

    for (i = 0; i < size; i++) {
        ctx->input_buffer[i] = input[i];
    }
}

void nn_mlp_infer_get_output(void* context, float* output, size_t size) {
    MlpInferContext* ctx = (MlpInferContext*)context;
    size_t i;

    if (ctx == NULL || output == NULL) {
        return;
    }

    if (size != ctx->config.output_size) {
        return;
    }

    for (i = 0; i < size; i++) {
        output[i] = ctx->output_buffer[i];
    }
}

int nn_mlp_infer_step(void* context) {
    MlpInferContext* ctx = (MlpInferContext*)context;
    float* current;
    float* next;
    size_t i;

    if (ctx == NULL) {
        return -1;
    }

    current = ctx->input_buffer;
    next = ctx->work_buffer_a;

    for (i = 0; i < ctx->layer_count; i++) {
        MlpDenseLayer* layer = ctx->layers[i];

        if (i == ctx->layer_count - 1) {
            mlp_dense_forward(layer, ctx->output_buffer, current);
        } else {
            mlp_dense_forward(layer, next, current);
            current = next;
            next = (next == ctx->work_buffer_a) ? ctx->work_buffer_b : ctx->work_buffer_a;
        }
    }

    return 0;
}

int nn_mlp_infer_auto_run(void* context, const float* input, float* output) {
    MlpInferContext* ctx = (MlpInferContext*)context;

    if (ctx == NULL || input == NULL || output == NULL) {
        return -1;
    }

    nn_mlp_infer_set_input(ctx, input, ctx->config.input_size);

    if (nn_mlp_infer_step(ctx) != 0) {
        return -1;
    }

    nn_mlp_infer_get_output(ctx, output, ctx->config.output_size);

    return 0;
}

int nn_mlp_load_weights(void* context, FILE* fp) {
    MlpInferContext* ctx = (MlpInferContext*)context;
    uint64_t file_hash;
    uint64_t file_layout_hash;
    uint32_t file_abi_version;
    size_t i;
    size_t j;
    int rc;

    if (ctx == NULL || fp == NULL) {
        return 0;
    }

    rc = (int)fread(&file_hash, sizeof(file_hash), 1, fp);
    if (rc != 1) return 0;

    rc = (int)fread(&file_layout_hash, sizeof(file_layout_hash), 1, fp);
    if (rc != 1) return 0;

    rc = (int)fread(&file_abi_version, sizeof(file_abi_version), 1, fp);
    if (rc != 1) return 0;

    if (file_abi_version != ABI_VERSION) {
        return 0;
    }

    for (i = 0; i < ctx->layer_count; i++) {
        MlpDenseLayer* layer = ctx->layers[i];

        rc = (int)fread(layer->weights, sizeof(float),
                        layer->input_size * layer->output_size, fp);
        if ((size_t)rc != layer->input_size * layer->output_size) {
            return 0;
        }

        rc = (int)fread(layer->bias, sizeof(float), layer->output_size, fp);
        if ((size_t)rc != layer->output_size) {
            return 0;
        }
    }

    return 1;
}

int nn_mlp_save_weights(void* context, FILE* fp) {
    MlpInferContext* ctx = (MlpInferContext*)context;
    uint64_t hash;
    uint64_t layout_hash;
    uint32_t abi_ver = ABI_VERSION;
    size_t i;
    int rc;

    if (ctx == NULL || fp == NULL) {
        return 0;
    }

    hash = nn_mlp_get_network_hash(ctx);
    layout_hash = compute_layout_hash(&ctx->config);

    rc = (int)fwrite(&hash, sizeof(hash), 1, fp);
    if (rc != 1) return 0;

    rc = (int)fwrite(&layout_hash, sizeof(layout_hash), 1, fp);
    if (rc != 1) return 0;

    rc = (int)fwrite(&abi_ver, sizeof(abi_ver), 1, fp);
    if (rc != 1) return 0;

    for (i = 0; i < ctx->layer_count; i++) {
        MlpDenseLayer* layer = ctx->layers[i];

        rc = (int)fwrite(layer->weights, sizeof(float),
                         layer->input_size * layer->output_size, fp);
        if ((size_t)rc != layer->input_size * layer->output_size) {
            return 0;
        }

        rc = (int)fwrite(layer->bias, sizeof(float), layer->output_size, fp);
        if ((size_t)rc != layer->output_size) {
            return 0;
        }
    }

    return 1;
}

uint64_t nn_mlp_get_network_hash(const void* context) {
    const MlpInferContext* ctx = (const MlpInferContext*)context;

    if (ctx == NULL) {
        return 0;
    }

    return compute_layout_hash(&ctx->config);
}
