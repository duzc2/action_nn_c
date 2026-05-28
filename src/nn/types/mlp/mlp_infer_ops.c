/**
 * @file mlp_infer_ops.c
 * @brief MLP inference operations implementation
 */

#include "mlp_infer_ops.h"
#include "../../norm/rms_norm.h"
#include "../../dropout/dropout.h"
#include "../../residual/skip_connection.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../../utils/error.h"

#define ABI_VERSION 1

/**
 * @section mlp_infer_design MLP inference execution notes
 *
 * The handwritten MLP inference backend is intentionally small, but it still
 * demonstrates the lifecycle expected by generated graph code: typed creation,
 * deterministic layer allocation, buffer reuse during forward execution, and
 * hash-guarded save/load support. The comments in this file therefore focus on
 * the ownership and scheduling decisions that make the backend reusable.
 */
/**
 * @brief Compute a compact structural hash from the active MLP config.
 *
 * The infer backend uses this lightweight hash to reject weight files whose
 * parameter layout no longer matches the current network shape.
 */
static uint64_t compute_layout_hash(const MlpConfig* config) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    const size_t* hidden_sizes;
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

    hidden_sizes = mlp_config_hidden_sizes_view(config);
    for (i = 0; i < config->hidden_layer_count; i++) {
        hash ^= (uint64_t)hidden_sizes[i];
        hash *= 0x100000001b3ULL;
    }

    hash ^= (uint64_t)config->hidden_activation;
    hash *= 0x100000001b3ULL;

    hash ^= (uint64_t)config->output_activation;
    hash *= 0x100000001b3ULL;

    /* P0: include optional-module flags so weight files are invalidated
     * when the user changes norm/dropout/skip settings. */
    hash ^= (uint64_t)config->use_rms_norm;
    hash *= 0x100000001b3ULL;
    hash ^= (uint64_t)config->use_dropout;
    hash *= 0x100000001b3ULL;
    hash ^= (uint64_t)config->use_skip;
    hash *= 0x100000001b3ULL;
    hash ^= (uint64_t)config->skip_mode;
    hash *= 0x100000001b3ULL;

    return hash;
}

/**
 * @brief Deliberately reject implicit construction without typed config data.
 */
MlpInferContext* nn_mlp_infer_create(void) {
    return NULL;
}

/**
 * @brief Allocate and initialize a typed MLP inference context.
 *
 * The context owns layer objects, reusable work buffers, and the expected hash
 * values used during weight-file compatibility checks. Creation is front-loaded
 * on purpose: once the context exists, inference steps should run without any
 * further heap allocation or topology-dependent branching.
 */
static int mlp_config_get_required_size(size_t hidden_layer_count, size_t* out_size) {
    size_t total_size;

    if (out_size == NULL) {
        return 0;
    }
    if (hidden_layer_count > ((size_t)-1 - sizeof(MlpConfig)) / sizeof(size_t)) {
        return 0;
    }

    total_size = sizeof(MlpConfig) + (hidden_layer_count * sizeof(size_t));
    *out_size = total_size;
    return 1;
}

/**
 * @brief Validate a serialized MLP config blob before rebuilding typed state.
 */
static int mlp_config_blob_is_valid(const void* config_data, size_t config_size) {
    const MlpConfig* config = (const MlpConfig*)config_data;
    const size_t* hidden_sizes;
    size_t required_size;
    size_t hidden_index;

    if (config_data == NULL) {
        return 0;
    }
    if (!mlp_config_get_required_size(config->hidden_layer_count, &required_size)) {
        return 0;
    }
    if (config_size != required_size) {
        return 0;
    }
    if (config->input_size == 0U || config->output_size == 0U) {
        return 0;
    }

    hidden_sizes = mlp_config_hidden_sizes_view(config);
    for (hidden_index = 0U; hidden_index < config->hidden_layer_count; ++hidden_index) {
        if (hidden_sizes[hidden_index] == 0U) {
            return 0;
        }
    }

    return 1;
}

/**
 * @brief Allocate and initialize a typed MLP inference context from raw config bytes.
 */
MlpInferContext* nn_mlp_infer_create_with_config_blob(
    const void* config_data,
    size_t config_size,
    uint32_t seed
) {
    MlpInferContext* ctx;
    const size_t* hidden_sizes;
    size_t i;
    size_t prev_size;
    size_t max_buffer_size;

    if (!mlp_config_blob_is_valid(config_data, config_size)) {
        return NULL;
    }

    ctx = (MlpInferContext*)malloc(sizeof(MlpInferContext));
    if (ctx == NULL) {
        return NULL;
    }

    memset(ctx, 0, sizeof(MlpInferContext));

    ctx->config = (MlpConfig*)malloc(config_size);
    if (ctx->config == NULL) {
        free(ctx);
        return NULL;
    }
    (void)memcpy(ctx->config, config_data, config_size);
    ctx->config_size = config_size;
    ctx->seed = seed;
    ctx->max_buffer_size = 0U;
    hidden_sizes = mlp_config_hidden_sizes_view(ctx->config);

    ctx->layer_count = ctx->config->hidden_layer_count + 1U;
    ctx->layers = (MlpDenseLayer**)malloc(ctx->layer_count * sizeof(MlpDenseLayer*));

    if (ctx->layers == NULL) {
        free(ctx->config);
        free(ctx);
        return NULL;
    }

    prev_size = ctx->config->input_size;

    /* Materialize hidden layers in declared order so weight layout stays stable. */
    for (i = 0; i < ctx->config->hidden_layer_count; i++) {
        ctx->layers[i] = mlp_dense_create(
            prev_size,
            hidden_sizes[i],
            ctx->config->hidden_activation,
            seed + (uint32_t)i
        );
        if (ctx->layers[i] == NULL) {
            size_t j;
            for (j = 0U; j < i; j++) {
                mlp_dense_free(ctx->layers[j]);
            }
            free(ctx->layers);
            free(ctx->config);
            free(ctx);
            return NULL;
        }
        prev_size = hidden_sizes[i];
    }

    /* The output layer is created last because it depends on the final hidden width. */
    ctx->layers[ctx->layer_count - 1] = mlp_dense_create(
        prev_size,
        ctx->config->output_size,
        ctx->config->output_activation,
        seed + (uint32_t)ctx->layer_count
    );

    if (ctx->layers[ctx->layer_count - 1] == NULL) {
        for (i = 0; i < ctx->layer_count - 1; i++) {
            mlp_dense_free(ctx->layers[i]);
        }
        free(ctx->layers);
        free(ctx->config);
        free(ctx);
        return NULL;
    }

    /* Reserve work buffers at the maximum layer width so hidden passes can reuse them. */
    max_buffer_size = ctx->config->input_size;
    if (ctx->config->output_size > max_buffer_size) {
        max_buffer_size = ctx->config->output_size;
    }
    for (i = 0; i < ctx->config->hidden_layer_count; i++) {
        if (hidden_sizes[i] > max_buffer_size) {
            max_buffer_size = hidden_sizes[i];
        }
    }

    /* Separate input/output storage avoids aliasing when auto_run uses caller buffers. */
    ctx->input_buffer = (float*)malloc(ctx->config->input_size * sizeof(float));
    ctx->output_buffer = (float*)malloc(ctx->config->output_size * sizeof(float));
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
        free(ctx->config);
        free(ctx);
        return NULL;
    }

    /* Zero-initialize every buffer so partially run contexts never expose garbage. */
    memset(ctx->input_buffer, 0, ctx->config->input_size * sizeof(float));
    memset(ctx->output_buffer, 0, ctx->config->output_size * sizeof(float));
    memset(ctx->work_buffer_a, 0, max_buffer_size * sizeof(float));
    memset(ctx->work_buffer_b, 0, max_buffer_size * sizeof(float));

    /* ─── P0: allocate shared-module buffers ─── */
    ctx->skips = NULL;
    ctx->dropouts = NULL;
    ctx->norm_gamma = NULL;
    ctx->norm_dgamma = NULL;

    if (ctx->config->use_rms_norm) {
        size_t total_norm = 0;
        size_t k;
        for (k = 0; k < ctx->layer_count; ++k) {
            total_norm += ctx->layers[k]->output_size;
        }
        ctx->norm_gamma  = (float*)malloc(total_norm * sizeof(float));
        ctx->norm_dgamma = (float*)malloc(total_norm * sizeof(float));
        if (ctx->norm_gamma == NULL || ctx->norm_dgamma == NULL) {
            free(ctx->norm_gamma);
            free(ctx->norm_dgamma);
            /* clean up previously allocated resources */
            for (i = 0; i < ctx->layer_count; i++) { mlp_dense_free(ctx->layers[i]); }
            free(ctx->layers); free(ctx->config);
            free(ctx->input_buffer); free(ctx->output_buffer);
            free(ctx->work_buffer_a); free(ctx->work_buffer_b);
            free(ctx);
            return NULL;
        }
        /* Initialise gamma to all-ones so RMSNorm starts as identity. */
        for (k = 0; k < total_norm; ++k) {
            ctx->norm_gamma[k] = 1.0f;
            ctx->norm_dgamma[k] = 0.0f;
        }
    }

    if (ctx->config->use_dropout) {
        size_t hidden_count = ctx->layer_count - 1U;
        ctx->dropouts = (DropoutLayer**)malloc(hidden_count * sizeof(DropoutLayer*));
        if (ctx->dropouts == NULL) {
            free(ctx->norm_gamma); free(ctx->norm_dgamma);
            for (i = 0; i < ctx->layer_count; i++) { mlp_dense_free(ctx->layers[i]); }
            free(ctx->layers); free(ctx->config);
            free(ctx->input_buffer); free(ctx->output_buffer);
            free(ctx->work_buffer_a); free(ctx->work_buffer_b);
            free(ctx);
            return NULL;
        }
        for (i = 0; i < hidden_count; ++i) {
            ctx->dropouts[i] = (DropoutLayer*)malloc(sizeof(DropoutLayer));
            if (ctx->dropouts[i] == NULL) {
                size_t j;
                for (j = 0; j < i; ++j) { dropout_free(ctx->dropouts[j]); free(ctx->dropouts[j]); }
                free(ctx->dropouts); free(ctx->norm_gamma); free(ctx->norm_dgamma);
                for (j = 0; j < ctx->layer_count; j++) { mlp_dense_free(ctx->layers[j]); }
                free(ctx->layers); free(ctx->config);
                free(ctx->input_buffer); free(ctx->output_buffer);
                free(ctx->work_buffer_a); free(ctx->work_buffer_b);
                free(ctx);
                return NULL;
            }
            dropout_init(ctx->dropouts[i], ctx->config->dropout_rate,
                         seed + (uint32_t)i + 1000U);
        }
    }

    if (ctx->config->use_skip) {
        ctx->skips = (SkipConnection*)malloc(ctx->layer_count * sizeof(SkipConnection));
        if (ctx->skips == NULL) {
            if (ctx->dropouts) {
                for (i = 0; i < ctx->layer_count - 1U; ++i) { dropout_free(ctx->dropouts[i]); free(ctx->dropouts[i]); }
                free(ctx->dropouts);
            }
            free(ctx->norm_gamma); free(ctx->norm_dgamma);
            for (i = 0; i < ctx->layer_count; i++) { mlp_dense_free(ctx->layers[i]); }
            free(ctx->layers); free(ctx->config);
            free(ctx->input_buffer); free(ctx->output_buffer);
            free(ctx->work_buffer_a); free(ctx->work_buffer_b);
            free(ctx);
            return NULL;
        }
        for (i = 0; i < ctx->layer_count; ++i) {
            skip_init(&ctx->skips[i], ctx->config->skip_mode);
        }
    }

    return ctx;
}

/**
 * @brief Compatibility wrapper for direct typed calls that already own a full blob.
 */
MlpInferContext* nn_mlp_infer_create_with_config(const MlpConfig* config, uint32_t seed) {
    if (config == NULL) {
        return NULL;
    }
    return nn_mlp_infer_create_with_config_blob(
        config,
        mlp_config_size_for_hidden_layers(config->hidden_layer_count),
        seed
    );
}

/**
 * @brief Release an inference context together with all owned layers and buffers.
 */
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
    free(ctx->config);

    for (i = 0; i < ctx->layer_count; i++) {
        if (ctx->layers[i] != NULL) {
            mlp_dense_free(ctx->layers[i]);
        }
    }

    /* P0: free shared-module allocations */
    free(ctx->norm_gamma);
    free(ctx->norm_dgamma);
    if (ctx->dropouts != NULL) {
        for (i = 0; i < ctx->layer_count - 1U; ++i) {
            dropout_free(ctx->dropouts[i]);
            free(ctx->dropouts[i]);
        }
        free(ctx->dropouts);
    }
    free(ctx->skips);

    free(ctx->layers);
    free(ctx);
}

/**
 * @brief Copy caller input values into the context-owned input buffer.
 */
void nn_mlp_infer_set_input(void* context, const float* input, size_t size) {
    MlpInferContext* ctx = (MlpInferContext*)context;
    size_t i;

    if (ctx == NULL || input == NULL) {
        return;
    }

    if (size != ctx->config->input_size) {
        return;
    }

    for (i = 0; i < size; i++) {
        ctx->input_buffer[i] = input[i];
    }
}

/**
 * @brief Copy the latest inference output into caller-owned storage.
 */
void nn_mlp_infer_get_output(void* context, float* output, size_t size) {
    MlpInferContext* ctx = (MlpInferContext*)context;
    size_t i;

    if (ctx == NULL || output == NULL) {
        return;
    }

    if (size != ctx->config->output_size) {
        return;
    }

    for (i = 0; i < size; i++) {
        output[i] = ctx->output_buffer[i];
    }
}

/**
 * @brief Run a full forward pass through every dense layer.
 *
 * Two reusable work buffers are alternated for hidden activations so the infer
 * path avoids per-step heap traffic while still supporting arbitrary depth.
 */
int nn_mlp_infer_step(void* context) {
    MlpInferContext* ctx = (MlpInferContext*)context;
    float* current;
    float* next;
    size_t i;
    size_t norm_offset;
    float epsilon;

    if (ctx == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    /* Ensure dropout is in inference (non-training) mode. */
    if (ctx->config->use_dropout && ctx->dropouts != NULL) {
        size_t hidden_count = ctx->layer_count - 1U;
        for (i = 0; i < hidden_count; ++i) {
            if (ctx->dropouts[i] != NULL) ctx->dropouts[i]->training = 0;
        }
    }

    current = ctx->input_buffer;
    next = ctx->work_buffer_a;
    norm_offset = 0;
    epsilon = ctx->config->norm_epsilon;
    if (epsilon <= 0.0f) epsilon = 1e-5f;

    /* Hidden layers ping-pong between two work buffers; only the final output persists. */
    for (i = 0; i < ctx->layer_count; i++) {
        MlpDenseLayer* layer = ctx->layers[i];

        if (i == ctx->layer_count - 1) {
            /* Output layer: no norm / dropout / skip */
            mlp_dense_forward(layer, ctx->output_buffer, current);
        } else {
            mlp_dense_forward(layer, next, current);

            /* P0: apply RMSNorm in-place */
            if (ctx->config->use_rms_norm && ctx->norm_gamma != NULL) {
                rms_norm_forward(next, next,
                    ctx->norm_gamma + norm_offset,
                    layer->output_size, epsilon);
            }

            /* P0: apply dropout in-place */
            if (ctx->config->use_dropout && ctx->dropouts != NULL && ctx->dropouts[i] != NULL) {
                dropout_forward(next, next, layer->output_size, ctx->dropouts[i]);
            }

            /* P0: apply skip connection (dimensions must match) */
            if (ctx->config->use_skip && ctx->skips != NULL
                && ctx->skips[i].mode != SKIP_NONE) {
                /* Only apply skip when current and next have the same dimension. */
                if (layer->output_size == layer->input_size) {
                    skip_forward(next, next, current, layer->output_size, &ctx->skips[i]);
                }
            }

            norm_offset += layer->output_size;
            current = next;
            next = (next == ctx->work_buffer_a) ? ctx->work_buffer_b : ctx->work_buffer_a;
        }
    }

    return 0;
}

/**
 * @brief Convenience wrapper that performs set-input, step, and get-output.
 */
int nn_mlp_infer_auto_run(void* context, const float* input, float* output) {
    MlpInferContext* ctx = (MlpInferContext*)context;

    if (ctx == NULL || input == NULL || output == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    nn_mlp_infer_set_input(ctx, input, ctx->config->input_size);

    if (nn_mlp_infer_step(ctx) != 0) {
        return ACTION_C_ERR_INTERNAL;
    }

    nn_mlp_infer_get_output(ctx, output, ctx->config->output_size);

    return 0;
}

/**
 * @brief Load weights after validating network hash, layout hash, and ABI tag.
 */
int nn_mlp_load_weights(void* context, FILE* fp) {
    MlpInferContext* ctx = (MlpInferContext*)context;
    uint64_t file_hash;
    uint64_t file_layout_hash;
    uint32_t file_abi_version;
    size_t i;
    int rc;

    if (ctx == NULL || fp == NULL) {
        return 0;
    }

    /* Read and validate the compatibility header before touching layer tensors. */
    rc = (int)fread(&file_hash, sizeof(file_hash), 1, fp);
    if (rc != 1) return 0;

    rc = (int)fread(&file_layout_hash, sizeof(file_layout_hash), 1, fp);
    if (rc != 1) return 0;

    rc = (int)fread(&file_abi_version, sizeof(file_abi_version), 1, fp);
    if (rc != 1) return 0;

    if (ctx->expected_network_hash != 0U &&
        file_hash != ctx->expected_network_hash) {
        return 0;
    }

    if (ctx->expected_layout_hash != 0U &&
        file_layout_hash != ctx->expected_layout_hash) {
        return 0;
    }

    if (file_abi_version != ABI_VERSION) {
        return 0;
    }

    /* Payload layout is strictly layer-major so save/load stay symmetric. */
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

    /* P0: load norm_gamma if present */
    if (ctx->config->use_rms_norm && ctx->norm_gamma != NULL) {
        size_t total_norm = 0;
        for (i = 0; i < ctx->layer_count; ++i) total_norm += ctx->layers[i]->output_size;
        rc = (int)fread(ctx->norm_gamma, sizeof(float), total_norm, fp);
        if ((size_t)rc != total_norm) {
            return 0;
        }
    }

    /* P0: load skip alpha_raw if present */
    if (ctx->config->use_skip && ctx->skips != NULL) {
        for (i = 0; i < ctx->layer_count; ++i) {
            float sr;
            rc = (int)fread(&sr, sizeof(float), 1, fp);
            if (rc != 1) return 0;
            ctx->skips[i].alpha_raw = sr;
        }
    }

    return 1;
}

/**
 * @brief Save weights together with compatibility metadata required at load time.
 */
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

    /* Prefer externally supplied hashes so generated wrappers control compatibility policy. */
    hash = (ctx->expected_network_hash != 0U)
        ? ctx->expected_network_hash
        : nn_mlp_get_network_hash(ctx);
    layout_hash = (ctx->expected_layout_hash != 0U)
        ? ctx->expected_layout_hash
        : compute_layout_hash(ctx->config);

    rc = (int)fwrite(&hash, sizeof(hash), 1, fp);
    if (rc != 1) return 0;

    rc = (int)fwrite(&layout_hash, sizeof(layout_hash), 1, fp);
    if (rc != 1) return 0;

    rc = (int)fwrite(&abi_ver, sizeof(abi_ver), 1, fp);
    if (rc != 1) return 0;

    /* Emit tensors in the same order nn_mlp_load_weights() expects to read them. */
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

    /* P0: save norm_gamma if present */
    if (ctx->config->use_rms_norm && ctx->norm_gamma != NULL) {
        size_t total_norm = 0;
        for (i = 0; i < ctx->layer_count; ++i) total_norm += ctx->layers[i]->output_size;
        rc = (int)fwrite(ctx->norm_gamma, sizeof(float), total_norm, fp);
        if ((size_t)rc != total_norm) {
            return 0;
        }
    }

    /* P0: save skip alpha_raw if present */
    if (ctx->config->use_skip && ctx->skips != NULL) {
        for (i = 0; i < ctx->layer_count; ++i) {
            float sr = ctx->skips[i].alpha_raw;
            rc = (int)fwrite(&sr, sizeof(float), 1, fp);
            if (rc != 1) return 0;
        }
    }

    return 1;
}

/**
 * @brief Expose the current MLP structural hash to save/load callers.
 */
uint64_t nn_mlp_get_network_hash(const void* context) {
    const MlpInferContext* ctx = (const MlpInferContext*)context;

    if (ctx == NULL) {
        return 0;
    }

    return compute_layout_hash(ctx->config);
}

/* ─── VTable backend ─── */

#include "../../nn_backend.h"

static uint32_t mlp_infer_abi_version(void) {
    return ABI_VERSION;
}

static uint64_t mlp_infer_layout_hash(const void* context) {
    const MlpInferContext* ctx = (const MlpInferContext*)context;
    if (ctx == NULL || ctx->config == NULL) return 0;
    return compute_layout_hash(ctx->config);
}

static int mlp_infer_get_output_int(const void* context, float* out, size_t out_size) {
    nn_mlp_infer_get_output((void*)context, out, out_size);
    return 0;
}

static void* mlp_infer_create_vtable(const void* config_blob, size_t config_size,
                                      struct Arena* arena) {
    (void)arena;
    return nn_mlp_infer_create_with_config_blob(config_blob, config_size, 0);
}

static int mlp_infer_save_weights_vtable(const void* context, FILE* fp) {
    return nn_mlp_save_weights((void*)context, fp) ? 0 : -1;
}

const NNInferBackend g_mlp_infer_backend = {
    .type_name        = "mlp",
    .create           = mlp_infer_create_vtable,
    .destroy          = nn_mlp_infer_destroy,
    .step             = nn_mlp_infer_step,
    .get_output       = mlp_infer_get_output_int,
    .save_weights     = mlp_infer_save_weights_vtable,
    .load_weights     = nn_mlp_load_weights,
    .get_network_hash = nn_mlp_get_network_hash,
    .get_layout_hash  = mlp_infer_layout_hash,
    .get_abi_version  = mlp_infer_abi_version,
};
