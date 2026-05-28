/**
 * @file transformer_train_ops.c
 * @brief Dynamically sized transformer training backend.
 */

#include "transformer_train_ops.h"
#include "transformer_forward.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../../../utils/error.h"
#include "../../norm/rms_norm.h"
#include "../../residual/skip_connection.h"

static void transformer_apply_gradient(float* parameter, float gradient, float learning_rate) {
    if (gradient > 5.0f) {
        gradient = 5.0f;
    } else if (gradient < -5.0f) {
        gradient = -5.0f;
    }
    *parameter -= learning_rate * gradient;
}

int nn_transformer_train_step(void* context) {
    TransformerTrainContext* train_ctx = (TransformerTrainContext*)context;
    TransformerInferContext* infer_ctx;
    struct TransformerForwardCache* cache = 0;
    float* classifier_gradient = 0;
    float* classifier_bias_gradient = 0;
    float* dlogits = 0;
    float loss;
    float pooled_norm;
    size_t class_index;
    size_t feature_index;
    size_t _arena_mark;
    int target_class;

    if (train_ctx == 0 || train_ctx->infer_ctx == 0 ||
        train_ctx->current_question == 0 || train_ctx->current_answer == 0) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    infer_ctx = train_ctx->infer_ctx;
    target_class = nn_transformer_find_or_add_class(infer_ctx, train_ctx->current_answer);
    if (target_class < 0) {
        return ACTION_C_ERR_NOT_FOUND;
    }

    /* Reuse the pre-allocated forward cache and arena for all scratch buffers. */
    cache = infer_ctx->forward_cache;
    if (cache == 0) {
        return ACTION_C_ERR_NULL_POINTER;
    }
    _arena_mark = arena_snapshot(infer_ctx->arena);

    /* Enable dropout in training mode for this forward pass. */
    if (infer_ctx->use_dropout) {
        infer_ctx->dropout_attn.training = 1;
    }

    /* Propagate P0 backward caches from train context to infer context. */
    infer_ctx->p0_attn_pre_norm = train_ctx->attn_pre_norm;
    infer_ctx->p0_skip_x_cache  = train_ctx->skip_x_cache;
    infer_ctx->p0_skip_fx_cache = train_ctx->skip_fx_cache;

    if (transformer_run_forward(infer_ctx, train_ctx->current_question, cache) != 0) {
        arena_restore(infer_ctx->arena, _arena_mark);
        return ACTION_C_ERR_INTERNAL;
    }

    classifier_gradient = ARENA_CALLOC(infer_ctx->arena, float,
                                       infer_ctx->class_count * infer_ctx->model_dim);
    classifier_bias_gradient = ARENA_CALLOC(infer_ctx->arena, float, infer_ctx->class_count);
    dlogits = ARENA_CALLOC(infer_ctx->arena, float, infer_ctx->class_count);
    if (classifier_gradient == 0 || classifier_bias_gradient == 0 || dlogits == 0) {
        arena_restore(infer_ctx->arena, _arena_mark);
        return ACTION_C_ERR_NULL_POINTER;
    }

    loss = -logf(cache->probabilities[(size_t)target_class] + 1.0e-6f);
    for (class_index = 0U; class_index < infer_ctx->class_count; ++class_index) {
        dlogits[class_index] = cache->probabilities[class_index];
    }
    dlogits[(size_t)target_class] -= 1.0f;

    for (class_index = 0U; class_index < infer_ctx->class_count; ++class_index) {
        classifier_bias_gradient[class_index] = dlogits[class_index];

        for (feature_index = 0U; feature_index < infer_ctx->model_dim; ++feature_index) {
            classifier_gradient[
                transformer_index(class_index, feature_index, infer_ctx->model_dim)
            ] = dlogits[class_index] * cache->pooled[feature_index];
        }
    }

    for (class_index = 0U; class_index < infer_ctx->class_count; ++class_index) {
        transformer_apply_gradient(
            &infer_ctx->classifier_bias[class_index],
            classifier_bias_gradient[class_index],
            train_ctx->learning_rate
        );

        for (feature_index = 0U; feature_index < infer_ctx->model_dim; ++feature_index) {
            size_t weight_index = transformer_index(class_index, feature_index, infer_ctx->model_dim);

            transformer_apply_gradient(
                &infer_ctx->classifier_weight[weight_index],
                classifier_gradient[weight_index],
                train_ctx->learning_rate
            );
        }
    }

    pooled_norm = transformer_vector_norm(cache->pooled, infer_ctx->model_dim);
    for (feature_index = 0U; feature_index < infer_ctx->model_dim; ++feature_index) {
        size_t weight_index = transformer_index(
            (size_t)target_class,
            feature_index,
            infer_ctx->model_dim
        );
        float normalized_feature = cache->pooled[feature_index] / pooled_norm;

        infer_ctx->classifier_weight[weight_index] =
            (0.85f * infer_ctx->classifier_weight[weight_index]) + (0.15f * normalized_feature);
    }
    infer_ctx->classifier_bias[(size_t)target_class] += 0.01f;

    train_ctx->last_loss = loss;
    train_ctx->total_steps += 1U;
    train_ctx->cumulative_loss += loss;
    train_ctx->average_loss = train_ctx->cumulative_loss / (float)train_ctx->total_steps;

    arena_restore(infer_ctx->arena, _arena_mark);
    return 0;
}

int nn_transformer_train_step_with_output_gradient(
    TransformerTrainContext* train_ctx,
    const float* input,
    const float* output_gradient,
    float* input_gradient
) {
    TransformerInferContext* infer_ctx;
    float* output_cache;
    size_t output_index;
    size_t _arena_mark;

    if (train_ctx == 0 || train_ctx->infer_ctx == 0 || input == 0 || output_gradient == 0) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    infer_ctx = train_ctx->infer_ctx;
    if (infer_ctx->graph_input_size == 0U || infer_ctx->graph_output_size == 0U) {
        return ACTION_C_ERR_DIM_MISMATCH;
    }

    _arena_mark = arena_snapshot(infer_ctx->arena);
    output_cache = ARENA_CALLOC(infer_ctx->arena, float, infer_ctx->graph_output_size);
    if (output_cache == 0) {
        arena_restore(infer_ctx->arena, _arena_mark);
        return ACTION_C_ERR_NO_MEMORY;
    }

    if (nn_transformer_graph_run(infer_ctx, input, output_cache) != 0) {
        arena_restore(infer_ctx->arena, _arena_mark);
        return ACTION_C_ERR_INTERNAL;
    }

    if (input_gradient != 0) {
        (void)memset(input_gradient, 0, infer_ctx->graph_input_size * sizeof(float));
    }

    for (output_index = 0U; output_index < infer_ctx->graph_output_size; ++output_index) {
        float dz = output_gradient[output_index] *
            (1.0f - (output_cache[output_index] * output_cache[output_index]));
        size_t input_index;

        transformer_apply_gradient(
            &infer_ctx->graph_projection_bias[output_index],
            dz,
            train_ctx->learning_rate
        );

        for (input_index = 0U; input_index < infer_ctx->graph_input_size; ++input_index) {
            size_t weight_index = transformer_index(
                input_index,
                output_index,
                infer_ctx->graph_output_size
            );
            float old_weight = infer_ctx->graph_projection_weight[weight_index];

            if (input_gradient != 0) {
                input_gradient[input_index] += dz * old_weight;
                if (input_index == output_index) {
                    input_gradient[input_index] += dz;
                }
            }

            transformer_apply_gradient(
                &infer_ctx->graph_projection_weight[weight_index],
                dz * input[input_index],
                train_ctx->learning_rate
            );
        }
    }

    train_ctx->total_steps += 1U;
    arena_restore(infer_ctx->arena, _arena_mark);
    return 0;
}

/* ─── VTable backend ─── */

#include "../../nn_backend.h"

static void transformer_train_destroy(void* context);

static void* transformer_train_create_vtable(const void* config_blob, size_t config_size,
                                              const void* infer_config_blob, size_t infer_config_size,
                                              struct Arena* arena) {
    TransformerInferContext* infer_ctx;
    TransformerTrainContext* train_ctx;
    const TransformerModelConfig* model_cfg;
    const TransformerTrainConfig* tr_cfg;
    (void)arena;

    if (config_blob == NULL || config_size < sizeof(TransformerTrainConfig)) return NULL;
    tr_cfg = (const TransformerTrainConfig*)config_blob;

    if (infer_config_blob == NULL || infer_config_size < sizeof(TransformerModelConfig)) return NULL;
    model_cfg = (const TransformerModelConfig*)infer_config_blob;

    infer_ctx = (TransformerInferContext*)calloc(1, sizeof(TransformerInferContext));
    if (infer_ctx == NULL) return NULL;
    nn_transformer_init_parameters(infer_ctx, model_cfg, 0, 0);

    train_ctx = (TransformerTrainContext*)calloc(1, sizeof(TransformerTrainContext));
    if (train_ctx == NULL) {
        nn_transformer_infer_destroy(infer_ctx);
        return NULL;
    }
    train_ctx->infer_ctx = infer_ctx;
    train_ctx->learning_rate = tr_cfg->learning_rate;

    /* Allocate P0 backward-pass cache buffers (state_count each). */
    {
        size_t state_count = model_cfg->max_seq_length * model_cfg->model_dim;
        train_ctx->attn_pre_norm = (float*)calloc(state_count, sizeof(float));
        train_ctx->skip_x_cache  = (float*)calloc(state_count, sizeof(float));
        train_ctx->skip_fx_cache = (float*)calloc(state_count, sizeof(float));
        if (train_ctx->attn_pre_norm == NULL || train_ctx->skip_x_cache == NULL ||
            train_ctx->skip_fx_cache == NULL) {
            transformer_train_destroy(train_ctx);
            return NULL;
        }
    }

    return train_ctx;
}

static void transformer_train_destroy(void* context) {
    TransformerTrainContext* ctx = (TransformerTrainContext*)context;
    if (ctx == NULL) return;

    /* Null P0 cache pointers in infer context before destroying it. */
    if (ctx->infer_ctx != NULL) {
        ctx->infer_ctx->p0_attn_pre_norm = NULL;
        ctx->infer_ctx->p0_skip_x_cache  = NULL;
        ctx->infer_ctx->p0_skip_fx_cache = NULL;
    }

    free(ctx->attn_pre_norm);
    free(ctx->skip_x_cache);
    free(ctx->skip_fx_cache);

    if (ctx->infer_ctx != NULL) nn_transformer_infer_destroy(ctx->infer_ctx);
    free(ctx);
}

static int transformer_train_step_vtable(void* context, const float* input, const float* target) {
    (void)input;
    (void)target;
    return nn_transformer_train_step(context);
}

static int transformer_train_step_with_data_vtable(void* context, const float* input,
                                                    const float* target, float* out_grad) {
    (void)input;
    (void)target;
    (void)out_grad;
    return nn_transformer_train_step(context);
}

static int transformer_train_save_ckpt_vtable(void* context, FILE* fp) {
    (void)context;
    (void)fp;
    return -1;
}

static int transformer_train_load_ckpt_vtable(void* context, FILE* fp) {
    (void)context;
    (void)fp;
    return -1;
}

const NNTrainBackend g_transformer_train_backend = {
    .type_name        = "transformer",
    .create           = transformer_train_create_vtable,
    .destroy          = transformer_train_destroy,
    .step             = transformer_train_step_vtable,
    .step_with_data   = transformer_train_step_with_data_vtable,
    .save_checkpoint  = transformer_train_save_ckpt_vtable,
    .load_checkpoint  = transformer_train_load_ckpt_vtable,
};
