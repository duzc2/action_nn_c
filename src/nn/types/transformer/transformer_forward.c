/**
 * @file transformer_forward.c
 * @brief Unified transformer forward pass implementation.
 *
 * A single copy of the forward logic used by both inference (TF_FWD_INFER)
 * and training (TF_FWD_TRAIN).  The flags parameter is passed through the
 * public signature for future expansion but all modes currently execute the
 * same arithmetic; the distinction matters only for the caller (whether it
 * uses the cache for backprop or just for prediction).
 */

#include "transformer_forward.h"

#include <math.h>
#include <string.h>
#include "../../../utils/error.h"
#include "../../norm/rms_norm.h"
#include "../../dropout/dropout.h"
#include "../../residual/skip_connection.h"

size_t transformer_index(size_t row, size_t column, size_t column_count) {
    return (row * column_count) + column;
}

void transformer_softmax(float* values, size_t count) {
    float max_value = values[0];
    float sum = 0.0f;
    size_t index;

    for (index = 1U; index < count; ++index) {
        if (values[index] > max_value) {
            max_value = values[index];
        }
    }
    for (index = 0U; index < count; ++index) {
        values[index] = expf(values[index] - max_value);
        sum += values[index];
    }
    if (sum <= 0.0f) {
        float uniform = 1.0f / (float)count;
        for (index = 0U; index < count; ++index) {
            values[index] = uniform;
        }
        return;
    }
    for (index = 0U; index < count; ++index) {
        values[index] /= sum;
    }
}

float transformer_vector_norm(const float* values, size_t count) {
    float sum = 0.0f;
    size_t index;

    for (index = 0U; index < count; ++index) {
        sum += values[index] * values[index];
    }
    return sqrtf(sum + 1.0e-6f);
}

int transformer_run_forward(
    TransformerInferContext* restrict context,
    const char* restrict question,
    struct TransformerForwardCache* restrict cache
) {
    size_t seq_index;
    size_t source_index;
    size_t feature_index;
    size_t output_index;
    float scale;

    if (context == 0 || question == 0 || cache == 0) {
        return ACTION_C_ERR_NULL_POINTER;
    }

    cache->seq_length = nn_transformer_tokenize_text(
        question,
        cache->tokens,
        context->max_seq_length,
        context->vocab_size
    );
    if (cache->seq_length == 0U) {
        return ACTION_C_ERR_DIM_MISMATCH;
    }

    /* ── Token + Position embeddings ── */
    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        size_t token_id = cache->tokens[seq_index];
        for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
            cache->input_states[transformer_index(seq_index, feature_index, context->model_dim)] =
                context->token_embedding[transformer_index(token_id, feature_index, context->model_dim)] +
                context->position_embedding[transformer_index(seq_index, feature_index, context->model_dim)];
        }
    }

    /* ── Q / K / V projections ── */
    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        for (output_index = 0U; output_index < context->model_dim; ++output_index) {
            float q_value = 0.0f;
            float k_value = 0.0f;
            float v_value = 0.0f;

            for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
                float input_value = cache->input_states[
                    transformer_index(seq_index, feature_index, context->model_dim)];
                q_value += input_value * context->query_weight[
                    transformer_index(feature_index, output_index, context->model_dim)];
                k_value += input_value * context->key_weight[
                    transformer_index(feature_index, output_index, context->model_dim)];
                v_value += input_value * context->value_weight[
                    transformer_index(feature_index, output_index, context->model_dim)];
            }

            cache->query[transformer_index(seq_index, output_index, context->model_dim)] = q_value;
            cache->key[transformer_index(seq_index, output_index, context->model_dim)] = k_value;
            cache->value[transformer_index(seq_index, output_index, context->model_dim)] = v_value;
        }
    }

    /* ── Scaled dot-product attention ── */
    scale = sqrtf((float)context->model_dim);
    if (scale <= 0.0f) {
        scale = 1.0f;
    }

    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        float* attention_row = cache->attention + (seq_index * context->max_seq_length);
        for (source_index = 0U; source_index < cache->seq_length; ++source_index) {
            float score = 0.0f;
            for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
                score += cache->query[transformer_index(seq_index, feature_index, context->model_dim)] *
                    cache->key[transformer_index(source_index, feature_index, context->model_dim)];
            }
            attention_row[source_index] = score / scale;
        }
        transformer_softmax(attention_row, cache->seq_length);
    }

    /* ── Step 1: Attention-weighted aggregation + output projection → attn_out ── */
    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
            float attended_value = 0.0f;
            float attn_out_val  = 0.0f;

            for (source_index = 0U; source_index < cache->seq_length; ++source_index) {
                attended_value += cache->attention[
                    transformer_index(seq_index, source_index, context->max_seq_length)
                ] * cache->value[
                    transformer_index(source_index, feature_index, context->model_dim)];
            }

            cache->attended[transformer_index(seq_index, feature_index, context->model_dim)] =
                attended_value;

            for (output_index = 0U; output_index < context->model_dim; ++output_index) {
                attn_out_val += cache->attended[
                    transformer_index(seq_index, output_index, context->model_dim)
                ] * context->output_weight[
                    transformer_index(output_index, feature_index, context->model_dim)];
            }

            cache->attn_out[transformer_index(seq_index, feature_index, context->model_dim)] =
                attn_out_val;
        }
    }

    /* ── Step 2: RMSNorm → Dropout → Skip/Residual → Tanh → Mean Pooling ── */
    (void)memset(cache->pooled, 0, context->model_dim * sizeof(float));
    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        float* attn_seq = cache->attn_out + (seq_index * context->model_dim);
        float* proj_seq = cache->projected + (seq_index * context->model_dim);
        float* hidn_seq = cache->hidden + (seq_index * context->model_dim);
        const float* inpt_seq = cache->input_states + (seq_index * context->model_dim);

        /* P0: RMSNorm on attention output (before residual) */
        if (context->use_rms_norm && context->norm_gamma_attn != NULL) {
            if (context->p0_attn_pre_norm != NULL) {
                (void)memcpy(context->p0_attn_pre_norm + (seq_index * context->model_dim),
                    attn_seq, context->model_dim * sizeof(float));
            }
            (void)rms_norm_forward(attn_seq, attn_seq,
                context->norm_gamma_attn, context->model_dim, context->norm_epsilon);
        }

        /* P0: Dropout on attention output */
        if (context->use_dropout) {
            (void)dropout_forward(attn_seq, attn_seq, context->model_dim, &context->dropout_attn);
        }

        /* P0: Residual / Skip connection */
        if (context->use_skip && context->skip_attn.mode != SKIP_NONE) {
            if (context->p0_skip_x_cache != NULL) {
                (void)memcpy(context->p0_skip_x_cache + (seq_index * context->model_dim),
                    inpt_seq, context->model_dim * sizeof(float));
            }
            if (context->p0_skip_fx_cache != NULL) {
                (void)memcpy(context->p0_skip_fx_cache + (seq_index * context->model_dim),
                    attn_seq, context->model_dim * sizeof(float));
            }
            (void)skip_forward(proj_seq, attn_seq, inpt_seq, context->model_dim,
                &context->skip_attn);
        } else {
            /* Default residual: projected = input + attn_out */
            for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
                proj_seq[feature_index] = inpt_seq[feature_index] + attn_seq[feature_index];
            }
        }

        /* Tanh activation + mean pooling */
        for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
            hidn_seq[feature_index] = tanhf(proj_seq[feature_index]);
            cache->pooled[feature_index] += hidn_seq[feature_index] / (float)cache->seq_length;
        }
    }

    /* ── Cosine-similarity classifier ── */
    for (output_index = 0U; output_index < context->class_count; ++output_index) {
        float dot = 0.0f;
        float pooled_norm = transformer_vector_norm(cache->pooled, context->model_dim);
        float class_norm = transformer_vector_norm(
            context->classifier_weight + (output_index * context->model_dim),
            context->model_dim);
        float logit = context->classifier_bias[output_index];

        for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
            dot += cache->pooled[feature_index] * context->classifier_weight[
                transformer_index(output_index, feature_index, context->model_dim)];
        }

        logit += 8.0f * dot / (pooled_norm * class_norm);
        cache->logits[output_index] = logit;
        cache->probabilities[output_index] = logit;
    }

    if (context->class_count > 0U) {
        transformer_softmax(cache->probabilities, context->class_count);
    }

    return 0;
}
