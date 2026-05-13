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
#include "../../../utils/error.h"

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
    const TransformerInferContext* context,
    const char* question,
    struct TransformerForwardCache* cache
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

    /* ── Attention-weighted aggregation + output projection + residual ── */
    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
            float attended_value = 0.0f;
            float projected_value = cache->input_states[
                transformer_index(seq_index, feature_index, context->model_dim)];

            for (source_index = 0U; source_index < cache->seq_length; ++source_index) {
                attended_value += cache->attention[
                    transformer_index(seq_index, source_index, context->max_seq_length)
                ] * cache->value[
                    transformer_index(source_index, feature_index, context->model_dim)];
            }

            cache->attended[transformer_index(seq_index, feature_index, context->model_dim)] =
                attended_value;

            for (output_index = 0U; output_index < context->model_dim; ++output_index) {
                projected_value += cache->attended[
                    transformer_index(seq_index, output_index, context->model_dim)
                ] * context->output_weight[
                    transformer_index(output_index, feature_index, context->model_dim)];
            }

            cache->projected[transformer_index(seq_index, feature_index, context->model_dim)] =
                projected_value;
            cache->hidden[transformer_index(seq_index, feature_index, context->model_dim)] =
                tanhf(projected_value);
            cache->pooled[feature_index] += cache->hidden[
                transformer_index(seq_index, feature_index, context->model_dim)
            ] / (float)cache->seq_length;
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
