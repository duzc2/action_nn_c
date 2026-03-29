/**
 * @file transformer_train_ops.c
 * @brief Dynamically sized transformer training backend.
 */

#include "transformer_train_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t seq_length;
    size_t* tokens;
    float* input_states;
    float* query;
    float* key;
    float* value;
    float* attention;
    float* attended;
    float* projected;
    float* hidden;
    float* pooled;
    float* logits;
    float* probabilities;
} TransformerTrainCache;

static size_t transformer_index(size_t row, size_t column, size_t column_count) {
    return (row * column_count) + column;
}

static int transformer_train_cache_init(
    TransformerTrainCache* cache,
    const TransformerInferContext* context
) {
    size_t state_count;
    size_t attention_count;

    if (cache == 0 || context == 0) {
        return -1;
    }

    (void)memset(cache, 0, sizeof(*cache));
    state_count = context->max_seq_length * context->model_dim;
    attention_count = context->max_seq_length * context->max_seq_length;

    cache->tokens = (size_t*)calloc(context->max_seq_length, sizeof(size_t));
    cache->input_states = (float*)calloc(state_count, sizeof(float));
    cache->query = (float*)calloc(state_count, sizeof(float));
    cache->key = (float*)calloc(state_count, sizeof(float));
    cache->value = (float*)calloc(state_count, sizeof(float));
    cache->attention = (float*)calloc(attention_count, sizeof(float));
    cache->attended = (float*)calloc(state_count, sizeof(float));
    cache->projected = (float*)calloc(state_count, sizeof(float));
    cache->hidden = (float*)calloc(state_count, sizeof(float));
    cache->pooled = (float*)calloc(context->model_dim, sizeof(float));
    cache->logits = (float*)calloc(context->max_response_classes, sizeof(float));
    cache->probabilities = (float*)calloc(context->max_response_classes, sizeof(float));

    if (cache->tokens == 0 || cache->input_states == 0 || cache->query == 0 ||
        cache->key == 0 || cache->value == 0 || cache->attention == 0 ||
        cache->attended == 0 || cache->projected == 0 || cache->hidden == 0 ||
        cache->pooled == 0 || cache->logits == 0 || cache->probabilities == 0) {
        free(cache->tokens);
        free(cache->input_states);
        free(cache->query);
        free(cache->key);
        free(cache->value);
        free(cache->attention);
        free(cache->attended);
        free(cache->projected);
        free(cache->hidden);
        free(cache->pooled);
        free(cache->logits);
        free(cache->probabilities);
        (void)memset(cache, 0, sizeof(*cache));
        return -1;
    }

    return 0;
}

static void transformer_train_cache_destroy(TransformerTrainCache* cache) {
    if (cache == 0) {
        return;
    }

    free(cache->tokens);
    free(cache->input_states);
    free(cache->query);
    free(cache->key);
    free(cache->value);
    free(cache->attention);
    free(cache->attended);
    free(cache->projected);
    free(cache->hidden);
    free(cache->pooled);
    free(cache->logits);
    free(cache->probabilities);
}

static void transformer_softmax(float* values, size_t count) {
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

static float transformer_vector_norm(const float* values, size_t count) {
    float sum = 0.0f;
    size_t index;

    for (index = 0U; index < count; ++index) {
        sum += values[index] * values[index];
    }

    return sqrtf(sum + 1.0e-6f);
}

static void transformer_apply_gradient(float* parameter, float gradient, float learning_rate) {
    if (gradient > 5.0f) {
        gradient = 5.0f;
    } else if (gradient < -5.0f) {
        gradient = -5.0f;
    }
    *parameter -= learning_rate * gradient;
}

static int transformer_run_training_forward(
    const TransformerInferContext* context,
    const char* question,
    TransformerTrainCache* cache
) {
    size_t seq_index;
    size_t source_index;
    size_t feature_index;
    size_t output_index;
    float scale;

    if (context == 0 || question == 0 || cache == 0) {
        return -1;
    }

    cache->seq_length = nn_transformer_tokenize_text(
        question,
        cache->tokens,
        context->max_seq_length,
        context->vocab_size
    );
    if (cache->seq_length == 0U) {
        return -1;
    }

    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        size_t token_id = cache->tokens[seq_index];

        for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
            cache->input_states[transformer_index(seq_index, feature_index, context->model_dim)] =
                context->token_embedding[transformer_index(token_id, feature_index, context->model_dim)] +
                context->position_embedding[transformer_index(seq_index, feature_index, context->model_dim)];
        }
    }

    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        for (output_index = 0U; output_index < context->model_dim; ++output_index) {
            float q_value = 0.0f;
            float k_value = 0.0f;
            float v_value = 0.0f;

            for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
                float input_value = cache->input_states[
                    transformer_index(seq_index, feature_index, context->model_dim)
                ];

                q_value += input_value * context->query_weight[
                    transformer_index(feature_index, output_index, context->model_dim)
                ];
                k_value += input_value * context->key_weight[
                    transformer_index(feature_index, output_index, context->model_dim)
                ];
                v_value += input_value * context->value_weight[
                    transformer_index(feature_index, output_index, context->model_dim)
                ];
            }

            cache->query[transformer_index(seq_index, output_index, context->model_dim)] = q_value;
            cache->key[transformer_index(seq_index, output_index, context->model_dim)] = k_value;
            cache->value[transformer_index(seq_index, output_index, context->model_dim)] = v_value;
        }
    }

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

    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
            float attended_value = 0.0f;
            float projected_value = cache->input_states[
                transformer_index(seq_index, feature_index, context->model_dim)
            ];

            for (source_index = 0U; source_index < cache->seq_length; ++source_index) {
                attended_value += cache->attention[
                    transformer_index(seq_index, source_index, context->max_seq_length)
                ] * cache->value[
                    transformer_index(source_index, feature_index, context->model_dim)
                ];
            }

            cache->attended[transformer_index(seq_index, feature_index, context->model_dim)] =
                attended_value;

            for (output_index = 0U; output_index < context->model_dim; ++output_index) {
                projected_value += cache->attended[
                    transformer_index(seq_index, output_index, context->model_dim)
                ] * context->output_weight[
                    transformer_index(output_index, feature_index, context->model_dim)
                ];
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

    for (output_index = 0U; output_index < context->class_count; ++output_index) {
        float dot = 0.0f;
        float pooled_norm = transformer_vector_norm(cache->pooled, context->model_dim);
        float class_norm = transformer_vector_norm(
            context->classifier_weight + (output_index * context->model_dim),
            context->model_dim
        );
        float logit = context->classifier_bias[output_index];

        for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
            dot += cache->pooled[feature_index] * context->classifier_weight[
                transformer_index(output_index, feature_index, context->model_dim)
            ];
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

int nn_transformer_train_step(void* context) {
    TransformerTrainContext* train_ctx = (TransformerTrainContext*)context;
    TransformerInferContext* infer_ctx;
    TransformerTrainCache cache;
    float* classifier_gradient = 0;
    float* classifier_bias_gradient = 0;
    float* dlogits = 0;
    float loss;
    float pooled_norm;
    size_t class_index;
    size_t feature_index;
    int target_class;
    int rc;

    if (train_ctx == 0 || train_ctx->infer_ctx == 0 ||
        train_ctx->current_question == 0 || train_ctx->current_answer == 0) {
        return -1;
    }

    infer_ctx = train_ctx->infer_ctx;
    target_class = nn_transformer_find_or_add_class(infer_ctx, train_ctx->current_answer);
    if (target_class < 0) {
        return -1;
    }

    rc = transformer_train_cache_init(&cache, infer_ctx);
    if (rc != 0) {
        return rc;
    }

    rc = transformer_run_training_forward(infer_ctx, train_ctx->current_question, &cache);
    if (rc != 0) {
        transformer_train_cache_destroy(&cache);
        return rc;
    }

    classifier_gradient = (float*)calloc(
        infer_ctx->class_count * infer_ctx->model_dim,
        sizeof(float)
    );
    classifier_bias_gradient = (float*)calloc(infer_ctx->class_count, sizeof(float));
    dlogits = (float*)calloc(infer_ctx->class_count, sizeof(float));
    if (classifier_gradient == 0 || classifier_bias_gradient == 0 || dlogits == 0) {
        free(classifier_gradient);
        free(classifier_bias_gradient);
        free(dlogits);
        transformer_train_cache_destroy(&cache);
        return -1;
    }

    loss = -logf(cache.probabilities[(size_t)target_class] + 1.0e-6f);
    for (class_index = 0U; class_index < infer_ctx->class_count; ++class_index) {
        dlogits[class_index] = cache.probabilities[class_index];
    }
    dlogits[(size_t)target_class] -= 1.0f;

    for (class_index = 0U; class_index < infer_ctx->class_count; ++class_index) {
        classifier_bias_gradient[class_index] = dlogits[class_index];

        for (feature_index = 0U; feature_index < infer_ctx->model_dim; ++feature_index) {
            classifier_gradient[
                transformer_index(class_index, feature_index, infer_ctx->model_dim)
            ] = dlogits[class_index] * cache.pooled[feature_index];
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

    pooled_norm = transformer_vector_norm(cache.pooled, infer_ctx->model_dim);
    for (feature_index = 0U; feature_index < infer_ctx->model_dim; ++feature_index) {
        size_t weight_index = transformer_index(
            (size_t)target_class,
            feature_index,
            infer_ctx->model_dim
        );
        float normalized_feature = cache.pooled[feature_index] / pooled_norm;

        infer_ctx->classifier_weight[weight_index] =
            (0.85f * infer_ctx->classifier_weight[weight_index]) + (0.15f * normalized_feature);
    }
    infer_ctx->classifier_bias[(size_t)target_class] += 0.01f;

    train_ctx->last_loss = loss;
    train_ctx->total_steps += 1U;
    train_ctx->cumulative_loss += loss;
    train_ctx->average_loss = train_ctx->cumulative_loss / (float)train_ctx->total_steps;

    free(classifier_gradient);
    free(classifier_bias_gradient);
    free(dlogits);
    transformer_train_cache_destroy(&cache);
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

    if (train_ctx == 0 || train_ctx->infer_ctx == 0 || input == 0 || output_gradient == 0) {
        return -1;
    }

    infer_ctx = train_ctx->infer_ctx;
    if (infer_ctx->graph_input_size == 0U || infer_ctx->graph_output_size == 0U) {
        return -1;
    }

    output_cache = (float*)calloc(infer_ctx->graph_output_size, sizeof(float));
    if (output_cache == 0) {
        return -1;
    }

    if (nn_transformer_graph_run(infer_ctx, input, output_cache) != 0) {
        free(output_cache);
        return -1;
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
    free(output_cache);
    return 0;
}
