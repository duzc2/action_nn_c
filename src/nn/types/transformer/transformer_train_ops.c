/**
 * @file transformer_train_ops.c
 * @brief Tiny transformer-style training backend used by generated code.
 *
 * This backend keeps training intentionally lightweight so the generated
 * pipeline can demonstrate graph training, checkpointing, and registry-based
 * dispatch without requiring a production-scale transformer implementation.
 */

#include "transformer_train_ops.h"

#include <math.h>
#include <string.h>

/**
 * @section transformer_train_design Tiny transformer training notes
 *
 * The transformer demo uses an intentionally lightweight training rule: predict
 * an answer class, compare it with the target class, and nudge the classifier
 * and selected embedding weights toward lower loss. The goal is not to model a
 * full modern optimizer, but to expose enough structure for profiler-generated
 * training and checkpoint flows to exercise a non-trivial backend.
 */

/**
 * @brief Forward scratch space reused by the training step.
 *
 * Training needs the same intermediate tensors as inference, but it also needs
 * them grouped into one cache object so classifier gradients can be derived
 * without mutating the persistent inference context mid-step.
 */
typedef struct {
    size_t seq_length;
    uint8_t tokens[TRANSFORMER_MAX_SEQ_LENGTH];
    float input_states[TRANSFORMER_MAX_SEQ_LENGTH][TRANSFORMER_MAX_MODEL_DIM];
    float query[TRANSFORMER_MAX_SEQ_LENGTH][TRANSFORMER_MAX_MODEL_DIM];
    float key[TRANSFORMER_MAX_SEQ_LENGTH][TRANSFORMER_MAX_MODEL_DIM];
    float value[TRANSFORMER_MAX_SEQ_LENGTH][TRANSFORMER_MAX_MODEL_DIM];
    float attention[TRANSFORMER_MAX_SEQ_LENGTH][TRANSFORMER_MAX_SEQ_LENGTH];
    float attended[TRANSFORMER_MAX_SEQ_LENGTH][TRANSFORMER_MAX_MODEL_DIM];
    float projected[TRANSFORMER_MAX_SEQ_LENGTH][TRANSFORMER_MAX_MODEL_DIM];
    float hidden[TRANSFORMER_MAX_SEQ_LENGTH][TRANSFORMER_MAX_MODEL_DIM];
    float pooled[TRANSFORMER_MAX_MODEL_DIM];
    float logits[TRANSFORMER_MAX_RESPONSE_CLASSES];
    float probabilities[TRANSFORMER_MAX_RESPONSE_CLASSES];
} TransformerTrainCache;

/**
 * @brief Normalize a probability vector in-place with softmax.
 */
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

/**
 * @brief Compute the Euclidean norm used to scale gradient updates.
 */
static float transformer_vector_norm(const float* values, size_t count) {
    float sum = 0.0f;
    size_t index;

    for (index = 0U; index < count; ++index) {
        sum += values[index] * values[index];
    }

    return sqrtf(sum + 1.0e-6f);
}

/**
 * @brief Apply one normalized gradient update to a dense parameter vector.
 */
static void transformer_apply_gradient(float* parameter, float gradient, float learning_rate) {
    if (gradient > 5.0f) {
        gradient = 5.0f;
    } else if (gradient < -5.0f) {
        gradient = -5.0f;
    }
    *parameter -= learning_rate * gradient;
}

/**
 * @brief Forward stage: run class prediction while collecting training buffers.
 *
 * Training reuses the inference-style forward path but preserves extra cached
 * values needed for loss evaluation and lightweight parameter updates. Keeping
 * that logic in one helper prevents prediction and training from diverging.
 */
static void transformer_run_training_forward(
    const TransformerInferContext* context,
    const char* question,
    TransformerTrainCache* cache
) {
    size_t seq_index;
    size_t source_index;
    size_t feature_index;
    size_t output_index;
    float scale;

    /* Stage 1: tokenize the question and reset scratch buffers. */
    (void)memset(cache, 0, sizeof(*cache));
    cache->seq_length = nn_transformer_tokenize_text(
        question,
        cache->tokens,
        context->max_seq_length
    );

    /* Stage 2: build token-plus-position input states. */
    /* Stage 3: form query/key/value projections for each token. */
    /* Stage 4: compute normalized attention weights for every token pair. */
    /* Stage 5: aggregate attended states, project them, and mean-pool. */
    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        uint8_t token_id = cache->tokens[seq_index];
        for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
            cache->input_states[seq_index][feature_index] =
                context->token_embedding[token_id][feature_index] +
                context->position_embedding[seq_index][feature_index];
        }
    }

    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        for (output_index = 0U; output_index < context->model_dim; ++output_index) {
            float q_value = 0.0f;
            float k_value = 0.0f;
            float v_value = 0.0f;

            for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
                float input_value = cache->input_states[seq_index][feature_index];
                q_value += input_value * context->query_weight[feature_index][output_index];
                k_value += input_value * context->key_weight[feature_index][output_index];
                v_value += input_value * context->value_weight[feature_index][output_index];
            }

            cache->query[seq_index][output_index] = q_value;
            cache->key[seq_index][output_index] = k_value;
            cache->value[seq_index][output_index] = v_value;
        }
    }

    scale = sqrtf((float)context->model_dim);
    if (scale <= 0.0f) {
        scale = 1.0f;
    }

    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        for (source_index = 0U; source_index < cache->seq_length; ++source_index) {
            float score = 0.0f;

            for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
                score += cache->query[seq_index][feature_index] *
                    cache->key[source_index][feature_index];
            }

            cache->attention[seq_index][source_index] = score / scale;
        }
        transformer_softmax(cache->attention[seq_index], cache->seq_length);
    }

    for (seq_index = 0U; seq_index < cache->seq_length; ++seq_index) {
        for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
            float attended_value = 0.0f;
            float projected_value = cache->input_states[seq_index][feature_index];

            for (source_index = 0U; source_index < cache->seq_length; ++source_index) {
                attended_value +=
                    cache->attention[seq_index][source_index] *
                    cache->value[source_index][feature_index];
            }
            cache->attended[seq_index][feature_index] = attended_value;

            for (output_index = 0U; output_index < context->model_dim; ++output_index) {
                projected_value += cache->attended[seq_index][output_index] *
                    context->output_weight[output_index][feature_index];
            }

            cache->projected[seq_index][feature_index] = projected_value;
            cache->hidden[seq_index][feature_index] = tanhf(projected_value);
            cache->pooled[feature_index] +=
                cache->hidden[seq_index][feature_index] / (float)cache->seq_length;
        }
    }

    /* Stage 6: score the pooled sentence representation against each class. */
    for (output_index = 0U; output_index < context->class_count; ++output_index) {
        float dot = 0.0f;
        float pooled_norm = transformer_vector_norm(cache->pooled, context->model_dim);
        float class_norm = transformer_vector_norm(
            context->classifier_weight[output_index],
            context->model_dim
        );
        float logit = context->classifier_bias[output_index];
        for (feature_index = 0U; feature_index < context->model_dim; ++feature_index) {
            dot += cache->pooled[feature_index] *
                context->classifier_weight[output_index][feature_index];
        }
        logit += 8.0f * dot / (pooled_norm * class_norm);
        cache->logits[output_index] = logit;
        cache->probabilities[output_index] = logit;
    }

    transformer_softmax(cache->probabilities, context->class_count);
}

/**
 * @brief Public API stage: perform one supervised update on question/answer text.
 *
 * This path treats the target answer as a class label, computes cross-entropy
 * on the predicted class distribution, and then nudges classifier parameters
 * and the target-class prototype toward the current pooled sentence feature.
 * The simplified update rule is deliberate: it keeps the demo backend readable
 * while still exercising the profiler's training contracts end to end.
 */
int nn_transformer_train_step(void* context) {
    TransformerTrainContext* train_ctx = (TransformerTrainContext*)context;
    TransformerInferContext* infer_ctx;
    TransformerTrainCache cache;
    float classifier_gradient[TRANSFORMER_MAX_RESPONSE_CLASSES][TRANSFORMER_MAX_MODEL_DIM];
    float classifier_bias_gradient[TRANSFORMER_MAX_RESPONSE_CLASSES];
    float dlogits[TRANSFORMER_MAX_RESPONSE_CLASSES];
    float loss;
    size_t class_index;
    size_t feature_index;
    int target_class;

    if (train_ctx == 0 || train_ctx->infer_ctx == 0 ||
        train_ctx->current_question == 0 || train_ctx->current_answer == 0) {
        return -1;
    }

    infer_ctx = train_ctx->infer_ctx;
    /* Lazily map the target answer text into a trainable classifier slot. */
    target_class = nn_transformer_find_or_add_class(infer_ctx, train_ctx->current_answer);
    if (target_class < 0) {
        return -1;
    }

    /* Reuse the shared forward routine so train and infer stay semantically aligned. */
    transformer_run_training_forward(infer_ctx, train_ctx->current_question, &cache);

    loss = -logf(cache.probabilities[target_class] + 1.0e-6f);
    train_ctx->last_loss = loss;
    train_ctx->total_steps += 1U;
    train_ctx->cumulative_loss += loss;
    train_ctx->average_loss = train_ctx->cumulative_loss / (float)train_ctx->total_steps;

    (void)memset(classifier_gradient, 0, sizeof(classifier_gradient));
    (void)memset(classifier_bias_gradient, 0, sizeof(classifier_bias_gradient));
    (void)memset(dlogits, 0, sizeof(dlogits));

    /* Cross-entropy on the class distribution gives a simple dL/dlogit signal. */
    for (class_index = 0U; class_index < infer_ctx->class_count; ++class_index) {
        dlogits[class_index] = cache.probabilities[class_index];
    }
    dlogits[target_class] -= 1.0f;

    /* Convert class-level gradients into classifier weight and bias updates. */
    for (class_index = 0U; class_index < infer_ctx->class_count; ++class_index) {
        classifier_bias_gradient[class_index] = dlogits[class_index];
        for (feature_index = 0U; feature_index < infer_ctx->model_dim; ++feature_index) {
            classifier_gradient[class_index][feature_index] =
                dlogits[class_index] * cache.pooled[feature_index];
        }
    }

    for (class_index = 0U; class_index < infer_ctx->class_count; ++class_index) {
        transformer_apply_gradient(
            &infer_ctx->classifier_bias[class_index],
            classifier_bias_gradient[class_index],
            train_ctx->learning_rate
        );
        for (feature_index = 0U; feature_index < infer_ctx->model_dim; ++feature_index) {
            transformer_apply_gradient(
                &infer_ctx->classifier_weight[class_index][feature_index],
                classifier_gradient[class_index][feature_index],
                train_ctx->learning_rate
            );
        }
    }

    {
        /* Blend the target-class prototype toward the normalized pooled feature. */
        float pooled_norm = transformer_vector_norm(cache.pooled, infer_ctx->model_dim);
        float prototype_rate = 0.15f;

        for (feature_index = 0U; feature_index < infer_ctx->model_dim; ++feature_index) {
            float normalized_feature = cache.pooled[feature_index] / pooled_norm;
            infer_ctx->classifier_weight[target_class][feature_index] =
                (1.0f - prototype_rate) *
                infer_ctx->classifier_weight[target_class][feature_index] +
                prototype_rate * normalized_feature;
        }
        infer_ctx->classifier_bias[target_class] += 0.01f;
    }

    return 0;
}

/**
 * @brief Graph stage: convert an external output gradient into parameter updates.
 *
 * Graph mode intentionally trains only the lightweight projection path exposed
 * by nn_transformer_graph_run(). That keeps composite-graph differentiation
 * simple while still allowing the transformer leaf to adapt to upstream losses.
 * The routine therefore acts as a contract bridge rather than a full language-
 * model training implementation.
 */
int nn_transformer_train_step_with_output_gradient(
    TransformerTrainContext* train_ctx,
    const float* input,
    const float* output_gradient,
    float* input_gradient
) {
    TransformerInferContext* infer_ctx;
    float output_cache[TRANSFORMER_MAX_MODEL_DIM];
    size_t output_index;

    if (train_ctx == 0 || train_ctx->infer_ctx == 0 || input == 0 || output_gradient == 0) {
        return -1;
    }

    infer_ctx = train_ctx->infer_ctx;
    if (infer_ctx->graph_input_size == 0U ||
        infer_ctx->graph_output_size == 0U ||
        infer_ctx->graph_input_size > infer_ctx->model_dim ||
        infer_ctx->graph_output_size > infer_ctx->model_dim) {
        return -1;
    }

    /* Rebuild the forward output so tanh derivative can be applied correctly. */
    if (nn_transformer_graph_run(infer_ctx, input, output_cache) != 0) {
        return -1;
    }

    if (input_gradient != 0) {
        (void)memset(input_gradient, 0, infer_ctx->graph_input_size * sizeof(float));
    }

    /* Update the lightweight graph-mode projection path and accumulate dL/dX. */
    for (output_index = 0U; output_index < infer_ctx->graph_output_size; ++output_index) {
        float dz = output_gradient[output_index] * (1.0f - output_cache[output_index] * output_cache[output_index]);
        size_t input_index;

        transformer_apply_gradient(
            &infer_ctx->classifier_bias[output_index],
            dz,
            train_ctx->learning_rate
        );

        for (input_index = 0U; input_index < infer_ctx->graph_input_size; ++input_index) {
            float old_weight = infer_ctx->output_weight[input_index][output_index];
            if (input_gradient != 0) {
                input_gradient[input_index] += dz * old_weight;
                if (input_index == output_index) {
                    input_gradient[input_index] += dz;
                }
            }
            transformer_apply_gradient(
                &infer_ctx->output_weight[input_index][output_index],
                dz * input[input_index],
                train_ctx->learning_rate
            );
        }
    }

    train_ctx->total_steps += 1U;
    return 0;
}
