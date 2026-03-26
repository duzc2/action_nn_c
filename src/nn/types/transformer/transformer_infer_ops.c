/**
 * @file transformer_infer_ops.c
 * @brief Tiny transformer-style inference backend used by generated code.
 *
 * The implementation intentionally focuses on a small, inspectable model for
 * demo scenarios. It still demonstrates the same lifecycle expected from any
 * backend integrated into the profiler pipeline: typed configuration, parameter
 * initialization, graph execution, text inference, and guarded persistence.
 */

#include "transformer_infer_ops.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define TRANSFORMER_ABI_VERSION 1U

/**
 * @section transformer_infer_design Tiny transformer inference notes
 *
 * This backend intentionally trades completeness for transparency. It is small
 * enough for demo scenarios, yet it still demonstrates the full profiler flow:
 * typed config reconstruction, deterministic parameter initialization, token
 * handling, attention-style scoring, class prediction, and hash-guarded weight
 * persistence. The comments in this file therefore focus on clarifying the
 * design intent of each stage rather than pretending the model is a production
 * transformer.
 */

/**
 * @brief Serialized weight blob written to disk by the tiny transformer backend.
 *
 * The blob intentionally contains both compatibility metadata and all parameter
 * arrays so save/load stay self-contained and deterministic.
 */
typedef struct {
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t abi_version;
    uint32_t vocab_size;
    uint32_t max_seq_length;
    uint32_t model_dim;
    uint32_t class_count;
    uint32_t rng_state;
    float token_embedding[TRANSFORMER_VOCAB_SIZE][TRANSFORMER_MAX_MODEL_DIM];
    float position_embedding[TRANSFORMER_MAX_SEQ_LENGTH][TRANSFORMER_MAX_MODEL_DIM];
    float query_weight[TRANSFORMER_MAX_MODEL_DIM][TRANSFORMER_MAX_MODEL_DIM];
    float key_weight[TRANSFORMER_MAX_MODEL_DIM][TRANSFORMER_MAX_MODEL_DIM];
    float value_weight[TRANSFORMER_MAX_MODEL_DIM][TRANSFORMER_MAX_MODEL_DIM];
    float output_weight[TRANSFORMER_MAX_MODEL_DIM][TRANSFORMER_MAX_MODEL_DIM];
    float classifier_weight[TRANSFORMER_MAX_RESPONSE_CLASSES][TRANSFORMER_MAX_MODEL_DIM];
    float classifier_bias[TRANSFORMER_MAX_RESPONSE_CLASSES];
    char class_texts[TRANSFORMER_MAX_RESPONSE_CLASSES][TRANSFORMER_MAX_TEXT_LENGTH];
    char fallback_answer[TRANSFORMER_MAX_TEXT_LENGTH];
} TransformerWeightBlob;

/**
 * @brief Scratch buffers produced by one forward pass.
 *
 * Keeping them in a dedicated stack object makes the forward path easier to
 * read and lets prediction code expose probabilities without mutating the
 * long-lived inference context.
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
} TransformerForwardCache;

/**
 * @brief Copy one NUL-terminated string into a bounded destination buffer.
 */
static void transformer_copy_text(char* destination, size_t capacity, const char* source) {
    if (destination == 0 || capacity == 0U) {
        return;
    }
    if (source == 0) {
        destination[0] = '\0';
        return;
    }

    (void)strncpy(destination, source, capacity - 1U);
    destination[capacity - 1U] = '\0';
}

/**
 * @brief Advance the tiny deterministic RNG used by the demo transformer.
 */
static uint32_t transformer_next_random(uint32_t* state) {
    uint32_t value = *state;
    value = value * 1664525U + 1013904223U;
    *state = value;
    return value;
}

/**
 * @brief Produce a small centered random weight for parameter initialization.
 */
static float transformer_random_weight(uint32_t* state, float scale) {
    uint32_t raw = transformer_next_random(state);
    float normalized = (float)(raw & 0xFFFFU) / 65535.0f;
    return (normalized - 0.5f) * scale;
}

/**
 * @brief Map a single character into the fixed demo vocabulary.
 */
static uint8_t transformer_char_to_token(char ch) {
    unsigned char lower = (unsigned char)tolower((unsigned char)ch);

    if (lower >= 'a' && lower <= 'z') {
        return (uint8_t)(1U + (lower - 'a'));
    }
    if (lower >= '0' && lower <= '9') {
        return (uint8_t)(27U + (lower - '0'));
    }
    if (lower == ' ') return 37U;
    if (lower == '\'') return 38U;
    if (lower == '.') return 39U;
    if (lower == ',') return 40U;
    if (lower == '?') return 41U;
    if (lower == '!') return 42U;
    if (lower == '-') return 43U;

    return 0U;
}

/**
 * @brief Tokenize a short text string into the fixed demo vocabulary.
 */
size_t nn_transformer_tokenize_text(
    const char* text,
    uint8_t* out_tokens,
    size_t token_capacity
) {
    size_t length = 0U;

    if (text == 0 || out_tokens == 0 || token_capacity == 0U) {
        return 0U;
    }

    while (*text != '\0' && length < token_capacity) {
        uint8_t token = transformer_char_to_token(*text);
        if (token != 0U) {
            out_tokens[length] = token;
            length += 1U;
        }
        text += 1;
    }

    if (length == 0U) {
        out_tokens[0] = 37U;
        return 1U;
    }

    return length;
}

/**
 * @brief Find an answer-class index by exact stored text match.
 */
int nn_transformer_find_class(
    const TransformerInferContext* context,
    const char* answer
) {
    size_t class_index;

    if (context == 0 || answer == 0) {
        return -1;
    }

    for (class_index = 0U; class_index < context->class_count; ++class_index) {
        if (strcmp(context->class_texts[class_index], answer) == 0) {
            return (int)class_index;
        }
    }

    return -1;
}

/**
 * @brief Reuse an existing answer class or allocate a new one lazily.
 */
int nn_transformer_find_or_add_class(
    TransformerInferContext* context,
    const char* answer
) {
    int existing_index;

    if (context == 0 || answer == 0 || answer[0] == '\0') {
        return -1;
    }

    existing_index = nn_transformer_find_class(context, answer);
    if (existing_index >= 0) {
        return existing_index;
    }
    if (context->class_count >= TRANSFORMER_MAX_RESPONSE_CLASSES) {
        return -1;
    }

    transformer_copy_text(
        context->class_texts[context->class_count],
        sizeof(context->class_texts[context->class_count]),
        answer
    );
    context->class_count += 1U;
    return (int)(context->class_count - 1U);
}

/**
 * @brief Initialize embeddings, attention matrices, and classifier weights.
 */
int nn_transformer_init_parameters(
    TransformerInferContext* context,
    size_t model_dim,
    uint32_t seed
) {
    size_t token_index;
    size_t position_index;
    size_t row;
    size_t column;
    float token_phase;
    float position_phase;

    if (context == 0) {
        return -1;
    }
    if (model_dim == 0U || model_dim > TRANSFORMER_MAX_MODEL_DIM) {
        return -1;
    }

    /* Reset the entire context so every later field starts from a known value. */
    (void)memset(context, 0, sizeof(*context));
    context->vocab_size = TRANSFORMER_VOCAB_SIZE;
    context->max_seq_length = TRANSFORMER_MAX_SEQ_LENGTH;
    context->model_dim = model_dim;
    context->rng_state = seed == 0U ? 1U : seed;
    transformer_copy_text(
        context->fallback_answer,
        sizeof(context->fallback_answer),
        "I can talk with simple English."
    );

    /* Build deterministic token embeddings for the fixed demo vocabulary. */
    for (token_index = 0U; token_index < TRANSFORMER_VOCAB_SIZE; ++token_index) {
        for (column = 0U; column < model_dim; ++column) {
            token_phase = (float)((token_index + 1U) * (column + 1U));
            context->token_embedding[token_index][column] =
                0.35f * sinf(token_phase * 0.173f) +
                0.25f * cosf(token_phase * 0.117f);
        }
    }

    /* Add simple position encodings so sequence order affects the forward pass. */
    for (position_index = 0U; position_index < TRANSFORMER_MAX_SEQ_LENGTH; ++position_index) {
        for (column = 0U; column < model_dim; ++column) {
            position_phase = (float)((position_index + 1U) * (column + 1U));
            context->position_embedding[position_index][column] =
                0.10f * sinf(position_phase * 0.071f) +
                0.08f * cosf(position_phase * 0.049f);
        }
    }

    /* Start projection matrices close to identity so early behaviour is stable. */
    for (row = 0U; row < model_dim; ++row) {
        for (column = 0U; column < model_dim; ++column) {
            float jitter = transformer_random_weight(&context->rng_state, 0.02f);
            float diagonal = row == column ? 1.0f : 0.0f;

            context->query_weight[row][column] = diagonal + jitter;
            context->key_weight[row][column] = diagonal + jitter;
            context->value_weight[row][column] = diagonal + jitter;
            context->output_weight[row][column] = 0.35f * diagonal + jitter;
        }
    }

    for (row = 0U; row < TRANSFORMER_MAX_RESPONSE_CLASSES; ++row) {
        for (column = 0U; column < model_dim; ++column) {
            context->classifier_weight[row][column] =
                transformer_random_weight(&context->rng_state, 0.01f);
        }
    }

    return 0;
}

/**
 * @brief Normalize a score vector in-place with a numerically stable softmax.
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
 * @brief Compute the Euclidean norm of a dense vector.
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
 * @brief Core inference stage: embed tokens, score classes, and emit logits.
 *
 * The helper intentionally keeps every intermediate tensor in a stack cache so
 * prediction, debug inspection, and training can all share the same forward
 * semantics without mutating long-lived model state.
 */
static void transformer_run_forward(
    const TransformerInferContext* context,
    const char* question,
    TransformerForwardCache* cache
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

    /* Stage 2: combine token and position embeddings into input states. */
    /* Stage 3: project each token into query/key/value spaces. */
    /* Stage 5: compute normalized token-to-token attention weights. */
    /* Stage 6: apply attention, output projection, tanh, and mean pooling. */
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

    /* Stage 4: apply the usual sqrt(d) scaling before softmax attention. */
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

    /* Stage 7: compare the pooled sentence representation with each class. */
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

    if (context->class_count > 0U) {
        transformer_softmax(cache->probabilities, context->class_count);
    }
}

/**
 * @brief Public API stage: classify one question into a learned answer bucket.
 *
 * Prediction is intentionally separated from answer-string formatting so the
 * training path can reuse the same probability outputs when computing loss. The
 * caller can therefore treat this function as the canonical class-selection
 * step for both interactive inference and supervised updates.
 */
int nn_transformer_predict_class(
    const TransformerInferContext* context,
    const char* question,
    float* out_probabilities,
    size_t probability_capacity,
    float* out_loss_hint
) {
    TransformerForwardCache cache;
    size_t class_index;
    size_t best_index = 0U;

    if (context == 0 || question == 0) {
        return -1;
    }
    if (context->class_count == 0U) {
        if (out_loss_hint != 0) {
            *out_loss_hint = 1.0f;
        }
        return -1;
    }

    /* Run the full forward pipeline once, then reuse its cached probabilities. */
    transformer_run_forward(context, question, &cache);

    for (class_index = 1U; class_index < context->class_count; ++class_index) {
        if (cache.probabilities[class_index] > cache.probabilities[best_index]) {
            best_index = class_index;
        }
    }

    if (out_probabilities != 0) {
        size_t copy_count = probability_capacity < context->class_count ?
            probability_capacity : context->class_count;
        for (class_index = 0U; class_index < copy_count; ++class_index) {
            out_probabilities[class_index] = cache.probabilities[class_index];
        }
    }
    if (out_loss_hint != 0) {
        *out_loss_hint = 1.0f - cache.probabilities[best_index];
    }

    return (int)best_index;
}

/**
 * @brief Adapt raw graph buffers to the transformer class-prediction helper.
 */
int nn_transformer_graph_run(void* context, const void* input, void* output) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;
    const float* input_values = (const float*)input;
    float* output_values = (float*)output;
    size_t output_index;

    if (infer_ctx == 0 || input_values == 0 || output_values == 0) {
        return -1;
    }
    if (infer_ctx->graph_input_size == 0U ||
        infer_ctx->graph_output_size == 0U ||
        infer_ctx->graph_input_size > infer_ctx->model_dim ||
        infer_ctx->graph_output_size > infer_ctx->model_dim) {
        return -1;
    }

    for (output_index = 0U; output_index < infer_ctx->graph_output_size; ++output_index) {
        float sum = infer_ctx->classifier_bias[output_index];
        size_t input_index;

        for (input_index = 0U; input_index < infer_ctx->graph_input_size; ++input_index) {
            sum += input_values[input_index] * infer_ctx->output_weight[input_index][output_index];
        }
        if (output_index < infer_ctx->graph_input_size) {
            sum += input_values[output_index];
        }
        output_values[output_index] = tanhf(sum);
    }

    return 0;
}

/**
 * @brief Public API stage: turn the predicted class back into an answer string.
 *
 * The generated runtime passes question and answer buffers through the context,
 * so this helper only needs to choose the best class and copy the final text.
 */
int nn_transformer_infer_step(void* context) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;
    int class_index;

    if (infer_ctx == 0 || infer_ctx->question == 0 ||
        infer_ctx->answer == 0 || infer_ctx->answer_capacity == 0U) {
        return -1;
    }

    class_index = nn_transformer_predict_class(
        infer_ctx,
        infer_ctx->question,
        0,
        0U,
        0
    );
    /* Fall back to a stable English sentence instead of leaving garbage output. */
    if (class_index < 0) {
        (void)snprintf(
            infer_ctx->answer,
            infer_ctx->answer_capacity,
            "%s",
            infer_ctx->fallback_answer
        );
        return 0;
    }

    (void)snprintf(
        infer_ctx->answer,
        infer_ctx->answer_capacity,
        "%s",
        infer_ctx->class_texts[class_index]
    );
    return 0;
}

/**
 * @brief Load transformer parameters after validating hash and ABI metadata.
 *
 * The loader rejects mismatched topology or layout before mutating the runtime
 * context so callers never observe partially loaded model state.
 */
int nn_transformer_load_weights(void* context, FILE* fp) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;
    TransformerWeightBlob blob;

    if (infer_ctx == 0 || fp == 0) {
        return 0;
    }
    if (fread(&blob, sizeof(blob), 1, fp) != 1) {
        return 0;
    }
    if (blob.abi_version != TRANSFORMER_ABI_VERSION) {
        return 0;
    }
    if (infer_ctx->expected_network_hash != 0U &&
        blob.network_hash != infer_ctx->expected_network_hash) {
        return 0;
    }
    if (infer_ctx->expected_layout_hash != 0U &&
        blob.layout_hash != infer_ctx->expected_layout_hash) {
        return 0;
    }
    if (blob.vocab_size != TRANSFORMER_VOCAB_SIZE ||
        blob.max_seq_length > TRANSFORMER_MAX_SEQ_LENGTH ||
        blob.model_dim == 0U ||
        blob.model_dim > TRANSFORMER_MAX_MODEL_DIM ||
        blob.class_count > TRANSFORMER_MAX_RESPONSE_CLASSES) {
        return 0;
    }

    /* Copy validated metadata first so later calls see a consistent context. */
    infer_ctx->vocab_size = blob.vocab_size;
    infer_ctx->max_seq_length = blob.max_seq_length;
    infer_ctx->model_dim = blob.model_dim;
    infer_ctx->class_count = blob.class_count;
    infer_ctx->rng_state = blob.rng_state;
    (void)memcpy(infer_ctx->token_embedding, blob.token_embedding, sizeof(blob.token_embedding));
    (void)memcpy(infer_ctx->position_embedding, blob.position_embedding, sizeof(blob.position_embedding));
    (void)memcpy(infer_ctx->query_weight, blob.query_weight, sizeof(blob.query_weight));
    (void)memcpy(infer_ctx->key_weight, blob.key_weight, sizeof(blob.key_weight));
    (void)memcpy(infer_ctx->value_weight, blob.value_weight, sizeof(blob.value_weight));
    (void)memcpy(infer_ctx->output_weight, blob.output_weight, sizeof(blob.output_weight));
    (void)memcpy(infer_ctx->classifier_weight, blob.classifier_weight, sizeof(blob.classifier_weight));
    (void)memcpy(infer_ctx->classifier_bias, blob.classifier_bias, sizeof(blob.classifier_bias));
    (void)memcpy(infer_ctx->class_texts, blob.class_texts, sizeof(blob.class_texts));
    (void)memcpy(infer_ctx->fallback_answer, blob.fallback_answer, sizeof(blob.fallback_answer));

    return 1;
}

/**
 * @brief Save transformer parameters together with compatibility metadata.
 *
 * Persistence uses one flat blob because the demo backend has a fixed upper
 * bound on every tensor size and benefits from deterministic binary layout.
 */
int nn_transformer_save_weights(void* context, FILE* fp) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;
    TransformerWeightBlob blob;

    if (infer_ctx == 0 || fp == 0) {
        return 0;
    }

    /* Zero-fill the blob so every serialized byte is deterministic. */
    (void)memset(&blob, 0, sizeof(blob));
    blob.network_hash = infer_ctx->expected_network_hash;
    blob.layout_hash = infer_ctx->expected_layout_hash;
    blob.abi_version = TRANSFORMER_ABI_VERSION;
    blob.vocab_size = (uint32_t)infer_ctx->vocab_size;
    blob.max_seq_length = (uint32_t)infer_ctx->max_seq_length;
    blob.model_dim = (uint32_t)infer_ctx->model_dim;
    blob.class_count = (uint32_t)infer_ctx->class_count;
    blob.rng_state = infer_ctx->rng_state;
    (void)memcpy(blob.token_embedding, infer_ctx->token_embedding, sizeof(blob.token_embedding));
    (void)memcpy(blob.position_embedding, infer_ctx->position_embedding, sizeof(blob.position_embedding));
    (void)memcpy(blob.query_weight, infer_ctx->query_weight, sizeof(blob.query_weight));
    (void)memcpy(blob.key_weight, infer_ctx->key_weight, sizeof(blob.key_weight));
    (void)memcpy(blob.value_weight, infer_ctx->value_weight, sizeof(blob.value_weight));
    (void)memcpy(blob.output_weight, infer_ctx->output_weight, sizeof(blob.output_weight));
    (void)memcpy(blob.classifier_weight, infer_ctx->classifier_weight, sizeof(blob.classifier_weight));
    (void)memcpy(blob.classifier_bias, infer_ctx->classifier_bias, sizeof(blob.classifier_bias));
    (void)memcpy(blob.class_texts, infer_ctx->class_texts, sizeof(blob.class_texts));
    (void)memcpy(blob.fallback_answer, infer_ctx->fallback_answer, sizeof(blob.fallback_answer));

    return fwrite(&blob, sizeof(blob), 1, fp) == 1 ? 1 : 0;
}
