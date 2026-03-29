/**
 * @file transformer_infer_ops.c
 * @brief Dynamically sized tiny transformer inference backend.
 */

#include "transformer_infer_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TRANSFORMER_ABI_VERSION 3U

typedef struct {
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t abi_version;
    uint64_t vocab_size;
    uint64_t max_seq_length;
    uint64_t model_dim;
    uint64_t max_response_classes;
    uint64_t max_text_length;
    uint64_t class_count;
    uint64_t graph_input_size;
    uint64_t graph_output_size;
    uint32_t rng_state;
} TransformerWeightHeader;

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
} TransformerForwardCache;

static size_t transformer_index(size_t row, size_t column, size_t column_count) {
    return (row * column_count) + column;
}

static char* transformer_class_text_mut(TransformerInferContext* context, size_t class_index) {
    return context->class_texts + (class_index * context->max_text_length);
}

static const char* transformer_class_text_view(
    const TransformerInferContext* context,
    size_t class_index
) {
    return context->class_texts + (class_index * context->max_text_length);
}

static void transformer_copy_text(char* destination, size_t capacity, const char* source) {
    size_t copy_length;

    if (destination == 0 || capacity == 0U) {
        return;
    }
    if (source == 0) {
        destination[0] = '\0';
        return;
    }

    copy_length = strlen(source);
    if (copy_length >= capacity) {
        copy_length = capacity - 1U;
    }
    (void)memcpy(destination, source, copy_length);
    destination[copy_length] = '\0';
}

static uint32_t transformer_next_random(uint32_t* state) {
    uint32_t value = *state;
    value = value * 1664525U + 1013904223U;
    *state = value;
    return value;
}

static float transformer_random_weight(uint32_t* state, float scale) {
    float normalized = (float)(transformer_next_random(state) & 0xFFFFU) / 65535.0f;
    return (normalized - 0.5f) * scale;
}

static void transformer_release_parameters(TransformerInferContext* context) {
    if (context == 0) {
        return;
    }

    free(context->token_embedding);
    free(context->position_embedding);
    free(context->query_weight);
    free(context->key_weight);
    free(context->value_weight);
    free(context->output_weight);
    free(context->classifier_weight);
    free(context->classifier_bias);
    free(context->class_texts);
    free(context->fallback_answer);
    free(context->graph_projection_weight);
    free(context->graph_projection_bias);

    context->token_embedding = 0;
    context->position_embedding = 0;
    context->query_weight = 0;
    context->key_weight = 0;
    context->value_weight = 0;
    context->output_weight = 0;
    context->classifier_weight = 0;
    context->classifier_bias = 0;
    context->class_texts = 0;
    context->fallback_answer = 0;
    context->graph_projection_weight = 0;
    context->graph_projection_bias = 0;
}

void nn_transformer_infer_destroy(void* ctx) {
    TransformerInferContext* context = (TransformerInferContext*)ctx;

    if (context == 0) {
        return;
    }
    transformer_release_parameters(context);
    free(context);
}

static int transformer_forward_cache_init(
    TransformerForwardCache* cache,
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

static void transformer_forward_cache_destroy(TransformerForwardCache* cache) {
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

size_t nn_transformer_tokenize_text(
    const char* text,
    size_t* out_tokens,
    size_t token_capacity,
    size_t vocab_size
) {
    size_t length = 0U;

    if (text == 0 || out_tokens == 0 || token_capacity == 0U || vocab_size < 2U) {
        return 0U;
    }

    while (*text != '\0' && length < token_capacity) {
        unsigned char raw_value = (unsigned char)(*text);
        out_tokens[length] = 1U + ((size_t)raw_value % (vocab_size - 1U));
        length += 1U;
        text += 1;
    }

    if (length == 0U) {
        out_tokens[0] = 1U;
        return 1U;
    }

    return length;
}

int nn_transformer_find_class(
    const TransformerInferContext* context,
    const char* answer
) {
    size_t class_index;

    if (context == 0 || answer == 0) {
        return -1;
    }

    for (class_index = 0U; class_index < context->class_count; ++class_index) {
        if (strcmp(transformer_class_text_view(context, class_index), answer) == 0) {
            return (int)class_index;
        }
    }

    return -1;
}

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
    if (context->class_count >= context->max_response_classes) {
        return -1;
    }

    transformer_copy_text(
        transformer_class_text_mut(context, context->class_count),
        context->max_text_length,
        answer
    );
    context->class_count += 1U;
    return (int)(context->class_count - 1U);
}

int nn_transformer_init_parameters(
    TransformerInferContext* context,
    const TransformerModelConfig* config,
    size_t graph_input_size,
    size_t graph_output_size
) {
    size_t token_index;
    size_t position_index;
    size_t row;
    size_t column;

    if (context == 0 || config == 0) {
        return -1;
    }
    if (config->vocab_size < 2U || config->model_dim == 0U || config->max_seq_length == 0U ||
        config->max_response_classes == 0U || config->max_text_length == 0U ||
        graph_input_size == 0U || graph_output_size == 0U) {
        return -1;
    }

    transformer_release_parameters(context);
    context->vocab_size = config->vocab_size;
    context->max_seq_length = config->max_seq_length;
    context->model_dim = config->model_dim;
    context->max_response_classes = config->max_response_classes;
    context->max_text_length = config->max_text_length;
    context->graph_input_size = graph_input_size;
    context->graph_output_size = graph_output_size;
    context->class_count = 0U;
    context->rng_state = config->seed == 0U ? 1U : config->seed;

    context->token_embedding = (float*)calloc(context->vocab_size * context->model_dim, sizeof(float));
    context->position_embedding = (float*)calloc(context->max_seq_length * context->model_dim, sizeof(float));
    context->query_weight = (float*)calloc(context->model_dim * context->model_dim, sizeof(float));
    context->key_weight = (float*)calloc(context->model_dim * context->model_dim, sizeof(float));
    context->value_weight = (float*)calloc(context->model_dim * context->model_dim, sizeof(float));
    context->output_weight = (float*)calloc(context->model_dim * context->model_dim, sizeof(float));
    context->classifier_weight = (float*)calloc(
        context->max_response_classes * context->model_dim,
        sizeof(float)
    );
    context->classifier_bias = (float*)calloc(context->max_response_classes, sizeof(float));
    context->class_texts = (char*)calloc(
        context->max_response_classes * context->max_text_length,
        sizeof(char)
    );
    context->fallback_answer = (char*)calloc(context->max_text_length, sizeof(char));
    context->graph_projection_weight = (float*)calloc(
        context->graph_input_size * context->graph_output_size,
        sizeof(float)
    );
    context->graph_projection_bias = (float*)calloc(context->graph_output_size, sizeof(float));
    if (context->token_embedding == 0 || context->position_embedding == 0 ||
        context->query_weight == 0 || context->key_weight == 0 ||
        context->value_weight == 0 || context->output_weight == 0 ||
        context->classifier_weight == 0 || context->classifier_bias == 0 ||
        context->class_texts == 0 || context->fallback_answer == 0 ||
        context->graph_projection_weight == 0 || context->graph_projection_bias == 0) {
        transformer_release_parameters(context);
        return -1;
    }

    transformer_copy_text(
        context->fallback_answer,
        context->max_text_length,
        "No trained response is available."
    );

    for (token_index = 0U; token_index < context->vocab_size; ++token_index) {
        for (column = 0U; column < context->model_dim; ++column) {
            float phase = (float)((token_index + 1U) * (column + 1U));
            context->token_embedding[transformer_index(token_index, column, context->model_dim)] =
                0.35f * sinf(phase * 0.173f) + 0.25f * cosf(phase * 0.117f);
        }
    }

    for (position_index = 0U; position_index < context->max_seq_length; ++position_index) {
        for (column = 0U; column < context->model_dim; ++column) {
            float phase = (float)((position_index + 1U) * (column + 1U));
            context->position_embedding[
                transformer_index(position_index, column, context->model_dim)
            ] = 0.10f * sinf(phase * 0.071f) + 0.08f * cosf(phase * 0.049f);
        }
    }

    for (row = 0U; row < context->model_dim; ++row) {
        for (column = 0U; column < context->model_dim; ++column) {
            float diagonal = row == column ? 1.0f : 0.0f;
            float jitter = transformer_random_weight(&context->rng_state, 0.02f);
            size_t index = transformer_index(row, column, context->model_dim);
            context->query_weight[index] = diagonal + jitter;
            context->key_weight[index] = diagonal + jitter;
            context->value_weight[index] = diagonal + jitter;
            context->output_weight[index] = 0.35f * diagonal + jitter;
        }
    }

    for (row = 0U; row < context->max_response_classes; ++row) {
        for (column = 0U; column < context->model_dim; ++column) {
            context->classifier_weight[transformer_index(row, column, context->model_dim)] =
                transformer_random_weight(&context->rng_state, 0.01f);
        }
    }

    for (row = 0U; row < context->graph_input_size; ++row) {
        for (column = 0U; column < context->graph_output_size; ++column) {
            float diagonal = row == column ? 0.25f : 0.0f;
            context->graph_projection_weight[
                transformer_index(row, column, context->graph_output_size)
            ] = diagonal + transformer_random_weight(&context->rng_state, 0.015f);
        }
    }

    return 0;
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

static int transformer_run_forward(
    const TransformerInferContext* context,
    const char* question,
    TransformerForwardCache* cache
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
    if (transformer_forward_cache_init(&cache, context) != 0) {
        return -1;
    }
    if (transformer_run_forward(context, question, &cache) != 0) {
        transformer_forward_cache_destroy(&cache);
        return -1;
    }

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

    transformer_forward_cache_destroy(&cache);
    return (int)best_index;
}

int nn_transformer_graph_run(void* context, const void* input, void* output) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;
    const float* input_values = (const float*)input;
    float* output_values = (float*)output;
    size_t output_index;

    if (infer_ctx == 0 || input_values == 0 || output_values == 0) {
        return -1;
    }

    for (output_index = 0U; output_index < infer_ctx->graph_output_size; ++output_index) {
        float sum = infer_ctx->graph_projection_bias[output_index];
        size_t input_index;

        for (input_index = 0U; input_index < infer_ctx->graph_input_size; ++input_index) {
            sum += input_values[input_index] * infer_ctx->graph_projection_weight[
                transformer_index(input_index, output_index, infer_ctx->graph_output_size)
            ];
        }
        if (output_index < infer_ctx->graph_input_size) {
            sum += input_values[output_index];
        }
        output_values[output_index] = tanhf(sum);
    }

    return 0;
}

int nn_transformer_infer_step(void* context) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;
    int class_index;

    if (infer_ctx == 0 || infer_ctx->question == 0 ||
        infer_ctx->answer == 0 || infer_ctx->answer_capacity == 0U) {
        return -1;
    }

    class_index = nn_transformer_predict_class(infer_ctx, infer_ctx->question, 0, 0U, 0);
    if (class_index < 0) {
        (void)snprintf(infer_ctx->answer, infer_ctx->answer_capacity, "%s", infer_ctx->fallback_answer);
        return 0;
    }

    (void)snprintf(
        infer_ctx->answer,
        infer_ctx->answer_capacity,
        "%s",
        transformer_class_text_view(infer_ctx, (size_t)class_index)
    );
    return 0;
}

static int transformer_transfer_block(FILE* fp, void* values, size_t size, int write_mode) {
    size_t transferred;

    if (fp == 0 || values == 0) {
        return 0;
    }

    transferred = write_mode ? fwrite(values, 1U, size, fp) : fread(values, 1U, size, fp);
    return transferred == size ? 1 : 0;
}

int nn_transformer_load_weights(void* context, FILE* fp) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;
    TransformerWeightHeader header;

    if (infer_ctx == 0 || fp == 0) {
        return 0;
    }
    if (fread(&header, sizeof(header), 1U, fp) != 1U) {
        return 0;
    }
    if (header.abi_version != TRANSFORMER_ABI_VERSION) {
        return 0;
    }
    if (infer_ctx->expected_network_hash != 0U &&
        header.network_hash != infer_ctx->expected_network_hash) {
        return 0;
    }
    if (infer_ctx->expected_layout_hash != 0U &&
        header.layout_hash != infer_ctx->expected_layout_hash) {
        return 0;
    }
    if (header.vocab_size != infer_ctx->vocab_size ||
        header.max_seq_length != infer_ctx->max_seq_length ||
        header.model_dim != infer_ctx->model_dim ||
        header.max_response_classes != infer_ctx->max_response_classes ||
        header.max_text_length != infer_ctx->max_text_length ||
        header.class_count > infer_ctx->max_response_classes ||
        header.graph_input_size != infer_ctx->graph_input_size ||
        header.graph_output_size != infer_ctx->graph_output_size) {
        return 0;
    }

    infer_ctx->class_count = header.class_count;
    infer_ctx->rng_state = header.rng_state;

    return transformer_transfer_block(
            fp,
            infer_ctx->token_embedding,
            infer_ctx->vocab_size * infer_ctx->model_dim * sizeof(float),
            0
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->position_embedding,
            infer_ctx->max_seq_length * infer_ctx->model_dim * sizeof(float),
            0
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->query_weight,
            infer_ctx->model_dim * infer_ctx->model_dim * sizeof(float),
            0
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->key_weight,
            infer_ctx->model_dim * infer_ctx->model_dim * sizeof(float),
            0
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->value_weight,
            infer_ctx->model_dim * infer_ctx->model_dim * sizeof(float),
            0
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->output_weight,
            infer_ctx->model_dim * infer_ctx->model_dim * sizeof(float),
            0
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->classifier_weight,
            infer_ctx->max_response_classes * infer_ctx->model_dim * sizeof(float),
            0
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->classifier_bias,
            infer_ctx->max_response_classes * sizeof(float),
            0
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->class_texts,
            infer_ctx->max_response_classes * infer_ctx->max_text_length * sizeof(char),
            0
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->fallback_answer,
            infer_ctx->max_text_length * sizeof(char),
            0
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->graph_projection_weight,
            infer_ctx->graph_input_size * infer_ctx->graph_output_size * sizeof(float),
            0
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->graph_projection_bias,
            infer_ctx->graph_output_size * sizeof(float),
            0
        );
}

int nn_transformer_save_weights(void* context, FILE* fp) {
    TransformerInferContext* infer_ctx = (TransformerInferContext*)context;
    TransformerWeightHeader header;

    if (infer_ctx == 0 || fp == 0) {
        return 0;
    }

    (void)memset(&header, 0, sizeof(header));
    header.network_hash = infer_ctx->expected_network_hash;
    header.layout_hash = infer_ctx->expected_layout_hash;
    header.abi_version = TRANSFORMER_ABI_VERSION;
    header.vocab_size = (uint64_t)infer_ctx->vocab_size;
    header.max_seq_length = (uint64_t)infer_ctx->max_seq_length;
    header.model_dim = (uint64_t)infer_ctx->model_dim;
    header.max_response_classes = (uint64_t)infer_ctx->max_response_classes;
    header.max_text_length = (uint64_t)infer_ctx->max_text_length;
    header.class_count = (uint64_t)infer_ctx->class_count;
    header.graph_input_size = (uint64_t)infer_ctx->graph_input_size;
    header.graph_output_size = (uint64_t)infer_ctx->graph_output_size;
    header.rng_state = infer_ctx->rng_state;

    if (fwrite(&header, sizeof(header), 1U, fp) != 1U) {
        return 0;
    }

    return transformer_transfer_block(
            fp,
            infer_ctx->token_embedding,
            infer_ctx->vocab_size * infer_ctx->model_dim * sizeof(float),
            1
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->position_embedding,
            infer_ctx->max_seq_length * infer_ctx->model_dim * sizeof(float),
            1
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->query_weight,
            infer_ctx->model_dim * infer_ctx->model_dim * sizeof(float),
            1
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->key_weight,
            infer_ctx->model_dim * infer_ctx->model_dim * sizeof(float),
            1
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->value_weight,
            infer_ctx->model_dim * infer_ctx->model_dim * sizeof(float),
            1
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->output_weight,
            infer_ctx->model_dim * infer_ctx->model_dim * sizeof(float),
            1
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->classifier_weight,
            infer_ctx->max_response_classes * infer_ctx->model_dim * sizeof(float),
            1
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->classifier_bias,
            infer_ctx->max_response_classes * sizeof(float),
            1
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->class_texts,
            infer_ctx->max_response_classes * infer_ctx->max_text_length * sizeof(char),
            1
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->fallback_answer,
            infer_ctx->max_text_length * sizeof(char),
            1
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->graph_projection_weight,
            infer_ctx->graph_input_size * infer_ctx->graph_output_size * sizeof(float),
            1
        ) &&
        transformer_transfer_block(
            fp,
            infer_ctx->graph_projection_bias,
            infer_ctx->graph_output_size * sizeof(float),
            1
        );
}
