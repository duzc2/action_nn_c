/**
 * @file rnn_infer_ops.c
 * @brief Tiny RNN inference backend used by generated graph code.
 */

#include "rnn_infer_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define RNN_ABI_VERSION 1U

/**
 * @brief Serialized header written before the flat RNN parameter arrays.
 */
typedef struct {
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t abi_version;
    uint32_t sequence_length;
    uint32_t input_feature_size;
    uint32_t hidden_size;
    uint32_t output_size;
    uint32_t hidden_activation;
    uint32_t output_activation;
} RnnWeightHeader;

/**
 * @brief Advance the tiny deterministic RNG used for parameter init.
 */
static uint32_t rnn_next_random(uint32_t* state) {
    uint32_t value = *state;
    value = value * 1664525U + 1013904223U;
    *state = value;
    return value;
}

/**
 * @brief Generate a small centered random weight.
 */
static float rnn_random_weight(uint32_t* state, float scale) {
    float normalized = (float)(rnn_next_random(state) & 0xFFFFU) / 65535.0f;
    return (normalized - 0.5f) * scale;
}

/**
 * @brief Apply the activation configured for one stage.
 */
static float rnn_apply_activation(float value, RnnActivationType activation) {
    switch (activation) {
        case RNN_ACT_TANH:
            return tanhf(value);
        case RNN_ACT_NONE:
        default:
            return value;
    }
}

/**
 * @brief Validate that the type config stays within the backend's fixed bounds.
 */
static int rnn_config_is_valid(const RnnConfig* config) {
    if (config == NULL) {
        return 0;
    }
    if (config->sequence_length == 0U) {
        return 0;
    }
    if (config->input_feature_size == 0U) {
        return 0;
    }
    if (config->hidden_size == 0U) {
        return 0;
    }
    if (config->output_size == 0U) {
        return 0;
    }
    return 1;
}

/**
 * @brief Compute a compact structural hash from the active RNN config.
 */
static uint64_t rnn_compute_layout_hash(const RnnConfig* config) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    const uint64_t prime = 0x100000001b3ULL;

    if (config == NULL) {
        return hash;
    }

    hash ^= (uint64_t)config->sequence_length;
    hash *= prime;
    hash ^= (uint64_t)config->input_feature_size;
    hash *= prime;
    hash ^= (uint64_t)config->hidden_size;
    hash *= prime;
    hash ^= (uint64_t)config->output_size;
    hash *= prime;
    hash ^= (uint64_t)config->hidden_activation;
    hash *= prime;
    hash ^= (uint64_t)config->output_activation;
    hash *= prime;
    return hash;
}

/**
 * @brief Deliberately reject implicit construction without typed config metadata.
 */
RnnInferContext* nn_rnn_infer_create(void) {
    return NULL;
}

/**
 * @brief Allocate and initialize one typed RNN inference context.
 */
RnnInferContext* nn_rnn_infer_create_with_config(const RnnConfig* config, uint32_t seed) {
    RnnInferContext* context;
    size_t input_to_hidden_count;
    size_t hidden_to_hidden_count;
    size_t hidden_to_output_count;
    size_t value_index;

    if (!rnn_config_is_valid(config)) {
        return NULL;
    }

    context = (RnnInferContext*)calloc(1U, sizeof(RnnInferContext));
    if (context == NULL) {
        return NULL;
    }

    context->config = *config;
    context->rng_state = seed != 0U ? seed : (config->seed != 0U ? config->seed : 1U);
    input_to_hidden_count = config->hidden_size * config->input_feature_size;
    hidden_to_hidden_count = config->hidden_size * config->hidden_size;
    hidden_to_output_count = config->output_size * config->hidden_size;

    context->input_to_hidden = (float*)calloc(input_to_hidden_count, sizeof(float));
    context->hidden_to_hidden = (float*)calloc(hidden_to_hidden_count, sizeof(float));
    context->hidden_bias = (float*)calloc(config->hidden_size, sizeof(float));
    context->hidden_to_output = (float*)calloc(hidden_to_output_count, sizeof(float));
    context->output_bias = (float*)calloc(config->output_size, sizeof(float));
    context->input_buffer = (float*)calloc(
        config->sequence_length * config->input_feature_size,
        sizeof(float)
    );
    context->output_buffer = (float*)calloc(config->output_size, sizeof(float));

    if (context->input_to_hidden == NULL || context->hidden_to_hidden == NULL ||
        context->hidden_bias == NULL || context->hidden_to_output == NULL ||
        context->output_bias == NULL || context->input_buffer == NULL ||
        context->output_buffer == NULL) {
        nn_rnn_infer_destroy(context);
        return NULL;
    }

    /* Initialize each parameter block with small deterministic values. */
    for (value_index = 0U; value_index < input_to_hidden_count; ++value_index) {
        context->input_to_hidden[value_index] = rnn_random_weight(&context->rng_state, 0.24f);
    }
    for (value_index = 0U; value_index < hidden_to_hidden_count; ++value_index) {
        context->hidden_to_hidden[value_index] = rnn_random_weight(&context->rng_state, 0.18f);
    }
    for (value_index = 0U; value_index < config->hidden_size; ++value_index) {
        context->hidden_bias[value_index] = rnn_random_weight(&context->rng_state, 0.05f);
    }
    for (value_index = 0U; value_index < hidden_to_output_count; ++value_index) {
        context->hidden_to_output[value_index] = rnn_random_weight(&context->rng_state, 0.20f);
    }
    for (value_index = 0U; value_index < config->output_size; ++value_index) {
        context->output_bias[value_index] = rnn_random_weight(&context->rng_state, 0.05f);
    }

    return context;
}

/**
 * @brief Free every owned RNN resource in the reverse order of construction.
 */
void nn_rnn_infer_destroy(void* ctx) {
    RnnInferContext* context = (RnnInferContext*)ctx;

    if (context == NULL) {
        return;
    }

    free(context->input_to_hidden);
    free(context->hidden_to_hidden);
    free(context->hidden_bias);
    free(context->hidden_to_output);
    free(context->output_bias);
    free(context->input_buffer);
    free(context->output_buffer);
    free(context);
}

/**
 * @brief Copy caller input into the owned inference buffer.
 */
void nn_rnn_infer_set_input(void* ctx, const float* input, size_t size) {
    RnnInferContext* context = (RnnInferContext*)ctx;
    size_t expected_size;

    if (context == NULL || input == NULL) {
        return;
    }

    expected_size = context->config.sequence_length * context->config.input_feature_size;
    if (size != expected_size) {
        return;
    }

    (void)memcpy(context->input_buffer, input, size * sizeof(float));
}

/**
 * @brief Copy the latest output into caller-owned storage.
 */
void nn_rnn_infer_get_output(void* ctx, float* output, size_t size) {
    RnnInferContext* context = (RnnInferContext*)ctx;

    if (context == NULL || output == NULL || size != context->config.output_size) {
        return;
    }

    (void)memcpy(output, context->output_buffer, size * sizeof(float));
}

/**
 * @brief Run the RNN forward pipeline with optional hidden-state caches.
 */
int nn_rnn_forward_pass(
    RnnInferContext* context,
    const float* input,
    float* output,
    float* hidden_cache,
    float* output_linear_cache
) {
    const RnnConfig* config;
    float previous_hidden[64U];
    size_t step_index;
    size_t hidden_index;

    if (context == NULL || input == NULL || output == NULL) {
        return -1;
    }

    config = &context->config;
    (void)memset(previous_hidden, 0, sizeof(previous_hidden));

    /* Walk the sequence from oldest frame feature to newest frame feature. */
    for (step_index = 0U; step_index < config->sequence_length; ++step_index) {
        const float* step_input = input + (step_index * config->input_feature_size);
        float current_hidden[64U];

        for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
            float linear_value = context->hidden_bias[hidden_index];
            size_t input_index;
            size_t recurrent_index;

            for (input_index = 0U; input_index < config->input_feature_size; ++input_index) {
                linear_value += context->input_to_hidden[
                    (hidden_index * config->input_feature_size) + input_index
                ] * step_input[input_index];
            }
            for (recurrent_index = 0U; recurrent_index < config->hidden_size; ++recurrent_index) {
                linear_value += context->hidden_to_hidden[
                    (hidden_index * config->hidden_size) + recurrent_index
                ] * previous_hidden[recurrent_index];
            }

            current_hidden[hidden_index] =
                rnn_apply_activation(linear_value, config->hidden_activation);
            if (hidden_cache != NULL) {
                hidden_cache[(step_index * config->hidden_size) + hidden_index] =
                    current_hidden[hidden_index];
            }
        }

        (void)memcpy(previous_hidden, current_hidden, config->hidden_size * sizeof(float));
    }

    /* Only the final hidden state is projected into the control output axes. */
    for (hidden_index = 0U; hidden_index < config->output_size; ++hidden_index) {
        float linear_value = context->output_bias[hidden_index];
        size_t source_index;

        for (source_index = 0U; source_index < config->hidden_size; ++source_index) {
            linear_value += context->hidden_to_output[
                (hidden_index * config->hidden_size) + source_index
            ] * previous_hidden[source_index];
        }

        if (output_linear_cache != NULL) {
            output_linear_cache[hidden_index] = linear_value;
        }
        output[hidden_index] = rnn_apply_activation(linear_value, config->output_activation);
    }

    return 0;
}

/**
 * @brief Execute one inference step using the context-owned input and output buffers.
 */
int nn_rnn_infer_step(void* ctx) {
    RnnInferContext* context = (RnnInferContext*)ctx;

    if (context == NULL) {
        return -1;
    }

    return nn_rnn_forward_pass(
        context,
        context->input_buffer,
        context->output_buffer,
        NULL,
        NULL
    );
}

/**
 * @brief Convenience wrapper that performs input copy, forward pass, and output copy.
 */
int nn_rnn_infer_auto_run(void* ctx, const float* input, float* output) {
    RnnInferContext* context = (RnnInferContext*)ctx;

    if (context == NULL || input == NULL || output == NULL) {
        return -1;
    }

    nn_rnn_infer_set_input(
        context,
        input,
        context->config.sequence_length * context->config.input_feature_size
    );
    if (nn_rnn_infer_step(context) != 0) {
        return -1;
    }
    nn_rnn_infer_get_output(context, output, context->config.output_size);
    return 0;
}

/**
 * @brief Load RNN parameters after validating topology and ABI metadata.
 */
int nn_rnn_load_weights(void* ctx, FILE* fp) {
    RnnInferContext* context = (RnnInferContext*)ctx;
    RnnWeightHeader header;
    size_t input_to_hidden_count;
    size_t hidden_to_hidden_count;
    size_t hidden_to_output_count;

    if (context == NULL || fp == NULL) {
        return 0;
    }
    if (fread(&header, sizeof(header), 1, fp) != 1U) {
        return 0;
    }
    if (header.abi_version != RNN_ABI_VERSION) {
        return 0;
    }
    if (context->expected_network_hash != 0U && header.network_hash != context->expected_network_hash) {
        return 0;
    }
    if (context->expected_layout_hash != 0U && header.layout_hash != context->expected_layout_hash) {
        return 0;
    }
    if (header.sequence_length != context->config.sequence_length ||
        header.input_feature_size != context->config.input_feature_size ||
        header.hidden_size != context->config.hidden_size ||
        header.output_size != context->config.output_size ||
        header.hidden_activation != (uint32_t)context->config.hidden_activation ||
        header.output_activation != (uint32_t)context->config.output_activation) {
        return 0;
    }

    input_to_hidden_count = context->config.hidden_size * context->config.input_feature_size;
    hidden_to_hidden_count = context->config.hidden_size * context->config.hidden_size;
    hidden_to_output_count = context->config.output_size * context->config.hidden_size;

    if (fread(context->input_to_hidden, sizeof(float), input_to_hidden_count, fp) != input_to_hidden_count) {
        return 0;
    }
    if (fread(context->hidden_to_hidden, sizeof(float), hidden_to_hidden_count, fp) != hidden_to_hidden_count) {
        return 0;
    }
    if (fread(context->hidden_bias, sizeof(float), context->config.hidden_size, fp) != context->config.hidden_size) {
        return 0;
    }
    if (fread(context->hidden_to_output, sizeof(float), hidden_to_output_count, fp) != hidden_to_output_count) {
        return 0;
    }
    if (fread(context->output_bias, sizeof(float), context->config.output_size, fp) != context->config.output_size) {
        return 0;
    }

    return 1;
}

/**
 * @brief Save RNN parameters together with compatibility metadata.
 */
int nn_rnn_save_weights(void* ctx, FILE* fp) {
    RnnInferContext* context = (RnnInferContext*)ctx;
    RnnWeightHeader header;
    size_t input_to_hidden_count;
    size_t hidden_to_hidden_count;
    size_t hidden_to_output_count;

    if (context == NULL || fp == NULL) {
        return 0;
    }

    header.network_hash = context->expected_network_hash != 0U ?
        context->expected_network_hash : rnn_compute_layout_hash(&context->config);
    header.layout_hash = context->expected_layout_hash != 0U ?
        context->expected_layout_hash : rnn_compute_layout_hash(&context->config);
    header.abi_version = RNN_ABI_VERSION;
    header.sequence_length = (uint32_t)context->config.sequence_length;
    header.input_feature_size = (uint32_t)context->config.input_feature_size;
    header.hidden_size = (uint32_t)context->config.hidden_size;
    header.output_size = (uint32_t)context->config.output_size;
    header.hidden_activation = (uint32_t)context->config.hidden_activation;
    header.output_activation = (uint32_t)context->config.output_activation;

    input_to_hidden_count = context->config.hidden_size * context->config.input_feature_size;
    hidden_to_hidden_count = context->config.hidden_size * context->config.hidden_size;
    hidden_to_output_count = context->config.output_size * context->config.hidden_size;

    if (fwrite(&header, sizeof(header), 1, fp) != 1U) {
        return 0;
    }
    if (fwrite(context->input_to_hidden, sizeof(float), input_to_hidden_count, fp) != input_to_hidden_count) {
        return 0;
    }
    if (fwrite(context->hidden_to_hidden, sizeof(float), hidden_to_hidden_count, fp) != hidden_to_hidden_count) {
        return 0;
    }
    if (fwrite(context->hidden_bias, sizeof(float), context->config.hidden_size, fp) != context->config.hidden_size) {
        return 0;
    }
    if (fwrite(context->hidden_to_output, sizeof(float), hidden_to_output_count, fp) != hidden_to_output_count) {
        return 0;
    }
    if (fwrite(context->output_bias, sizeof(float), context->config.output_size, fp) != context->config.output_size) {
        return 0;
    }

    return 1;
}

/**
 * @brief Expose the current structural hash to persistence callers.
 */
uint64_t nn_rnn_get_network_hash(const void* ctx) {
    const RnnInferContext* context = (const RnnInferContext*)ctx;

    if (context == NULL) {
        return 0U;
    }

    return rnn_compute_layout_hash(&context->config);
}
