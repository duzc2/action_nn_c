/**
 * @file gnn_infer_ops.c
 * @brief Tiny fixed-topology GNN inference backend.
 *
 * This backend is intentionally compact. It accepts one flattened graph input,
 * rebuilds per-node hidden states with a deterministic message-passing loop,
 * and exports one small readout vector that can feed another leaf in a
 * profiler-generated composed network.
 */

#include "gnn_infer_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GNN_ABI_VERSION 2U

/**
 * @brief Serialized header written before the flat GNN parameter arrays.
 */
typedef struct {
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t abi_version;
    uint32_t node_count;
    uint32_t node_feature_size;
    uint32_t hidden_size;
    uint32_t output_size;
    uint32_t message_passes;
    uint32_t slot_count;
    uint32_t aggregator_type;
    uint32_t readout_type;
    uint32_t node_mask_feature_index;
    uint32_t primary_anchor_feature_index;
    uint32_t secondary_anchor_feature_index;
    uint32_t hidden_activation;
    uint32_t output_activation;
} GnnWeightHeader;

/**
 * @brief Advance the tiny deterministic RNG used for parameter init.
 */
static uint32_t gnn_next_random(uint32_t* state) {
    uint32_t value = *state;

    value = value * 1664525U + 1013904223U;
    *state = value;
    return value;
}

/**
 * @brief Generate a small centered random weight.
 */
static float gnn_random_weight(uint32_t* state, float scale) {
    float normalized = (float)(gnn_next_random(state) & 0xFFFFU) / 65535.0f;

    return (normalized - 0.5f) * scale;
}

/**
 * @brief Apply the activation configured for one stage.
 */
static float gnn_apply_activation(float value, GnnActivationType activation) {
    switch (activation) {
        case GNN_ACT_RELU:
            return value > 0.0f ? value : 0.0f;
        case GNN_ACT_TANH:
            return tanhf(value);
        case GNN_ACT_NONE:
        default:
            return value;
    }
}

/**
 * @brief Validate that the config stays within the backend's fixed bounds.
 */
static int gnn_config_is_valid(const GnnConfig* config) {
    size_t node_index;
    size_t slot_index;

    if (config == NULL) {
        return 0;
    }
    if (config->node_count == 0U || config->node_count > GNN_MAX_NODE_COUNT) {
        return 0;
    }
    if (config->node_feature_size == 0U || config->node_feature_size > GNN_MAX_NODE_FEATURE_SIZE) {
        return 0;
    }
    if (config->hidden_size == 0U || config->hidden_size > GNN_MAX_HIDDEN_SIZE) {
        return 0;
    }
    if (config->output_size == 0U || config->output_size > GNN_MAX_OUTPUT_SIZE) {
        return 0;
    }
    if (config->message_passes > GNN_MAX_MESSAGE_PASSES) {
        return 0;
    }
    if (config->slot_count > GNN_MAX_SLOT_COUNT) {
        return 0;
    }
    if (config->node_mask_feature_index != GNN_FEATURE_INDEX_NONE &&
        config->node_mask_feature_index >= config->node_feature_size) {
        return 0;
    }
    if (config->primary_anchor_feature_index != GNN_FEATURE_INDEX_NONE &&
        config->primary_anchor_feature_index >= config->node_feature_size) {
        return 0;
    }
    if (config->secondary_anchor_feature_index != GNN_FEATURE_INDEX_NONE &&
        config->secondary_anchor_feature_index >= config->node_feature_size) {
        return 0;
    }
    if (config->aggregator_type != GNN_AGG_MEAN) {
        return 0;
    }
    if (config->readout_type == GNN_READOUT_ANCHOR_SLOTS && config->slot_count == 0U) {
        return 0;
    }
    if (config->readout_type != GNN_READOUT_GRAPH_POOL &&
        config->readout_type != GNN_READOUT_ANCHOR_SLOTS) {
        return 0;
    }

    for (node_index = 0U; node_index < config->node_count; ++node_index) {
        for (slot_index = 0U; slot_index < config->slot_count; ++slot_index) {
            int neighbor = config->neighbor_index[node_index][slot_index];

            if (neighbor >= 0 && (size_t)neighbor >= config->node_count) {
                return 0;
            }
        }
    }

    return 1;
}

/**
 * @brief Fold one 64-bit value into the lightweight structural hash.
 */
static uint64_t gnn_hash_u64(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= 0x100000001b3ULL;
    return hash;
}

/**
 * @brief Compute a compact structural hash from the active GNN config.
 */
static uint64_t gnn_compute_layout_hash(const GnnConfig* config) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    size_t node_index;
    size_t slot_index;

    if (config == NULL) {
        return hash;
    }

    hash = gnn_hash_u64(hash, (uint64_t)config->node_count);
    hash = gnn_hash_u64(hash, (uint64_t)config->node_feature_size);
    hash = gnn_hash_u64(hash, (uint64_t)config->hidden_size);
    hash = gnn_hash_u64(hash, (uint64_t)config->output_size);
    hash = gnn_hash_u64(hash, (uint64_t)config->message_passes);
    hash = gnn_hash_u64(hash, (uint64_t)config->slot_count);
    hash = gnn_hash_u64(hash, (uint64_t)config->aggregator_type);
    hash = gnn_hash_u64(hash, (uint64_t)config->readout_type);
    hash = gnn_hash_u64(hash, (uint64_t)config->node_mask_feature_index);
    hash = gnn_hash_u64(hash, (uint64_t)config->primary_anchor_feature_index);
    hash = gnn_hash_u64(hash, (uint64_t)config->secondary_anchor_feature_index);
    hash = gnn_hash_u64(hash, (uint64_t)config->hidden_activation);
    hash = gnn_hash_u64(hash, (uint64_t)config->output_activation);

    for (node_index = 0U; node_index < config->node_count; ++node_index) {
        for (slot_index = 0U; slot_index < config->slot_count; ++slot_index) {
            uint64_t encoded_neighbor = (uint64_t)(config->neighbor_index[node_index][slot_index] + 1);
            hash = gnn_hash_u64(hash, encoded_neighbor);
        }
    }

    return hash;
}

/**
 * @brief Return the total flattened input width expected by this context.
 */
static size_t gnn_total_input_size(const GnnConfig* config) {
    return config->node_count * config->node_feature_size;
}

/**
 * @brief Return the number of hidden values stored for one message-passing stage.
 */
static size_t gnn_stage_stride(const GnnConfig* config) {
    return config->node_count * config->hidden_size;
}

/**
 * @brief Report whether one node participates in the active graph instance.
 */
static int gnn_node_is_active(const GnnConfig* config, const float* input, size_t node_index) {
    size_t offset;

    if (config->node_mask_feature_index == GNN_FEATURE_INDEX_NONE) {
        return 1;
    }
    offset = (node_index * config->node_feature_size) + config->node_mask_feature_index;
    return input[offset] > 0.5f;
}

/**
 * @brief Find the first active node that can safely serve as an anchor fallback.
 */
static size_t gnn_find_default_active_node(const GnnConfig* config, const float* input) {
    size_t node_index;

    for (node_index = 0U; node_index < config->node_count; ++node_index) {
        if (gnn_node_is_active(config, input, node_index)) {
            return node_index;
        }
    }

    return 0U;
}

/**
 * @brief Find the active node with the strongest anchor feature.
 */
static size_t gnn_find_anchor_node(
    const GnnConfig* config,
    const float* input,
    size_t feature_index,
    size_t fallback_node
) {
    float best_value = -1000000.0f;
    size_t best_node = fallback_node;
    size_t node_index;
    int found = 0;

    if (best_node >= config->node_count || !gnn_node_is_active(config, input, best_node)) {
        best_node = gnn_find_default_active_node(config, input);
    }

    if (feature_index == GNN_FEATURE_INDEX_NONE) {
        return best_node;
    }

    for (node_index = 0U; node_index < config->node_count; ++node_index) {
        float value = input[(node_index * config->node_feature_size) + feature_index];

        if (!gnn_node_is_active(config, input, node_index)) {
            continue;
        }
        if (!found || value > best_value) {
            best_value = value;
            best_node = node_index;
            found = 1;
        }
    }

    return best_node;
}

/**
 * @brief Mean-pool the hidden state of every active node into one graph vector.
 */
static size_t gnn_collect_graph_pool(
    const GnnConfig* config,
    const float* input,
    const float* final_stage,
    float* pooled_hidden
) {
    size_t active_count = 0U;
    size_t node_index;
    size_t hidden_index;

    (void)memset(pooled_hidden, 0, config->hidden_size * sizeof(float));

    for (node_index = 0U; node_index < config->node_count; ++node_index) {
        if (!gnn_node_is_active(config, input, node_index)) {
            continue;
        }

        for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
            pooled_hidden[hidden_index] +=
                final_stage[(node_index * config->hidden_size) + hidden_index];
        }
        active_count += 1U;
    }

    if (active_count > 0U) {
        for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
            pooled_hidden[hidden_index] /= (float)active_count;
        }
    }

    return active_count;
}

/**
 * @brief Resolve one slot-selected neighbor while skipping inactive nodes.
 */
static int gnn_get_active_neighbor(
    const GnnConfig* config,
    const float* input,
    size_t source_node,
    size_t slot_index
) {
    int neighbor;

    if (slot_index >= config->slot_count || source_node >= config->node_count) {
        return -1;
    }

    neighbor = config->neighbor_index[source_node][slot_index];
    if (neighbor < 0) {
        return -1;
    }
    if (!gnn_node_is_active(config, input, (size_t)neighbor)) {
        return -1;
    }
    return neighbor;
}

/**
 * @brief Read or write one contiguous float parameter block.
 */
static int gnn_transfer_float_block(FILE* fp, float* values, size_t count, int write_mode) {
    size_t transferred;

    if (fp == NULL || values == NULL) {
        return 0;
    }

    transferred = write_mode ?
        fwrite(values, sizeof(float), count, fp) :
        fread(values, sizeof(float), count, fp);
    return transferred == count ? 1 : 0;
}

/**
 * @brief Deliberately reject implicit construction without typed config metadata.
 */
GnnInferContext* nn_gnn_infer_create(void) {
    return NULL;
}

/**
 * @brief Allocate and initialize one typed GNN inference context.
 */
GnnInferContext* nn_gnn_infer_create_with_config(const GnnConfig* config, uint32_t seed) {
    GnnInferContext* context;
    size_t input_weight_count;
    size_t hidden_weight_count;
    size_t readout_weight_count;
    size_t input_value_count;
    size_t value_index;

    if (!gnn_config_is_valid(config)) {
        return NULL;
    }

    context = (GnnInferContext*)calloc(1U, sizeof(GnnInferContext));
    if (context == NULL) {
        return NULL;
    }

    context->config = *config;
    context->rng_state = seed != 0U ? seed : (config->seed != 0U ? config->seed : 1U);
    input_weight_count = config->hidden_size * config->node_feature_size;
    hidden_weight_count = config->hidden_size * config->hidden_size;
    readout_weight_count = config->output_size * config->hidden_size;
    input_value_count = gnn_total_input_size(config);

    context->input_weight = (float*)calloc(input_weight_count, sizeof(float));
    context->input_bias = (float*)calloc(config->hidden_size, sizeof(float));
    context->self_weight = (float*)calloc(hidden_weight_count, sizeof(float));
    context->message_weight = (float*)calloc(hidden_weight_count, sizeof(float));
    context->message_bias = (float*)calloc(config->hidden_size, sizeof(float));
    context->readout_primary = (float*)calloc(readout_weight_count, sizeof(float));
    context->readout_secondary = (float*)calloc(readout_weight_count, sizeof(float));
    context->readout_neighbor = (float*)calloc(readout_weight_count, sizeof(float));
    context->output_bias = (float*)calloc(config->output_size, sizeof(float));
    context->input_buffer = (float*)calloc(input_value_count, sizeof(float));
    context->output_buffer = (float*)calloc(config->output_size, sizeof(float));

    if (context->input_weight == NULL || context->input_bias == NULL ||
        context->self_weight == NULL || context->message_weight == NULL ||
        context->message_bias == NULL || context->readout_primary == NULL ||
        context->readout_secondary == NULL || context->readout_neighbor == NULL ||
        context->output_bias == NULL || context->input_buffer == NULL ||
        context->output_buffer == NULL) {
        nn_gnn_infer_destroy(context);
        return NULL;
    }

    /* Each parameter block receives deterministic small random initialization. */
    for (value_index = 0U; value_index < input_weight_count; ++value_index) {
        context->input_weight[value_index] = gnn_random_weight(&context->rng_state, 0.24f);
    }
    for (value_index = 0U; value_index < config->hidden_size; ++value_index) {
        context->input_bias[value_index] = gnn_random_weight(&context->rng_state, 0.05f);
        context->message_bias[value_index] = gnn_random_weight(&context->rng_state, 0.05f);
    }
    for (value_index = 0U; value_index < hidden_weight_count; ++value_index) {
        context->self_weight[value_index] = gnn_random_weight(&context->rng_state, 0.18f);
        context->message_weight[value_index] = gnn_random_weight(&context->rng_state, 0.18f);
    }
    for (value_index = 0U; value_index < readout_weight_count; ++value_index) {
        context->readout_primary[value_index] = gnn_random_weight(&context->rng_state, 0.20f);
        context->readout_secondary[value_index] = gnn_random_weight(&context->rng_state, 0.20f);
        context->readout_neighbor[value_index] = gnn_random_weight(&context->rng_state, 0.20f);
    }
    for (value_index = 0U; value_index < config->output_size; ++value_index) {
        context->output_bias[value_index] = gnn_random_weight(&context->rng_state, 0.05f);
    }

    return context;
}

/**
 * @brief Free every owned GNN resource in the reverse order of construction.
 */
void nn_gnn_infer_destroy(void* ctx) {
    GnnInferContext* context = (GnnInferContext*)ctx;

    if (context == NULL) {
        return;
    }

    free(context->input_weight);
    free(context->input_bias);
    free(context->self_weight);
    free(context->message_weight);
    free(context->message_bias);
    free(context->readout_primary);
    free(context->readout_secondary);
    free(context->readout_neighbor);
    free(context->output_bias);
    free(context->input_buffer);
    free(context->output_buffer);
    free(context);
}

/**
 * @brief Copy caller input into the owned inference buffer.
 */
void nn_gnn_infer_set_input(void* ctx, const float* input, size_t size) {
    GnnInferContext* context = (GnnInferContext*)ctx;
    size_t expected_size;

    if (context == NULL || input == NULL) {
        return;
    }

    expected_size = gnn_total_input_size(&context->config);
    if (size != expected_size) {
        return;
    }

    (void)memcpy(context->input_buffer, input, size * sizeof(float));
}

/**
 * @brief Copy the latest output into caller-owned storage.
 */
void nn_gnn_infer_get_output(void* ctx, float* output, size_t size) {
    GnnInferContext* context = (GnnInferContext*)ctx;

    if (context == NULL || output == NULL) {
        return;
    }

    if (size != context->config.output_size) {
        return;
    }

    (void)memcpy(output, context->output_buffer, size * sizeof(float));
}

/**
 * @brief Run the full node-encoding, message-passing, and configurable readout pipeline.
 */
int nn_gnn_forward_pass(
    GnnInferContext* context,
    const float* input,
    float* output,
    float* hidden_cache
) {
    const GnnConfig* config;
    float local_hidden_cache[(GNN_MAX_MESSAGE_PASSES + 1U) * GNN_MAX_NODE_COUNT * GNN_MAX_HIDDEN_SIZE];
    float* cache = hidden_cache != NULL ? hidden_cache : local_hidden_cache;
    size_t stage_stride;
    size_t node_index;
    size_t hidden_index;
    size_t pass_index;
    size_t output_index;
    const float* final_stage;

    if (context == NULL || input == NULL || output == NULL) {
        return -1;
    }

    config = &context->config;
    if (!gnn_config_is_valid(config)) {
        return -1;
    }

    stage_stride = gnn_stage_stride(config);

    /* Stage 0 encodes each active node feature vector into the hidden state space. */
    for (node_index = 0U; node_index < config->node_count; ++node_index) {
        float* stage0_node = cache + (node_index * config->hidden_size);
        const float* input_node = input + (node_index * config->node_feature_size);

        if (!gnn_node_is_active(config, input, node_index)) {
            (void)memset(stage0_node, 0, config->hidden_size * sizeof(float));
            continue;
        }

        for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
            float linear = context->input_bias[hidden_index];
            size_t feature_index;

            for (feature_index = 0U; feature_index < config->node_feature_size; ++feature_index) {
                linear += context->input_weight[
                    (hidden_index * config->node_feature_size) + feature_index
                ] * input_node[feature_index];
            }
            stage0_node[hidden_index] = gnn_apply_activation(linear, config->hidden_activation);
        }
    }

    /* Each message-passing stage mixes the node's own state with the mean of active neighbors. */
    for (pass_index = 1U; pass_index <= config->message_passes; ++pass_index) {
        const float* previous_stage = cache + ((pass_index - 1U) * stage_stride);
        float* current_stage = cache + (pass_index * stage_stride);

        for (node_index = 0U; node_index < config->node_count; ++node_index) {
            float aggregated[GNN_MAX_HIDDEN_SIZE];
            float* current_node_hidden = current_stage + (node_index * config->hidden_size);
            size_t neighbor_count = 0U;
            size_t slot_index;

            (void)memset(aggregated, 0, config->hidden_size * sizeof(float));

            if (!gnn_node_is_active(config, input, node_index)) {
                (void)memset(current_node_hidden, 0, config->hidden_size * sizeof(float));
                continue;
            }

            for (slot_index = 0U; slot_index < config->slot_count; ++slot_index) {
                int neighbor = gnn_get_active_neighbor(config, input, node_index, slot_index);

                if (neighbor >= 0) {
                    const float* neighbor_hidden = previous_stage + ((size_t)neighbor * config->hidden_size);

                    for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
                        aggregated[hidden_index] += neighbor_hidden[hidden_index];
                    }
                    neighbor_count += 1U;
                }
            }

            if (neighbor_count > 0U) {
                for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
                    aggregated[hidden_index] /= (float)neighbor_count;
                }
            }

            for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
                float linear = context->message_bias[hidden_index];
                const float* previous_node_hidden = previous_stage + (node_index * config->hidden_size);
                size_t source_hidden;

                for (source_hidden = 0U; source_hidden < config->hidden_size; ++source_hidden) {
                    linear += context->self_weight[
                        (hidden_index * config->hidden_size) + source_hidden
                    ] * previous_node_hidden[source_hidden];
                    linear += context->message_weight[
                        (hidden_index * config->hidden_size) + source_hidden
                    ] * aggregated[source_hidden];
                }
                current_node_hidden[hidden_index] = gnn_apply_activation(linear, config->hidden_activation);
            }
        }
    }

    final_stage = cache + (config->message_passes * stage_stride);

    if (config->readout_type == GNN_READOUT_GRAPH_POOL) {
        float pooled_hidden[GNN_MAX_HIDDEN_SIZE];

        (void)gnn_collect_graph_pool(config, input, final_stage, pooled_hidden);

        for (output_index = 0U; output_index < config->output_size; ++output_index) {
            float linear = context->output_bias[output_index];

            for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
                linear += context->readout_primary[
                    (output_index * config->hidden_size) + hidden_index
                ] * pooled_hidden[hidden_index];
            }

            output[output_index] = gnn_apply_activation(linear, config->output_activation);
        }
    } else {
        size_t primary_anchor = gnn_find_anchor_node(
            config,
            input,
            config->primary_anchor_feature_index,
            0U
        );
        size_t secondary_anchor = gnn_find_anchor_node(
            config,
            input,
            config->secondary_anchor_feature_index,
            primary_anchor
        );

        /* Anchor-slot readout exports one vector around the selected primary anchor. */
        for (output_index = 0U; output_index < config->output_size; ++output_index) {
            size_t slot_index = output_index % config->slot_count;
            int neighbor = gnn_get_active_neighbor(config, input, primary_anchor, slot_index);
            const float* primary_hidden = final_stage + (primary_anchor * config->hidden_size);
            const float* secondary_hidden = final_stage + (secondary_anchor * config->hidden_size);
            const float* neighbor_hidden = NULL;
            float linear = context->output_bias[output_index];

            if (neighbor >= 0) {
                neighbor_hidden = final_stage + ((size_t)neighbor * config->hidden_size);
            }

            for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
                linear += context->readout_primary[
                    (output_index * config->hidden_size) + hidden_index
                ] * primary_hidden[hidden_index];
                linear += context->readout_secondary[
                    (output_index * config->hidden_size) + hidden_index
                ] * secondary_hidden[hidden_index];
                if (neighbor_hidden != NULL) {
                    linear += context->readout_neighbor[
                        (output_index * config->hidden_size) + hidden_index
                    ] * neighbor_hidden[hidden_index];
                }
            }

            output[output_index] = gnn_apply_activation(linear, config->output_activation);
        }
    }

    return 0;
}

/**
 * @brief Execute one inference step using the context-owned input buffer.
 */
int nn_gnn_infer_step(void* ctx) {
    GnnInferContext* context = (GnnInferContext*)ctx;

    if (context == NULL) {
        return -1;
    }

    return nn_gnn_forward_pass(
        context,
        context->input_buffer,
        context->output_buffer,
        NULL
    );
}

/**
 * @brief Convenience wrapper that performs set-input, step, and get-output.
 */
int nn_gnn_infer_auto_run(void* ctx, const float* input, float* output) {
    GnnInferContext* context = (GnnInferContext*)ctx;

    if (context == NULL || input == NULL || output == NULL) {
        return -1;
    }

    nn_gnn_infer_set_input(context, input, gnn_total_input_size(&context->config));
    if (nn_gnn_infer_step(context) != 0) {
        return -1;
    }
    nn_gnn_infer_get_output(context, output, context->config.output_size);
    return 0;
}

/**
 * @brief Load weights after validating network hash, layout hash, and ABI tag.
 */
int nn_gnn_load_weights(void* ctx, FILE* fp) {
    GnnInferContext* context = (GnnInferContext*)ctx;
    GnnWeightHeader header;
    size_t input_weight_count;
    size_t hidden_weight_count;
    size_t readout_weight_count;

    if (context == NULL || fp == NULL) {
        return 0;
    }

    if (fread(&header, sizeof(header), 1U, fp) != 1U) {
        return 0;
    }
    if (header.abi_version != GNN_ABI_VERSION) {
        return 0;
    }
    if (header.node_count != context->config.node_count ||
        header.node_feature_size != context->config.node_feature_size ||
        header.hidden_size != context->config.hidden_size ||
        header.output_size != context->config.output_size ||
        header.message_passes != context->config.message_passes ||
        header.slot_count != context->config.slot_count ||
        header.aggregator_type != (uint32_t)context->config.aggregator_type ||
        header.readout_type != (uint32_t)context->config.readout_type ||
        header.node_mask_feature_index != (uint32_t)context->config.node_mask_feature_index ||
        header.primary_anchor_feature_index != (uint32_t)context->config.primary_anchor_feature_index ||
        header.secondary_anchor_feature_index != (uint32_t)context->config.secondary_anchor_feature_index ||
        header.hidden_activation != (uint32_t)context->config.hidden_activation ||
        header.output_activation != (uint32_t)context->config.output_activation) {
        return 0;
    }
    if (context->expected_network_hash != 0U && header.network_hash != context->expected_network_hash) {
        return 0;
    }
    if (context->expected_layout_hash != 0U && header.layout_hash != context->expected_layout_hash) {
        return 0;
    }
    if (context->expected_layout_hash == 0U && header.layout_hash != gnn_compute_layout_hash(&context->config)) {
        return 0;
    }

    input_weight_count = context->config.hidden_size * context->config.node_feature_size;
    hidden_weight_count = context->config.hidden_size * context->config.hidden_size;
    readout_weight_count = context->config.output_size * context->config.hidden_size;

    return gnn_transfer_float_block(fp, context->input_weight, input_weight_count, 0) &&
        gnn_transfer_float_block(fp, context->input_bias, context->config.hidden_size, 0) &&
        gnn_transfer_float_block(fp, context->self_weight, hidden_weight_count, 0) &&
        gnn_transfer_float_block(fp, context->message_weight, hidden_weight_count, 0) &&
        gnn_transfer_float_block(fp, context->message_bias, context->config.hidden_size, 0) &&
        gnn_transfer_float_block(fp, context->readout_primary, readout_weight_count, 0) &&
        gnn_transfer_float_block(fp, context->readout_secondary, readout_weight_count, 0) &&
        gnn_transfer_float_block(fp, context->readout_neighbor, readout_weight_count, 0) &&
        gnn_transfer_float_block(fp, context->output_bias, context->config.output_size, 0);
}

/**
 * @brief Save weights together with a compatibility header.
 */
int nn_gnn_save_weights(void* ctx, FILE* fp) {
    GnnInferContext* context = (GnnInferContext*)ctx;
    GnnWeightHeader header;
    size_t input_weight_count;
    size_t hidden_weight_count;
    size_t readout_weight_count;

    if (context == NULL || fp == NULL) {
        return 0;
    }

    header.network_hash = nn_gnn_get_network_hash(context);
    header.layout_hash = context->expected_layout_hash != 0U ?
        context->expected_layout_hash :
        gnn_compute_layout_hash(&context->config);
    header.abi_version = GNN_ABI_VERSION;
    header.node_count = (uint32_t)context->config.node_count;
    header.node_feature_size = (uint32_t)context->config.node_feature_size;
    header.hidden_size = (uint32_t)context->config.hidden_size;
    header.output_size = (uint32_t)context->config.output_size;
    header.message_passes = (uint32_t)context->config.message_passes;
    header.slot_count = (uint32_t)context->config.slot_count;
    header.aggregator_type = (uint32_t)context->config.aggregator_type;
    header.readout_type = (uint32_t)context->config.readout_type;
    header.node_mask_feature_index = (uint32_t)context->config.node_mask_feature_index;
    header.primary_anchor_feature_index = (uint32_t)context->config.primary_anchor_feature_index;
    header.secondary_anchor_feature_index = (uint32_t)context->config.secondary_anchor_feature_index;
    header.hidden_activation = (uint32_t)context->config.hidden_activation;
    header.output_activation = (uint32_t)context->config.output_activation;

    if (fwrite(&header, sizeof(header), 1U, fp) != 1U) {
        return 0;
    }

    input_weight_count = context->config.hidden_size * context->config.node_feature_size;
    hidden_weight_count = context->config.hidden_size * context->config.hidden_size;
    readout_weight_count = context->config.output_size * context->config.hidden_size;

    return gnn_transfer_float_block(fp, context->input_weight, input_weight_count, 1) &&
        gnn_transfer_float_block(fp, context->input_bias, context->config.hidden_size, 1) &&
        gnn_transfer_float_block(fp, context->self_weight, hidden_weight_count, 1) &&
        gnn_transfer_float_block(fp, context->message_weight, hidden_weight_count, 1) &&
        gnn_transfer_float_block(fp, context->message_bias, context->config.hidden_size, 1) &&
        gnn_transfer_float_block(fp, context->readout_primary, readout_weight_count, 1) &&
        gnn_transfer_float_block(fp, context->readout_secondary, readout_weight_count, 1) &&
        gnn_transfer_float_block(fp, context->readout_neighbor, readout_weight_count, 1) &&
        gnn_transfer_float_block(fp, context->output_bias, context->config.output_size, 1);
}

/**
 * @brief Return the externally supplied network hash when available.
 */
uint64_t nn_gnn_get_network_hash(const void* ctx) {
    const GnnInferContext* context = (const GnnInferContext*)ctx;

    if (context == NULL) {
        return 0U;
    }
    if (context->expected_network_hash != 0U) {
        return context->expected_network_hash;
    }
    if (context->expected_layout_hash != 0U) {
        return context->expected_layout_hash;
    }
    return gnn_compute_layout_hash(&context->config);
}

