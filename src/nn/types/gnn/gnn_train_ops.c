/**
 * @file gnn_train_ops.c
 * @brief Dynamically configured GNN training backend.
 *
 * The training path mirrors the handwritten infer backend so profiler-generated
 * composed graphs can backpropagate through the GNN leaf just like they do for
 * other builtin network types.
 */

#include "gnn_train_ops.h"

#include <stdlib.h>
#include <string.h>

/**
 * @brief Recover d(activation)/d(linear) from the post-activation output value.
 */
static float gnn_activation_derivative_from_output(float output, GnnActivationType activation) {
    switch (activation) {
        case GNN_ACT_RELU:
            return output > 0.0f ? 1.0f : 0.0f;
        case GNN_ACT_TANH:
            return 1.0f - (output * output);
        case GNN_ACT_NONE:
        default:
            return 1.0f;
    }
}

/**
 * @brief Return the flattened input size for one graph sample.
 */
static size_t gnn_total_input_size(const GnnConfig* config) {
    return config->node_count * config->node_feature_size;
}

/**
 * @brief Return the hidden value count stored for one message-passing stage.
 */
static size_t gnn_stage_stride(const GnnConfig* config) {
    return config->node_count * config->hidden_size;
}

/**
 * @brief Read one neighbor entry from the flattened topology table.
 */
static int gnn_neighbor_at(const GnnConfig* config, size_t node_index, size_t slot_index) {
    return gnn_config_neighbor_row_view(config, node_index)[slot_index];
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
 * @brief Resolve one active neighbor slot while ignoring inactive nodes.
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

    neighbor = gnn_neighbor_at(config, source_node, slot_index);
    if (neighbor < 0) {
        return -1;
    }
    if (!gnn_node_is_active(config, input, (size_t)neighbor)) {
        return -1;
    }
    return neighbor;
}

/**
 * @brief Clear all gradient buffers before a new backward pass.
 */
static void gnn_zero_gradients(GnnTrainContext* context) {
    const GnnConfig* config;
    size_t input_weight_count;
    size_t hidden_weight_count;
    size_t readout_weight_count;

    if (context == NULL || context->infer_ctx == NULL) {
        return;
    }

    config = context->infer_ctx->config;
    input_weight_count = config->hidden_size * config->node_feature_size;
    hidden_weight_count = config->hidden_size * config->hidden_size;
    readout_weight_count = config->output_size * config->hidden_size;

    (void)memset(context->input_weight_grad, 0, input_weight_count * sizeof(float));
    (void)memset(context->input_bias_grad, 0, config->hidden_size * sizeof(float));
    (void)memset(context->self_weight_grad, 0, hidden_weight_count * sizeof(float));
    (void)memset(context->message_weight_grad, 0, hidden_weight_count * sizeof(float));
    (void)memset(context->message_bias_grad, 0, config->hidden_size * sizeof(float));
    (void)memset(context->readout_primary_grad, 0, readout_weight_count * sizeof(float));
    (void)memset(context->readout_secondary_grad, 0, readout_weight_count * sizeof(float));
    (void)memset(context->readout_neighbor_grad, 0, readout_weight_count * sizeof(float));
    (void)memset(context->output_bias_grad, 0, config->output_size * sizeof(float));
}

/**
 * @brief Apply the accumulated gradients with a simple SGD-style update.
 */
static void gnn_apply_parameter_update(GnnTrainContext* context) {
    GnnInferContext* infer_ctx;
    const GnnConfig* config;
    size_t input_weight_count;
    size_t hidden_weight_count;
    size_t readout_weight_count;
    size_t value_index;

    infer_ctx = context->infer_ctx;
    config = infer_ctx->config;
    input_weight_count = config->hidden_size * config->node_feature_size;
    hidden_weight_count = config->hidden_size * config->hidden_size;
    readout_weight_count = config->output_size * config->hidden_size;

    for (value_index = 0U; value_index < input_weight_count; ++value_index) {
        infer_ctx->input_weight[value_index] -= context->config.learning_rate * (
            context->input_weight_grad[value_index] +
            (context->config.weight_decay * infer_ctx->input_weight[value_index])
        );
    }
    for (value_index = 0U; value_index < config->hidden_size; ++value_index) {
        infer_ctx->input_bias[value_index] -=
            context->config.learning_rate * context->input_bias_grad[value_index];
        infer_ctx->message_bias[value_index] -=
            context->config.learning_rate * context->message_bias_grad[value_index];
    }
    for (value_index = 0U; value_index < hidden_weight_count; ++value_index) {
        infer_ctx->self_weight[value_index] -= context->config.learning_rate * (
            context->self_weight_grad[value_index] +
            (context->config.weight_decay * infer_ctx->self_weight[value_index])
        );
        infer_ctx->message_weight[value_index] -= context->config.learning_rate * (
            context->message_weight_grad[value_index] +
            (context->config.weight_decay * infer_ctx->message_weight[value_index])
        );
    }
    for (value_index = 0U; value_index < readout_weight_count; ++value_index) {
        infer_ctx->readout_primary[value_index] -= context->config.learning_rate * (
            context->readout_primary_grad[value_index] +
            (context->config.weight_decay * infer_ctx->readout_primary[value_index])
        );
        infer_ctx->readout_secondary[value_index] -= context->config.learning_rate * (
            context->readout_secondary_grad[value_index] +
            (context->config.weight_decay * infer_ctx->readout_secondary[value_index])
        );
        infer_ctx->readout_neighbor[value_index] -= context->config.learning_rate * (
            context->readout_neighbor_grad[value_index] +
            (context->config.weight_decay * infer_ctx->readout_neighbor[value_index])
        );
    }
    for (value_index = 0U; value_index < config->output_size; ++value_index) {
        infer_ctx->output_bias[value_index] -=
            context->config.learning_rate * context->output_bias_grad[value_index];
    }
}

/**
 * @brief Backpropagate one externally supplied dL/dY through the GNN leaf.
 */
static int gnn_backpropagate(
    GnnTrainContext* context,
    const float* input,
    const float* output_gradient,
    float* input_gradient
) {
    GnnInferContext* infer_ctx;
    const GnnConfig* config;
    const float* final_stage;
    const float* current_stage;
    const float* previous_stage;
    float* stage_grad;
    float* previous_stage_grad;
    float* pooled_hidden = NULL;
    float* pooled_grad = NULL;
    float* aggregated = NULL;
    float* aggregated_grad = NULL;
    size_t stage_stride;
    size_t node_index;
    size_t output_index;
    size_t hidden_index;
    size_t pass_index;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || output_gradient == NULL) {
        return -1;
    }

    infer_ctx = context->infer_ctx;
    config = infer_ctx->config;
    stage_stride = gnn_stage_stride(config);
    gnn_zero_gradients(context);
    stage_grad = (float*)calloc(stage_stride, sizeof(float));
    previous_stage_grad = (float*)calloc(stage_stride, sizeof(float));
    aggregated = (float*)calloc(config->hidden_size, sizeof(float));
    aggregated_grad = (float*)calloc(config->hidden_size, sizeof(float));
    if (stage_grad == NULL || previous_stage_grad == NULL ||
        aggregated == NULL || aggregated_grad == NULL) {
        free(stage_grad);
        free(previous_stage_grad);
        free(aggregated);
        free(aggregated_grad);
        return -1;
    }

    if (input_gradient != NULL) {
        (void)memset(input_gradient, 0, gnn_total_input_size(config) * sizeof(float));
    }

    final_stage = context->hidden_cache + (config->message_passes * stage_stride);

    if (config->readout_type == GNN_READOUT_GRAPH_POOL) {
        size_t active_count;

        pooled_hidden = (float*)calloc(config->hidden_size, sizeof(float));
        pooled_grad = (float*)calloc(config->hidden_size, sizeof(float));
        if (pooled_hidden == NULL || pooled_grad == NULL) {
            free(stage_grad);
            free(previous_stage_grad);
            free(pooled_hidden);
            free(pooled_grad);
            free(aggregated);
            free(aggregated_grad);
            return -1;
        }
        active_count = gnn_collect_graph_pool(config, input, final_stage, pooled_hidden);

        /* Graph-pool readout only depends on the pooled hidden state and readout_primary. */
        for (output_index = 0U; output_index < config->output_size; ++output_index) {
            float dz = output_gradient[output_index] * gnn_activation_derivative_from_output(
                infer_ctx->output_buffer[output_index],
                config->output_activation
            );

            context->output_bias_grad[output_index] += dz;
            for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
                size_t weight_index = (output_index * config->hidden_size) + hidden_index;

                context->readout_primary_grad[weight_index] += dz * pooled_hidden[hidden_index];
                pooled_grad[hidden_index] += infer_ctx->readout_primary[weight_index] * dz;
            }
        }

        if (active_count > 0U) {
            for (node_index = 0U; node_index < config->node_count; ++node_index) {
                if (!gnn_node_is_active(config, input, node_index)) {
                    continue;
                }

                for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
                    stage_grad[(node_index * config->hidden_size) + hidden_index] +=
                        pooled_grad[hidden_index] / (float)active_count;
                }
            }
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

        /* Anchor-slot readout flows back to the selected anchor nodes and slot neighbor. */
        for (output_index = 0U; output_index < config->output_size; ++output_index) {
            size_t slot_index = output_index % config->slot_count;
            int neighbor = gnn_get_active_neighbor(config, input, primary_anchor, slot_index);
            float dz = output_gradient[output_index] * gnn_activation_derivative_from_output(
                infer_ctx->output_buffer[output_index],
                config->output_activation
            );
            const float* primary_hidden = final_stage + (primary_anchor * config->hidden_size);
            const float* secondary_hidden = final_stage + (secondary_anchor * config->hidden_size);

            context->output_bias_grad[output_index] += dz;
            for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
                size_t weight_index = (output_index * config->hidden_size) + hidden_index;

                context->readout_primary_grad[weight_index] += dz * primary_hidden[hidden_index];
                context->readout_secondary_grad[weight_index] += dz * secondary_hidden[hidden_index];
                stage_grad[(primary_anchor * config->hidden_size) + hidden_index] +=
                    infer_ctx->readout_primary[weight_index] * dz;
                stage_grad[(secondary_anchor * config->hidden_size) + hidden_index] +=
                    infer_ctx->readout_secondary[weight_index] * dz;
                if (neighbor >= 0) {
                    const float* neighbor_hidden = final_stage + ((size_t)neighbor * config->hidden_size);

                    context->readout_neighbor_grad[weight_index] += dz * neighbor_hidden[hidden_index];
                    stage_grad[((size_t)neighbor * config->hidden_size) + hidden_index] +=
                        infer_ctx->readout_neighbor[weight_index] * dz;
                }
            }
        }
    }

    /* Then unroll the message-passing stack in reverse order. */
    for (pass_index = config->message_passes; pass_index > 0U; --pass_index) {
        current_stage = context->hidden_cache + (pass_index * stage_stride);
        previous_stage = context->hidden_cache + ((pass_index - 1U) * stage_stride);
        (void)memset(previous_stage_grad, 0, stage_stride * sizeof(float));

        for (node_index = 0U; node_index < config->node_count; ++node_index) {
            size_t neighbor_count = 0U;
            size_t slot_index;

            (void)memset(aggregated, 0, config->hidden_size * sizeof(float));
            (void)memset(aggregated_grad, 0, config->hidden_size * sizeof(float));

            if (!gnn_node_is_active(config, input, node_index)) {
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
                float delta = stage_grad[(node_index * config->hidden_size) + hidden_index] *
                    gnn_activation_derivative_from_output(
                        current_stage[(node_index * config->hidden_size) + hidden_index],
                        config->hidden_activation
                    );
                size_t source_hidden;

                context->message_bias_grad[hidden_index] += delta;
                for (source_hidden = 0U; source_hidden < config->hidden_size; ++source_hidden) {
                    size_t weight_index = (hidden_index * config->hidden_size) + source_hidden;
                    float previous_value = previous_stage[(node_index * config->hidden_size) + source_hidden];

                    context->self_weight_grad[weight_index] += delta * previous_value;
                    context->message_weight_grad[weight_index] += delta * aggregated[source_hidden];
                    previous_stage_grad[(node_index * config->hidden_size) + source_hidden] +=
                        infer_ctx->self_weight[weight_index] * delta;
                    aggregated_grad[source_hidden] += infer_ctx->message_weight[weight_index] * delta;
                }
            }

            if (neighbor_count > 0U) {
                for (slot_index = 0U; slot_index < config->slot_count; ++slot_index) {
                    int neighbor = gnn_get_active_neighbor(config, input, node_index, slot_index);

                    if (neighbor >= 0) {
                        for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
                            previous_stage_grad[((size_t)neighbor * config->hidden_size) + hidden_index] +=
                                aggregated_grad[hidden_index] / (float)neighbor_count;
                        }
                    }
                }
            }
        }

        (void)memcpy(stage_grad, previous_stage_grad, stage_stride * sizeof(float));
    }

    /* Finally backpropagate through the node-feature encoder. */
    previous_stage = context->hidden_cache;
    for (node_index = 0U; node_index < config->node_count; ++node_index) {
        const float* input_node = input + (node_index * config->node_feature_size);

        if (!gnn_node_is_active(config, input, node_index)) {
            continue;
        }

        for (hidden_index = 0U; hidden_index < config->hidden_size; ++hidden_index) {
            float delta = stage_grad[(node_index * config->hidden_size) + hidden_index] *
                gnn_activation_derivative_from_output(
                    previous_stage[(node_index * config->hidden_size) + hidden_index],
                    config->hidden_activation
                );
            size_t feature_index;

            context->input_bias_grad[hidden_index] += delta;
            for (feature_index = 0U; feature_index < config->node_feature_size; ++feature_index) {
                size_t weight_index = (hidden_index * config->node_feature_size) + feature_index;

                context->input_weight_grad[weight_index] += delta * input_node[feature_index];
                if (input_gradient != NULL) {
                    input_gradient[(node_index * config->node_feature_size) + feature_index] +=
                        infer_ctx->input_weight[weight_index] * delta;
                }
            }
        }
    }

    gnn_apply_parameter_update(context);
    free(stage_grad);
    free(previous_stage_grad);
    free(pooled_hidden);
    free(pooled_grad);
    free(aggregated);
    free(aggregated_grad);
    return 0;
}

/**
 * @brief Create one GNN training context wrapped around an existing infer context.
 */
GnnTrainContext* nn_gnn_train_create(void* infer_ctx_ptr, const GnnTrainConfig* config) {
    GnnTrainContext* context;
    GnnInferContext* infer_ctx = (GnnInferContext*)infer_ctx_ptr;
    const GnnConfig* infer_config;
    size_t input_weight_count;
    size_t hidden_weight_count;
    size_t readout_weight_count;

    if (infer_ctx == NULL || config == NULL) {
        return NULL;
    }

    infer_config = infer_ctx->config;
    input_weight_count = infer_config->hidden_size * infer_config->node_feature_size;
    hidden_weight_count = infer_config->hidden_size * infer_config->hidden_size;
    readout_weight_count = infer_config->output_size * infer_config->hidden_size;

    context = (GnnTrainContext*)calloc(1U, sizeof(GnnTrainContext));
    if (context == NULL) {
        return NULL;
    }

    context->infer_ctx = infer_ctx;
    context->config = *config;
    context->hidden_cache = (float*)calloc(
        (infer_config->message_passes + 1U) * gnn_stage_stride(infer_config),
        sizeof(float)
    );
    context->input_weight_grad = (float*)calloc(input_weight_count, sizeof(float));
    context->input_bias_grad = (float*)calloc(infer_config->hidden_size, sizeof(float));
    context->self_weight_grad = (float*)calloc(hidden_weight_count, sizeof(float));
    context->message_weight_grad = (float*)calloc(hidden_weight_count, sizeof(float));
    context->message_bias_grad = (float*)calloc(infer_config->hidden_size, sizeof(float));
    context->readout_primary_grad = (float*)calloc(readout_weight_count, sizeof(float));
    context->readout_secondary_grad = (float*)calloc(readout_weight_count, sizeof(float));
    context->readout_neighbor_grad = (float*)calloc(readout_weight_count, sizeof(float));
    context->output_bias_grad = (float*)calloc(infer_config->output_size, sizeof(float));

    if (context->hidden_cache == NULL || context->input_weight_grad == NULL ||
        context->input_bias_grad == NULL || context->self_weight_grad == NULL ||
        context->message_weight_grad == NULL || context->message_bias_grad == NULL ||
        context->readout_primary_grad == NULL || context->readout_secondary_grad == NULL ||
        context->readout_neighbor_grad == NULL || context->output_bias_grad == NULL) {
        nn_gnn_train_destroy(context);
        return NULL;
    }

    return context;
}

/**
 * @brief Free every training-side scratch buffer owned by the GNN trainer.
 */
void nn_gnn_train_destroy(GnnTrainContext* context) {
    if (context == NULL) {
        return;
    }

    free(context->hidden_cache);
    free(context->input_weight_grad);
    free(context->input_bias_grad);
    free(context->self_weight_grad);
    free(context->message_weight_grad);
    free(context->message_bias_grad);
    free(context->readout_primary_grad);
    free(context->readout_secondary_grad);
    free(context->readout_neighbor_grad);
    free(context->output_bias_grad);
    free(context);
}

/**
 * @brief Run one graph-mode update using a caller-supplied output gradient.
 */
int nn_gnn_train_step_with_output_gradient(
    GnnTrainContext* context,
    const float* input,
    const float* output_gradient,
    float* input_gradient
) {
    int rc;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || output_gradient == NULL) {
        return -1;
    }

    rc = nn_gnn_forward_pass(
        context->infer_ctx,
        input,
        context->infer_ctx->output_buffer,
        context->hidden_cache
    );
    if (rc != 0) {
        return rc;
    }

    rc = gnn_backpropagate(context, input, output_gradient, input_gradient);
    if (rc != 0) {
        return rc;
    }

    context->total_steps += 1U;
    return 0;
}

/**
 * @brief Run one supervised update with an explicit target tensor.
 */
int nn_gnn_train_step_with_data(GnnTrainContext* context, const float* input, const float* target) {
    GnnInferContext* infer_ctx;
    const GnnConfig* config;
    float* output_gradient;
    float loss = 0.0f;
    size_t output_index;
    int rc;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || target == NULL) {
        return -1;
    }

    infer_ctx = context->infer_ctx;
    config = infer_ctx->config;
    output_gradient = (float*)calloc(config->output_size, sizeof(float));
    if (output_gradient == NULL) {
        return -1;
    }
    rc = nn_gnn_forward_pass(
        infer_ctx,
        input,
        infer_ctx->output_buffer,
        context->hidden_cache
    );
    if (rc != 0) {
        free(output_gradient);
        return rc;
    }

    /* The standalone supervised path uses simple MSE to stay transparent. */
    for (output_index = 0U; output_index < config->output_size; ++output_index) {
        float diff = infer_ctx->output_buffer[output_index] - target[output_index];

        loss += diff * diff;
        output_gradient[output_index] = (2.0f * diff) / (float)config->output_size;
    }

    rc = gnn_backpropagate(context, input, output_gradient, NULL);
    if (rc != 0) {
        free(output_gradient);
        return rc;
    }

    free(output_gradient);
    context->total_steps += 1U;
    context->last_loss = loss / (float)config->output_size;
    context->cumulative_loss += context->last_loss;
    context->average_loss = context->cumulative_loss / (float)context->total_steps;
    return 0;
}

/**
 * @brief Report coarse training statistics consumed by generated wrappers.
 */
void nn_gnn_train_get_stats(
    GnnTrainContext* context,
    size_t* out_epochs,
    size_t* out_steps,
    float* out_avg_loss
) {
    if (context == NULL) {
        return;
    }

    if (out_epochs != NULL) {
        *out_epochs = context->total_epochs;
    }
    if (out_steps != NULL) {
        *out_steps = context->total_steps;
    }
    if (out_avg_loss != NULL) {
        *out_avg_loss = context->average_loss;
    }
}

/**
 * @brief Minimal compatibility wrapper for the registry's small train-step hook.
 */
int nn_gnn_train_step(void* ctx) {
    GnnTrainContext* context = (GnnTrainContext*)ctx;
    float* dummy_input;
    float* dummy_target;
    int rc;

    if (context == NULL || context->infer_ctx == NULL) {
        return -1;
    }

    dummy_input = (float*)calloc(gnn_total_input_size(context->infer_ctx->config), sizeof(float));
    dummy_target = (float*)calloc(context->infer_ctx->config->output_size, sizeof(float));
    if (dummy_input == NULL || dummy_target == NULL) {
        free(dummy_input);
        free(dummy_target);
        return -1;
    }

    rc = nn_gnn_train_step_with_data(context, dummy_input, dummy_target);
    free(dummy_input);
    free(dummy_target);
    return rc;
}

