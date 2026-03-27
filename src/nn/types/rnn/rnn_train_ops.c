/**
 * @file rnn_train_ops.c
 * @brief Tiny RNN training backend used by generated graph code.
 */

#include "rnn_train_ops.h"

#include <stdlib.h>
#include <string.h>

/**
 * @brief Recover d(activation)/d(linear) from the post-activation output value.
 */
static float rnn_activation_derivative_from_output(float output, RnnActivationType activation) {
    switch (activation) {
        case RNN_ACT_TANH:
            return 1.0f - (output * output);
        case RNN_ACT_NONE:
        default:
            return 1.0f;
    }
}

/**
 * @brief Clear all gradient buffers before a new backward pass.
 */
static void rnn_zero_gradients(RnnTrainContext* context) {
    const RnnConfig* config;
    size_t input_to_hidden_count;
    size_t hidden_to_hidden_count;
    size_t hidden_to_output_count;

    if (context == NULL || context->infer_ctx == NULL) {
        return;
    }

    config = &context->infer_ctx->config;
    input_to_hidden_count = config->hidden_size * config->input_feature_size;
    hidden_to_hidden_count = config->hidden_size * config->hidden_size;
    hidden_to_output_count = config->output_size * config->hidden_size;

    (void)memset(context->input_to_hidden_grad, 0, input_to_hidden_count * sizeof(float));
    (void)memset(context->hidden_to_hidden_grad, 0, hidden_to_hidden_count * sizeof(float));
    (void)memset(context->hidden_bias_grad, 0, config->hidden_size * sizeof(float));
    (void)memset(context->hidden_to_output_grad, 0, hidden_to_output_count * sizeof(float));
    (void)memset(context->output_bias_grad, 0, config->output_size * sizeof(float));
}

/**
 * @brief Apply the accumulated gradients with a simple SGD-style update.
 */
static void rnn_apply_parameter_update(RnnTrainContext* context) {
    RnnInferContext* infer_ctx;
    const RnnConfig* config;
    size_t input_to_hidden_count;
    size_t hidden_to_hidden_count;
    size_t hidden_to_output_count;
    size_t value_index;

    infer_ctx = context->infer_ctx;
    config = &infer_ctx->config;
    input_to_hidden_count = config->hidden_size * config->input_feature_size;
    hidden_to_hidden_count = config->hidden_size * config->hidden_size;
    hidden_to_output_count = config->output_size * config->hidden_size;

    for (value_index = 0U; value_index < input_to_hidden_count; ++value_index) {
        infer_ctx->input_to_hidden[value_index] -= context->config.learning_rate * (
            context->input_to_hidden_grad[value_index] +
            (context->config.weight_decay * infer_ctx->input_to_hidden[value_index])
        );
    }
    for (value_index = 0U; value_index < hidden_to_hidden_count; ++value_index) {
        infer_ctx->hidden_to_hidden[value_index] -= context->config.learning_rate * (
            context->hidden_to_hidden_grad[value_index] +
            (context->config.weight_decay * infer_ctx->hidden_to_hidden[value_index])
        );
    }
    for (value_index = 0U; value_index < config->hidden_size; ++value_index) {
        infer_ctx->hidden_bias[value_index] -=
            context->config.learning_rate * context->hidden_bias_grad[value_index];
    }
    for (value_index = 0U; value_index < hidden_to_output_count; ++value_index) {
        infer_ctx->hidden_to_output[value_index] -= context->config.learning_rate * (
            context->hidden_to_output_grad[value_index] +
            (context->config.weight_decay * infer_ctx->hidden_to_output[value_index])
        );
    }
    for (value_index = 0U; value_index < config->output_size; ++value_index) {
        infer_ctx->output_bias[value_index] -=
            context->config.learning_rate * context->output_bias_grad[value_index];
    }
}

/**
 * @brief Backpropagate one externally supplied dL/dY through the recurrent leaf.
 */
static int rnn_backpropagate(
    RnnTrainContext* context,
    const float* input,
    const float* output_gradient,
    float* input_gradient
) {
    RnnInferContext* infer_ctx;
    const RnnConfig* config;
    float dh_next[64U];
    size_t output_index;
    size_t hidden_index;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || output_gradient == NULL) {
        return -1;
    }

    infer_ctx = context->infer_ctx;
    config = &infer_ctx->config;
    rnn_zero_gradients(context);
    (void)memset(dh_next, 0, sizeof(dh_next));

    if (input_gradient != NULL) {
        (void)memset(
            input_gradient,
            0,
            (config->sequence_length * config->input_feature_size) * sizeof(float)
        );
    }

    /* Stage 1: backpropagate from control outputs to the final hidden state. */
    for (output_index = 0U; output_index < config->output_size; ++output_index) {
        float output_value = infer_ctx->output_buffer[output_index];
        float dz = output_gradient[output_index] *
            rnn_activation_derivative_from_output(output_value, config->output_activation);
        size_t source_index;

        context->output_bias_grad[output_index] += dz;
        for (source_index = 0U; source_index < config->hidden_size; ++source_index) {
            float last_hidden = context->hidden_cache[
                ((config->sequence_length - 1U) * config->hidden_size) + source_index
            ];
            context->hidden_to_output_grad[
                (output_index * config->hidden_size) + source_index
            ] += dz * last_hidden;
            dh_next[source_index] += infer_ctx->hidden_to_output[
                (output_index * config->hidden_size) + source_index
            ] * dz;
        }
    }

    /* Stage 2: unroll the recurrent state backward through time. */
    for (hidden_index = config->sequence_length; hidden_index > 0U; --hidden_index) {
        size_t step_index = hidden_index - 1U;
        const float* step_input = input + (step_index * config->input_feature_size);
        const float* current_hidden = context->hidden_cache + (step_index * config->hidden_size);
        const float* previous_hidden = step_index == 0U ?
            NULL :
            (context->hidden_cache + ((step_index - 1U) * config->hidden_size));
        float dh_current[64U];
        size_t current_index;

        (void)memcpy(dh_current, dh_next, config->hidden_size * sizeof(float));
        (void)memset(dh_next, 0, config->hidden_size * sizeof(float));

        for (current_index = 0U; current_index < config->hidden_size; ++current_index) {
            float dz = dh_current[current_index] *
                rnn_activation_derivative_from_output(
                    current_hidden[current_index],
                    config->hidden_activation
                );
            size_t input_index;
            size_t recurrent_index;

            context->hidden_bias_grad[current_index] += dz;
            for (input_index = 0U; input_index < config->input_feature_size; ++input_index) {
                size_t weight_index = (current_index * config->input_feature_size) + input_index;
                context->input_to_hidden_grad[weight_index] += dz * step_input[input_index];
                if (input_gradient != NULL) {
                    input_gradient[(step_index * config->input_feature_size) + input_index] +=
                        infer_ctx->input_to_hidden[weight_index] * dz;
                }
            }
            for (recurrent_index = 0U; recurrent_index < config->hidden_size; ++recurrent_index) {
                float prev_value = previous_hidden != NULL ? previous_hidden[recurrent_index] : 0.0f;
                size_t weight_index = (current_index * config->hidden_size) + recurrent_index;
                context->hidden_to_hidden_grad[weight_index] += dz * prev_value;
                dh_next[recurrent_index] += infer_ctx->hidden_to_hidden[weight_index] * dz;
            }
        }
    }

    rnn_apply_parameter_update(context);
    return 0;
}

/**
 * @brief Create one RNN training context wrapped around an existing infer context.
 */
RnnTrainContext* nn_rnn_train_create(void* infer_ctx_ptr, const RnnTrainConfig* config) {
    RnnTrainContext* context;
    RnnInferContext* infer_ctx = (RnnInferContext*)infer_ctx_ptr;
    const RnnConfig* infer_config;
    size_t input_to_hidden_count;
    size_t hidden_to_hidden_count;
    size_t hidden_to_output_count;

    if (infer_ctx == NULL || config == NULL) {
        return NULL;
    }

    infer_config = &infer_ctx->config;
    input_to_hidden_count = infer_config->hidden_size * infer_config->input_feature_size;
    hidden_to_hidden_count = infer_config->hidden_size * infer_config->hidden_size;
    hidden_to_output_count = infer_config->output_size * infer_config->hidden_size;

    context = (RnnTrainContext*)calloc(1U, sizeof(RnnTrainContext));
    if (context == NULL) {
        return NULL;
    }

    context->infer_ctx = infer_ctx;
    context->config = *config;
    context->hidden_cache = (float*)calloc(
        infer_config->sequence_length * infer_config->hidden_size,
        sizeof(float)
    );
    context->output_linear_cache = (float*)calloc(infer_config->output_size, sizeof(float));
    context->input_to_hidden_grad = (float*)calloc(input_to_hidden_count, sizeof(float));
    context->hidden_to_hidden_grad = (float*)calloc(hidden_to_hidden_count, sizeof(float));
    context->hidden_bias_grad = (float*)calloc(infer_config->hidden_size, sizeof(float));
    context->hidden_to_output_grad = (float*)calloc(hidden_to_output_count, sizeof(float));
    context->output_bias_grad = (float*)calloc(infer_config->output_size, sizeof(float));

    if (context->hidden_cache == NULL || context->output_linear_cache == NULL ||
        context->input_to_hidden_grad == NULL || context->hidden_to_hidden_grad == NULL ||
        context->hidden_bias_grad == NULL || context->hidden_to_output_grad == NULL ||
        context->output_bias_grad == NULL) {
        nn_rnn_train_destroy(context);
        return NULL;
    }

    return context;
}

/**
 * @brief Free every training-side scratch buffer owned by the RNN trainer.
 */
void nn_rnn_train_destroy(RnnTrainContext* context) {
    if (context == NULL) {
        return;
    }

    free(context->hidden_cache);
    free(context->output_linear_cache);
    free(context->input_to_hidden_grad);
    free(context->hidden_to_hidden_grad);
    free(context->hidden_bias_grad);
    free(context->hidden_to_output_grad);
    free(context->output_bias_grad);
    free(context);
}

/**
 * @brief Run one graph-mode update using a caller-supplied output gradient.
 */
int nn_rnn_train_step_with_output_gradient(
    RnnTrainContext* context,
    const float* input,
    const float* output_gradient,
    float* input_gradient
) {
    RnnInferContext* infer_ctx;
    int rc;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || output_gradient == NULL) {
        return -1;
    }

    infer_ctx = context->infer_ctx;
    rc = nn_rnn_forward_pass(
        infer_ctx,
        input,
        infer_ctx->output_buffer,
        context->hidden_cache,
        context->output_linear_cache
    );
    if (rc != 0) {
        return rc;
    }

    rc = rnn_backpropagate(context, input, output_gradient, input_gradient);
    if (rc != 0) {
        return rc;
    }

    context->total_steps += 1U;
    return 0;
}

/**
 * @brief Run one supervised update with an explicit target tensor.
 */
int nn_rnn_train_step_with_data(RnnTrainContext* context, const float* input, const float* target) {
    RnnInferContext* infer_ctx;
    const RnnConfig* config;
    float output_gradient[64U];
    float loss = 0.0f;
    size_t output_index;
    int rc;

    if (context == NULL || context->infer_ctx == NULL || input == NULL || target == NULL) {
        return -1;
    }

    infer_ctx = context->infer_ctx;
    config = &infer_ctx->config;
    rc = nn_rnn_forward_pass(
        infer_ctx,
        input,
        infer_ctx->output_buffer,
        context->hidden_cache,
        context->output_linear_cache
    );
    if (rc != 0) {
        return rc;
    }

    /* The standalone supervised path uses simple MSE to stay transparent. */
    for (output_index = 0U; output_index < config->output_size; ++output_index) {
        float diff = infer_ctx->output_buffer[output_index] - target[output_index];
        loss += diff * diff;
        output_gradient[output_index] = (2.0f * diff) / (float)config->output_size;
    }

    rc = rnn_backpropagate(context, input, output_gradient, NULL);
    if (rc != 0) {
        return rc;
    }

    context->total_steps += 1U;
    context->last_loss = loss / (float)config->output_size;
    context->cumulative_loss += context->last_loss;
    context->average_loss = context->cumulative_loss / (float)context->total_steps;
    return 0;
}

/**
 * @brief Report coarse training statistics consumed by generated wrappers.
 */
void nn_rnn_train_get_stats(
    RnnTrainContext* context,
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
int nn_rnn_train_step(void* ctx) {
    RnnTrainContext* context = (RnnTrainContext*)ctx;
    float* dummy_input;
    float* dummy_target;
    int rc;

    if (context == NULL || context->infer_ctx == NULL) {
        return -1;
    }

    dummy_input = (float*)calloc(
        context->infer_ctx->config.sequence_length * context->infer_ctx->config.input_feature_size,
        sizeof(float)
    );
    dummy_target = (float*)calloc(context->infer_ctx->config.output_size, sizeof(float));
    if (dummy_input == NULL || dummy_target == NULL) {
        free(dummy_input);
        free(dummy_target);
        return -1;
    }

    rc = nn_rnn_train_step_with_data(context, dummy_input, dummy_target);
    free(dummy_input);
    free(dummy_target);
    return rc;
}
