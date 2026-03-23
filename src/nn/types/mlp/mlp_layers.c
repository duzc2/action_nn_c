/**
 * @file mlp_layers.c
 * @brief MLP layer mathematics implementation
 */

#include "mlp_layers.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LEAKY_RELU_ALPHA 0.01f

/**
 * @brief Simple pseudo-random number generator (PCG-like)
 *
 * Uses a simple linear congruential generator for reproducibility.
 */
static uint32_t pcg_rand(uint32_t* state) {
    uint64_t old_state = *state;
    *state = old_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xsh = (uint32_t)((old_state >> 28u) ^ old_state);
    uint32_t rot = (uint32_t)(old_state >> 59u);
    return (xsh >> rot) | (xsh << ((-rot) & 31));
}

static float pcg_rand_float(uint32_t* state) {
    return (float)(pcg_rand(state) >> 8) / 16777216.0f;
}

float mlp_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float mlp_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float mlp_tanh(float x) {
    return tanhf(x);
}

float mlp_leaky_relu(float x) {
    return x > 0.0f ? x : LEAKY_RELU_ALPHA * x;
}

ActivationFn mlp_get_activation(MlpActivationType act_type) {
    switch (act_type) {
        case MLP_ACT_RELU: return mlp_relu;
        case MLP_ACT_SIGMOID: return mlp_sigmoid;
        case MLP_ACT_TANH: return mlp_tanh;
        case MLP_ACT_LEAKY_RELU: return mlp_leaky_relu;
        default: return NULL;
    }
}

void mlp_activation(float* output, const float* input, size_t size, MlpActivationType act_type) {
    size_t i;

    if (output == NULL || input == NULL) {
        return;
    }

    switch (act_type) {
        case MLP_ACT_NONE:
            for (i = 0; i < size; i++) {
                output[i] = input[i];
            }
            break;

        case MLP_ACT_RELU:
            for (i = 0; i < size; i++) {
                output[i] = mlp_relu(input[i]);
            }
            break;

        case MLP_ACT_SIGMOID:
            for (i = 0; i < size; i++) {
                output[i] = mlp_sigmoid(input[i]);
            }
            break;

        case MLP_ACT_TANH:
            for (i = 0; i < size; i++) {
                output[i] = mlp_tanh(input[i]);
            }
            break;

        case MLP_ACT_LEAKY_RELU:
            for (i = 0; i < size; i++) {
                output[i] = mlp_leaky_relu(input[i]);
            }
            break;

        case MLP_ACT_SOFTMAX: {
            float max_val = input[0];
            float sum = 0.0f;
            size_t j;

            for (i = 1; i < size; i++) {
                if (input[i] > max_val) {
                    max_val = input[i];
                }
            }

            for (j = 0; j < size; j++) {
                output[j] = expf(input[j] - max_val);
                sum += output[j];
            }

            for (j = 0; j < size; j++) {
                output[j] /= sum;
            }
            break;
        }
    }
}

void mlp_dense_forward(const MlpDenseLayer* layer, float* output, const float* input) {
    size_t i;
    size_t j;

    if (layer == NULL || output == NULL || input == NULL) {
        return;
    }

    if (layer->weights == NULL || layer->bias == NULL) {
        for (i = 0; i < layer->output_size; i++) {
            output[i] = 0.0f;
        }
        return;
    }

    for (i = 0; i < layer->output_size; i++) {
        float sum = layer->bias[i];

        for (j = 0; j < layer->input_size; j++) {
            sum += layer->weights[i * layer->input_size + j] * input[j];
        }

        output[i] = sum;
    }

    if (layer->activation != MLP_ACT_NONE) {
        mlp_activation(output, output, layer->output_size, layer->activation);
    }
}

void mlp_dense_init(MlpDenseLayer* layer, uint32_t seed) {
    uint32_t state = seed;
    size_t total_params;
    float bound;
    size_t i;

    if (layer == NULL) {
        return;
    }

    if (layer->weights == NULL || layer->bias == NULL) {
        return;
    }

    total_params = layer->input_size + layer->output_size;
    if (total_params == 0) {
        return;
    }

    bound = sqrtf(6.0f / (float)total_params);

    for (i = 0; i < layer->input_size * layer->output_size; i++) {
        layer->weights[i] = (pcg_rand_float(&state) * 2.0f - 1.0f) * bound;
    }

    for (i = 0; i < layer->output_size; i++) {
        layer->bias[i] = 0.0f;
    }
}

MlpDenseLayer* mlp_dense_create(
    size_t input_size,
    size_t output_size,
    MlpActivationType activation,
    uint32_t seed
) {
    MlpDenseLayer* layer;

    if (input_size == 0 || output_size == 0) {
        return NULL;
    }

    layer = (MlpDenseLayer*)malloc(sizeof(MlpDenseLayer));
    if (layer == NULL) {
        return NULL;
    }

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;
    layer->weights = (float*)malloc(input_size * output_size * sizeof(float));
    layer->bias = (float*)malloc(output_size * sizeof(float));

    if (layer->weights == NULL || layer->bias == NULL) {
        free(layer->weights);
        free(layer->bias);
        free(layer);
        return NULL;
    }

    mlp_dense_init(layer, seed);

    return layer;
}

void mlp_dense_free(MlpDenseLayer* layer) {
    if (layer == NULL) {
        return;
    }

    free(layer->weights);
    free(layer->bias);
    free(layer);
}
