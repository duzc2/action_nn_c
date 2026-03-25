#ifndef TRANSFORMER_CONFIG_H
#define TRANSFORMER_CONFIG_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t model_dim;
    uint32_t seed;
} TransformerModelConfig;

typedef struct {
    float learning_rate;
} TransformerTrainConfig;

#endif
