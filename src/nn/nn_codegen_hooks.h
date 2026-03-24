#ifndef NN_CODEGEN_HOOKS_H
#define NN_CODEGEN_HOOKS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    const char* network_type;
    size_t input_size;
    size_t hidden_layer_count;
    size_t hidden_sizes[4];
    size_t output_size;
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t seed;
} NNCodegenInferConfig;

typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    size_t batch_size;
    uint32_t seed;
} NNCodegenTrainConfig;

typedef void* (*NNInferCreateFn)(const NNCodegenInferConfig* config);
typedef void (*NNInferDestroyFn)(void* context);
typedef int (*NNInferAutoRunFn)(void* context, const void* input, void* output);
typedef int (*NNInferLoadWeightsFn)(void* context, FILE* fp);
typedef int (*NNInferSaveWeightsFn)(void* context, FILE* fp);

typedef void* (*NNTrainCreateFn)(void* infer_ctx, const NNCodegenTrainConfig* config);
typedef void (*NNTrainDestroyFn)(void* context);
typedef int (*NNTrainStepWithDataFn)(void* context, const void* input, const void* target);
typedef void (*NNTrainGetStatsFn)(void* context, size_t* out_epochs, size_t* out_steps, float* out_avg_loss);

#endif
