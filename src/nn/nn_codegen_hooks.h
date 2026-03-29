/**
 * @file nn_codegen_hooks.h
 * @brief Shared hook contracts between profiler-generated code and NN backends.
 *
 * The profiler is required to stay generic: it must describe a network using
 * stable public structs and then bind that description to concrete backends
 * through registration. These hook typedefs are the narrow seam between the
 * generated orchestration code and the hand-written network implementations.
 */

#ifndef NN_CODEGEN_HOOKS_H
#define NN_CODEGEN_HOOKS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @brief Canonical configuration passed to inference backends created by codegen.
 *
 * The generated modules normalize graph metadata into a backend-agnostic shape.
 * Each concrete implementation may additionally interpret @ref type_config as a
 * type-specific configuration blob whose C type is described by
 * @ref type_config_type_name.
 */
typedef struct {
    const char* network_type;           /**< Semantic type name resolved via registry. */
    size_t input_size;                  /**< Executable input width for this leaf subnet. */
    size_t hidden_layer_count;          /**< Number of hidden layers materialized in config. */
    const size_t* hidden_sizes;         /**< Optional caller-owned hidden-width list, or NULL. */
    size_t output_size;                 /**< Executable output width for this leaf subnet. */
    uint64_t network_hash;              /**< Hash guarding cross-network weight reuse. */
    uint64_t layout_hash;               /**< Hash guarding parameter layout compatibility. */
    uint32_t seed;                      /**< Deterministic seed for reproducible init paths. */
    const void* type_config;            /**< Opaque type-specific inference configuration blob. */
    size_t type_config_size;            /**< Size of @ref type_config in bytes. */
    const char* type_config_type_name;  /**< Concrete C type reconstructed in generated code. */
} NNCodegenInferConfig;

/**
 * @brief Canonical configuration passed to training backends created by codegen.
 *
 * Training keeps optimizer metadata separate from inference-only structural
 * metadata because generated train.c composes a training backend around an
 * already-created inference context.
 */
typedef struct {
    float learning_rate;                /**< Primary optimizer step size. */
    float momentum;                     /**< Momentum term used by SGD-like optimizers. */
    float weight_decay;                 /**< Explicit regularization factor. */
    size_t batch_size;                  /**< Batch size chosen by generated training loops. */
    uint32_t seed;                      /**< Deterministic seed for reproducible training state. */
    const void* type_config;            /**< Opaque type-specific training configuration blob. */
    size_t type_config_size;            /**< Size of @ref type_config in bytes. */
    const char* type_config_type_name;  /**< Concrete C type reconstructed in generated code. */
} NNCodegenTrainConfig;

/* Inference hook set exposed by each enabled backend. */
typedef void* (*NNInferCreateFn)(const NNCodegenInferConfig* config);
typedef void (*NNInferDestroyFn)(void* context);
typedef int (*NNInferAutoRunFn)(void* context, const void* input, void* output);
typedef int (*NNInferGraphRunFn)(void* context, const void* input, void* output);
typedef int (*NNInferLoadWeightsFn)(void* context, FILE* fp);
typedef int (*NNInferSaveWeightsFn)(void* context, FILE* fp);

/* Training hook set exposed by each enabled backend. */
typedef void* (*NNTrainCreateFn)(void* infer_ctx, const NNCodegenTrainConfig* config);
typedef void (*NNTrainDestroyFn)(void* context);
typedef int (*NNTrainStepWithDataFn)(void* context, const void* input, const void* target);
typedef int (*NNTrainStepWithOutputGradientFn)(
    void* context,
    const void* input,
    const void* output_gradient,
    void* input_gradient
);
typedef void (*NNTrainGetStatsFn)(void* context, size_t* out_epochs, size_t* out_steps, float* out_avg_loss);

#endif
