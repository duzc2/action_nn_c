/**
 * @file nn_train_registry.h
 * @brief Compile-time registry for enabled training backends.
 *
 * The training registry mirrors the inference registry so the profiler can
 * validate and generate both phases through one consistent mechanism.
 */

#ifndef NN_TRAIN_REGISTRY_H
#define NN_TRAIN_REGISTRY_H

#include "nn_codegen_hooks.h"

#include <stddef.h>

/**
 * @brief Simplest training execution hook used by runtime dispatch.
 */
typedef int (*NNTrainStepFn)(void* context);

/**
 * @brief Full training backend entry stored in the registry.
 */
typedef struct {
    const char* type_name;                               /**< Semantic network type name. */
    NNTrainStepFn train_step;                            /**< Minimal single-step training entry. */
    NNTrainCreateFn create;                              /**< Factory used by generated train.c. */
    NNTrainDestroyFn destroy;                            /**< Cleanup hook paired with @ref create. */
    NNTrainStepWithDataFn step_with_data;                /**< Optional standalone data-driven step. */
    NNTrainStepWithOutputGradientFn step_with_output_gradient; /**< Optional graph backprop hook. */
    NNTrainGetStatsFn get_stats;                         /**< Optional statistics accessor. */
} NNTrainRegistryEntry;

int nn_train_registry_register(const NNTrainRegistryEntry* entry);
int nn_train_registry_get(const char* type_name, NNTrainStepFn* out_train_step);
const NNTrainRegistryEntry* nn_train_registry_find_entry(const char* type_name);
int nn_train_registry_is_registered(const char* type_name);
int nn_train_registry_clear(void);
int nn_train_registry_bootstrap(void);

/**
 * @brief Return the build-generated builtin training entry array.
 */
const NNTrainRegistryEntry* const* nn_train_registry_builtin_entries(size_t* out_count);

#endif
