/**
 * @file nn_train_registry.h
 * @brief Static registry for enabled training backends.
 *
 * Mirror of nn_infer_registry.h. Keeps legacy entry types for auto-generated
 * code and adds VTable-based API for new dispatch.
 */

#ifndef NN_TRAIN_REGISTRY_H
#define NN_TRAIN_REGISTRY_H

#include "nn_backend.h"
#include "nn_codegen_hooks.h"

#include <stddef.h>

#define NN_TRAIN_MAX_BACKENDS 32

/* ─── Forward-compatible legacy type ─── */
typedef int (*NNTrainStepFn)(void* context);

/**
 * @brief Legacy training backend entry stored in the registry.
 */
typedef struct {
    const char* type_name;
    NNTrainStepFn train_step;
    NNTrainCreateFn create;
    NNTrainDestroyFn destroy;
    NNTrainStepWithDataFn step_with_data;
    NNTrainStepWithOutputGradientFn step_with_output_gradient;
    NNTrainGetStatsFn get_stats;
} NNTrainRegistryEntry;

/* ─── Legacy entry API (auto-generated code depends on this) ─── */
int nn_train_registry_register(const NNTrainRegistryEntry* entry);
int nn_train_registry_get(const char* type_name, NNTrainStepFn* out_train_step);
const NNTrainRegistryEntry* nn_train_registry_find_entry(const char* type_name);
int nn_train_registry_is_registered(const char* type_name);
int nn_train_registry_clear(void);
int nn_train_registry_bootstrap(void);
const NNTrainRegistryEntry* const* nn_train_registry_builtin_entries(size_t* out_count);

/* ─── VTable-based API (new dispatch) ─── */
int nn_train_vtable_register(const NNTrainBackend* backend);
const NNTrainBackend* nn_train_vtable_find(const char* type_name);
size_t nn_train_vtable_count(void);
const NNTrainBackend* nn_train_vtable_get(size_t index);

#endif
