/**
 * @file nn_infer_registry.h
 * @brief Static registry for enabled inference backends.
 *
 * Stores entries both as legacy NNInferRegistryEntry (for auto-generated code)
 * and as VTable-based NNInferBackend (for new dispatch code).
 */

#ifndef NN_INFER_REGISTRY_H
#define NN_INFER_REGISTRY_H

#include "nn_backend.h"
#include "nn_codegen_hooks.h"

#include <stddef.h>

#define NN_INFER_MAX_BACKENDS 32

/* ─── Forward-compatible legacy type ─── */
/**
 * @brief Simplest inference execution hook used by runtime dispatch.
 */
typedef int (*NNInferStepFn)(void* context);

/**
 * @brief Legacy inference backend entry stored in the registry.
 *
 * Kept for auto-generated code compatibility. Will be replaced by
 * NNInferBackend VTable in a future step.
 */
typedef struct {
    const char* type_name;
    NNInferStepFn infer_step;
    NNInferCreateFn create;
    NNInferDestroyFn destroy;
    NNInferAutoRunFn auto_run;
    NNInferGraphRunFn graph_run;
    NNInferLoadWeightsFn load_weights;
    NNInferSaveWeightsFn save_weights;
} NNInferRegistryEntry;

/* ─── Legacy entry API (auto-generated code depends on this) ─── */
int nn_infer_registry_register(const NNInferRegistryEntry* entry);
int nn_infer_registry_get(const char* type_name, NNInferStepFn* out_infer_step);
const NNInferRegistryEntry* nn_infer_registry_find_entry(const char* type_name);
int nn_infer_registry_is_registered(const char* type_name);
int nn_infer_registry_clear(void);
int nn_infer_registry_bootstrap(void);
const NNInferRegistryEntry* const* nn_infer_registry_builtin_entries(size_t* out_count);

/* ─── VTable-based API (new dispatch) ─── */
int nn_infer_vtable_register(const NNInferBackend* backend);
const NNInferBackend* nn_infer_vtable_find(const char* type_name);
size_t nn_infer_vtable_count(void);
const NNInferBackend* nn_infer_vtable_get(size_t index);

#endif
