/**
 * @file nn_infer_registry.h
 * @brief Compile-time registry for enabled inference backends.
 *
 * Only CMake-enabled network types are allowed to appear here. The registry is
 * purposely simple and static so bootstrap stays deterministic and generated
 * code can assume that unavailable types were filtered out during compilation.
 */

#ifndef NN_INFER_REGISTRY_H
#define NN_INFER_REGISTRY_H

#include "nn_codegen_hooks.h"

#include <stddef.h>

/**
 * @brief Simplest inference execution hook used by runtime dispatch.
 */
typedef int (*NNInferStepFn)(void* context);

/**
 * @brief Full inference backend entry stored in the registry.
 *
 * The profiler and runtime use different subsets of these hooks. Keeping them
 * together in one entry lets validation and code generation derive richer graph
 * contracts without touching backend-specific headers directly.
 */
typedef struct {
    const char* type_name;            /**< Semantic network type name. */
    NNInferStepFn infer_step;         /**< Minimal single-step inference entry. */
    NNInferCreateFn create;           /**< Factory used by generated init code. */
    NNInferDestroyFn destroy;         /**< Cleanup hook paired with @ref create. */
    NNInferAutoRunFn auto_run;        /**< Optional autonomous inference loop. */
    NNInferGraphRunFn graph_run;      /**< Optional graph-leaf execution hook. */
    NNInferLoadWeightsFn load_weights;/**< Backend-specific weight loader. */
    NNInferSaveWeightsFn save_weights;/**< Backend-specific weight saver. */
} NNInferRegistryEntry;

int nn_infer_registry_register(const NNInferRegistryEntry* entry);
int nn_infer_registry_get(const char* type_name, NNInferStepFn* out_infer_step);
const NNInferRegistryEntry* nn_infer_registry_find_entry(const char* type_name);
int nn_infer_registry_is_registered(const char* type_name);
int nn_infer_registry_clear(void);
int nn_infer_registry_bootstrap(void);

/**
 * @brief Return the compile-time builtin entry array emitted by the build.
 */
const NNInferRegistryEntry* const* nn_infer_registry_builtin_entries(size_t* out_count);

#endif
