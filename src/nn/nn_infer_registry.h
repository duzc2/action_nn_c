#ifndef NN_INFER_REGISTRY_H
#define NN_INFER_REGISTRY_H

#include <stddef.h>

typedef int (*NNInferStepFn)(void* context);

typedef struct {
    const char* type_name;
    NNInferStepFn infer_step;
} NNInferRegistryEntry;

int nn_infer_registry_register(const char* type_name, NNInferStepFn infer_step);
int nn_infer_registry_get(const char* type_name, NNInferStepFn* out_infer_step);
int nn_infer_registry_is_registered(const char* type_name);
int nn_infer_registry_clear(void);
int nn_infer_registry_bootstrap(void);

const NNInferRegistryEntry* const* nn_infer_registry_builtin_entries(size_t* out_count);

#endif
