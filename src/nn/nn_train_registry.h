#ifndef NN_TRAIN_REGISTRY_H
#define NN_TRAIN_REGISTRY_H

#include "nn_codegen_hooks.h"

#include <stddef.h>

typedef int (*NNTrainStepFn)(void* context);

typedef struct {
    const char* type_name;
    NNTrainStepFn train_step;
    NNTrainCreateFn create;
    NNTrainDestroyFn destroy;
    NNTrainStepWithDataFn step_with_data;
    NNTrainStepWithOutputGradientFn step_with_output_gradient;
    NNTrainGetStatsFn get_stats;
} NNTrainRegistryEntry;

int nn_train_registry_register(const NNTrainRegistryEntry* entry);
int nn_train_registry_get(const char* type_name, NNTrainStepFn* out_train_step);
const NNTrainRegistryEntry* nn_train_registry_find_entry(const char* type_name);
int nn_train_registry_is_registered(const char* type_name);
int nn_train_registry_clear(void);
int nn_train_registry_bootstrap(void);

const NNTrainRegistryEntry* const* nn_train_registry_builtin_entries(size_t* out_count);

#endif
