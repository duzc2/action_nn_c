#ifndef NN_TRAIN_REGISTRY_H
#define NN_TRAIN_REGISTRY_H

#include <stddef.h>

typedef int (*NNTrainStepFn)(void* context);

typedef struct {
    const char* type_name;
    NNTrainStepFn train_step;
} NNTrainRegistryEntry;

int nn_train_registry_register(const char* type_name, NNTrainStepFn train_step);
int nn_train_registry_get(const char* type_name, NNTrainStepFn* out_train_step);
int nn_train_registry_is_registered(const char* type_name);
int nn_train_registry_clear(void);
int nn_train_registry_bootstrap(void);

const NNTrainRegistryEntry* const* nn_train_registry_builtin_entries(size_t* out_count);

#endif
