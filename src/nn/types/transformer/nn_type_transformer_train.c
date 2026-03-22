#include "nn_train_registry.h"
#include "transformer_train_ops.h"

const NNTrainRegistryEntry nn_type_transformer_train_entry = {
    "transformer",
    nn_transformer_train_step
};
