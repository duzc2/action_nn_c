#include "nn_train_registry.h"
#include "mlp_train_ops.h"

const NNTrainRegistryEntry nn_type_mlp_train_entry = {
    "mlp",
    nn_mlp_train_step
};
