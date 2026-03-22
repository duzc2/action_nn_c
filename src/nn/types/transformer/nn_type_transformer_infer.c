#include "nn_infer_registry.h"
#include "transformer_infer_ops.h"

const NNInferRegistryEntry nn_type_transformer_infer_entry = {
    "transformer",
    nn_transformer_infer_step
};
