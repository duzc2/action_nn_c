#include "nn_infer_registry.h"
#include "mlp_infer_ops.h"

const NNInferRegistryEntry nn_type_mlp_infer_entry = {
    "mlp",
    nn_mlp_infer_step
};
