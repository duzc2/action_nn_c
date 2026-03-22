#include "infer_runtime.h"

#include "nn_infer_registry.h"

int nn_infer_runtime_step(const NNInferRequest* request) {
    NNInferStepFn step = 0;
    if (request == 0 || request->network_type == 0) {
        return -1;
    }
    if (nn_infer_registry_bootstrap() != 0) {
        return -2;
    }
    if (nn_infer_registry_get(request->network_type, &step) != 0) {
        return -3;
    }
    return step(request->context);
}
