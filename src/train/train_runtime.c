#include "train_runtime.h"

#include "nn_train_registry.h"

int nn_train_runtime_step(const NNTrainRequest* request) {
    NNTrainStepFn step = 0;
    if (request == 0 || request->network_type == 0) {
        return -1;
    }
    if (nn_train_registry_bootstrap() != 0) {
        return -2;
    }
    if (nn_train_registry_get(request->network_type, &step) != 0) {
        return -3;
    }
    return step(request->context);
}
