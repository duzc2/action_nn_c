/**
 * @file train_runtime.c
 * @brief Thin adapter that routes training work through the registry layer.
 *
 * The training runtime intentionally mirrors infer_runtime.c so the generated
 * pipeline can treat both phases symmetrically. That symmetry matters because
 * the documentation requires training and inference to stay decoupled while
 * still sharing the same compile-time registration mechanism.
 */

#include "train_runtime.h"

#include "nn_train_registry.h"

/**
 * @brief Dispatch one training step to the selected backend.
 *
 * Negative return values identify which stage failed so callers can stop at
 * the first error without adding backend-specific branches in the runtime.
 */
int nn_train_runtime_step(const NNTrainRequest* request) {
    NNTrainStepFn step = 0;

    /* Validate the dispatch envelope before touching the global registry. */
    if (request == 0 || request->network_type == 0) {
        return -1;
    }

    /* Ensure all enabled training backends have been registered once. */
    if (nn_train_registry_bootstrap() != 0) {
        return -2;
    }

    /* Resolve the semantic type name into a concrete training callback. */
    if (nn_train_registry_get(request->network_type, &step) != 0) {
        return -3;
    }

    /* Forward the opaque context directly to the type-specific backend. */
    return step(request->context);
}
