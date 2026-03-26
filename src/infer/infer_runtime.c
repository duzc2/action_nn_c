/**
 * @file infer_runtime.c
 * @brief Minimal inference dispatch adapter built on top of the registry.
 *
 * The implementation intentionally stays tiny: it validates the request, asks
 * the registry to materialize all compile-time enabled inference backends, and
 * forwards the opaque context to the resolved step function. This keeps the
 * runtime layer free from network-specific branching.
 */

#include "infer_runtime.h"

#include "nn_infer_registry.h"

/**
 * @brief Dispatch one inference step to the registered backend.
 *
 * Return codes are deliberately simple because the higher-level profiler flow
 * already enforces "first error stops" semantics. Each negative code therefore
 * points to a single failure stage:
 * - -1: invalid request metadata,
 * - -2: builtin registry bootstrap failed,
 * - -3: requested network type is unavailable.
 */
int nn_infer_runtime_step(const NNInferRequest* request) {
    NNInferStepFn step = 0;

    /* Reject incomplete dispatch requests before touching global state. */
    if (request == 0 || request->network_type == 0) {
        return -1;
    }

    /* Materialize the compile-time enabled inference hooks exactly once. */
    if (nn_infer_registry_bootstrap() != 0) {
        return -2;
    }

    /* Resolve the semantic type name into the concrete step callback. */
    if (nn_infer_registry_get(request->network_type, &step) != 0) {
        return -3;
    }

    /* Forward the opaque backend context without imposing extra policy. */
    return step(request->context);
}
