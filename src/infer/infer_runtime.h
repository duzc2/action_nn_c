/**
 * @file infer_runtime.h
 * @brief Lightweight runtime adapter for inference-only execution.
 *
 * This header deliberately exposes a tiny surface area because demos and
 * profiler-generated code are expected to drive all concrete network types
 * through the registry layer instead of hard-coding type-specific entry points.
 * The adapter therefore only needs enough information to:
 * - name the enabled network type,
 * - carry the opaque runtime context created by that type,
 * - dispatch one inference step through the registry.
 *
 * Keeping this interface stable helps generated code remain independent from
 * the concrete implementation details that live under src/nn/types/.
 */

#ifndef INFER_RUNTIME_H
#define INFER_RUNTIME_H

/**
 * @brief Opaque dispatch request for one inference invocation.
 *
 * The request keeps only the minimum routing metadata. The registry uses
 * @ref network_type to resolve the compiled-in implementation, while
 * @ref context is passed through untouched so each backend can own its
 * memory layout and internal state machine.
 */
typedef struct {
    const char* network_type;  /**< Semantic network type name such as "mlp". */
    void* context;             /**< Backend-owned inference context instance. */
} NNInferRequest;

/**
 * @brief Execute exactly one inference step through the registry.
 *
 * The runtime does not interpret model buffers, tensors, or scheduling policy.
 * Its job is only to validate the dispatch request, ensure builtin registry
 * entries are bootstrapped, and then forward control to the type-specific
 * implementation that was enabled at configure time.
 *
 * @param request Inference request
 * @return 0 on success, negative value when validation or dispatch fails
 */
int nn_infer_runtime_step(const NNInferRequest* request);

#endif
