/**
 * @file train_runtime.h
 * @brief Lightweight runtime adapter for training dispatch.
 *
 * Training uses the same registry-driven design as inference: the caller only
 * names the enabled network type and provides an opaque training context. The
 * runtime then forwards the request to the compiled training backend without
 * exposing type-specific headers to the rest of the system.
 */

#ifndef TRAIN_RUNTIME_H
#define TRAIN_RUNTIME_H

/**
 * @brief Opaque request used to dispatch a single training step.
 *
 * @ref network_type selects the registry entry, while @ref context carries the
 * backend-owned state that was created by the matching training implementation.
 */
typedef struct {
    const char* network_type;  /**< Semantic network type such as "mlp". */
    void* context;             /**< Backend-owned training context instance. */
} NNTrainRequest;

/**
 * @brief Execute one training step through the training registry.
 *
 * @param request Dispatch request containing type name and opaque context
 * @return 0 on success, negative value when validation or dispatch fails
 */
int nn_train_runtime_step(const NNTrainRequest* request);

#endif
