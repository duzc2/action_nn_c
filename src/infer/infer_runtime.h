/**
 * @file infer_runtime.h
 * @brief Inference runtime interface
 *
 * The inference library is standalone and does not contain any training code.
 * User code executes inference via nn_infer_runtime_step().
 */

#ifndef INFER_RUNTIME_H
#define INFER_RUNTIME_H

/**
 * @brief Inference request structure
 *
 * Contains:
 * - network_type: Network type name (e.g., "mlp", "transformer")
 * - context: Network context, created by xxx_create()
 *
 * Note: Users do not need to care about specific network implementation,
 * only need to look up the corresponding inference function via registry.
 */
typedef struct {
    const char* network_type;  /**< Network type name */
    void* context;             /**< Network context */
} NNInferRequest;

/**
 * @brief Execute one step of inference
 *
 * Looks up the inference function for the corresponding type via network registry,
 * and calls that function to execute inference.
 *
 * @param request Inference request
 * @return 0 success, non-zero failure
 */
int nn_infer_runtime_step(const NNInferRequest* request);

#endif
