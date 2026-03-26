/**
 * @file nn_graph_contract.h
 * @brief Cached graph-execution contracts derived from registry entries.
 *
 * Validation and code generation need a slightly richer view than the raw
 * registries provide. In particular, they must answer questions such as
 * "does this type support graph execution?" and "can this type propagate
 * gradients through a composed graph?". These structs cache those derived
 * capabilities beside the original create/destroy hooks.
 */

#ifndef NN_GRAPH_CONTRACT_H
#define NN_GRAPH_CONTRACT_H

#include "nn_codegen_hooks.h"

/**
 * @brief Inference-side graph contract for one semantic network type.
 */
typedef struct {
    const char* type_name;              /**< Semantic network type name. */
    NNInferCreateFn create;             /**< Factory used by generated runtime init code. */
    NNInferDestroyFn destroy;           /**< Cleanup hook paired with @ref create. */
    NNInferAutoRunFn auto_run;          /**< Optional self-contained inference loop. */
    NNInferGraphRunFn graph_run;        /**< Optional leaf-level graph execution entry. */
    NNInferLoadWeightsFn load_weights;  /**< Backend-specific weight deserializer. */
    NNInferSaveWeightsFn save_weights;  /**< Backend-specific weight serializer. */
    int supports_graph_mode;            /**< Non-zero only when @ref graph_run is available. */
} NNGraphInferContract;

/**
 * @brief Training-side graph contract for one semantic network type.
 */
typedef struct {
    const char* type_name;                              /**< Semantic network type name. */
    NNTrainCreateFn create;                             /**< Factory that wraps an inference context. */
    NNTrainDestroyFn destroy;                           /**< Cleanup hook paired with @ref create. */
    NNTrainStepWithDataFn step_with_data;               /**< Optional standalone data-driven step. */
    NNTrainStepWithOutputGradientFn step_with_output_gradient; /**< Graph backprop entry point. */
    NNTrainGetStatsFn get_stats;                        /**< Optional training statistics accessor. */
    int supports_graph_backprop;                        /**< Non-zero only when gradient backprop is supported. */
} NNGraphTrainContract;

/**
 * @brief Resolve and cache an inference graph contract by semantic type name.
 */
const NNGraphInferContract* nn_graph_infer_contract_find(const char* type_name);

/**
 * @brief Return non-zero when the type can execute as a graph leaf node.
 */
int nn_graph_infer_contract_supports_graph_mode(const char* type_name);

/**
 * @brief Resolve and cache a training graph contract by semantic type name.
 */
const NNGraphTrainContract* nn_graph_train_contract_find(const char* type_name);

/**
 * @brief Return non-zero when the type can backpropagate through graph edges.
 */
int nn_graph_train_contract_supports_backprop(const char* type_name);

#endif
