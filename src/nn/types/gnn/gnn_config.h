/**
 * @file gnn_config.h
 * @brief POD-style configuration shared by the tiny GNN backend and generated code.
 *
 * The first GNN backend in this repository is intentionally compact. It assumes
 * one fixed-size graph with a bounded node count and up to four directed slots
 * per node. The profiler only needs plain-old-data metadata, so this header
 * keeps the topology description serializable and detached from the backend
 * implementation details.
 */

#ifndef GNN_CONFIG_H
#define GNN_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#define GNN_MAX_NODE_COUNT 64U
#define GNN_MAX_NODE_FEATURE_SIZE 8U
#define GNN_MAX_HIDDEN_SIZE 32U
#define GNN_MAX_OUTPUT_SIZE 16U
#define GNN_MAX_SLOT_COUNT 4U
#define GNN_MAX_MESSAGE_PASSES 4U
#define GNN_FEATURE_INDEX_NONE ((size_t)(~(size_t)0))

/**
 * @brief Activation types used by the tiny GNN backend.
 */
typedef enum {
    GNN_ACT_NONE = 0,
    GNN_ACT_RELU = 1,
    GNN_ACT_TANH = 2
} GnnActivationType;

/**
 * @brief Neighbor aggregation rule used during message passing.
 */
typedef enum {
    GNN_AGG_MEAN = 0
} GnnAggregatorType;

/**
 * @brief Readout strategy used to export one graph-level output vector.
 */
typedef enum {
    GNN_READOUT_GRAPH_POOL = 0,
    GNN_READOUT_ANCHOR_SLOTS = 1
} GnnReadoutType;

/**
 * @brief Structural configuration required to build one GNN leaf.
 */
typedef struct {
    size_t node_count;                                     /**< Number of graph nodes packed into one input. */
    size_t node_feature_size;                              /**< Feature width carried by each node. */
    size_t hidden_size;                                    /**< Width of the per-node hidden state. */
    size_t output_size;                                    /**< Width of the readout vector exported to the graph. */
    size_t message_passes;                                 /**< Number of message-passing rounds. */
    size_t slot_count;                                     /**< Slot count used by slot-oriented readout modes. */
    size_t node_mask_feature_index;                        /**< Optional node-activity feature index, or @ref GNN_FEATURE_INDEX_NONE. */
    size_t primary_anchor_feature_index;                   /**< Optional primary anchor selector feature for readout. */
    size_t secondary_anchor_feature_index;                 /**< Optional secondary anchor selector feature for readout. */
    GnnAggregatorType aggregator_type;                     /**< Neighbor aggregation rule used in message passing. */
    GnnReadoutType readout_type;                           /**< Strategy used to export the graph output vector. */
    GnnActivationType hidden_activation;                   /**< Activation used by node-state updates. */
    GnnActivationType output_activation;                   /**< Activation used by the exported readout vector. */
    uint32_t seed;                                         /**< Deterministic seed for reproducible init. */
    int neighbor_index[GNN_MAX_NODE_COUNT][GNN_MAX_SLOT_COUNT]; /**< Fixed neighbor table stored in user-defined slot order. */
} GnnConfig;

/**
 * @brief Minimal train-time hyperparameters for the tiny GNN backend.
 */
typedef struct {
    float learning_rate;                                   /**< Step size used by the simple SGD update. */
    float momentum;                                        /**< Reserved for future optimizer growth. */
    float weight_decay;                                    /**< L2-style decay applied to trainable weights. */
    size_t batch_size;                                     /**< Batch size requested by generated wrappers. */
    uint32_t seed;                                         /**< Reserved deterministic seed for future train state. */
} GnnTrainConfig;

#endif
