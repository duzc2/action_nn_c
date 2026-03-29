/**
 * @file gnn_config.h
 * @brief POD-style configuration shared by the GNN backend and generated code.
 *
 * The profiler only needs plain-old-data metadata, so this header keeps the
 * topology description serializable and detached from backend implementation
 * details while allowing user-provided graph sizes and slot layouts.
 */

#ifndef GNN_CONFIG_H
#define GNN_CONFIG_H

#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>

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
 * @brief Variable-sized structural configuration required to build one GNN leaf.
 *
 * The neighbor table is stored as one flattened trailing array with
 * node_count * slot_count entries so the profiler can copy one self-contained
 * blob without imposing builtin topology limits.
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
    int neighbor_index[];                                  /**< Flattened [node_count * slot_count] neighbor table. */
} GnnConfig;

/**
 * @brief Return the exact byte size required for one GNN config blob.
 */
static inline size_t gnn_config_size_for_topology(size_t node_count, size_t slot_count) {
    return sizeof(GnnConfig) + (node_count * slot_count * sizeof(int));
}

/**
 * @brief Return a writable pointer to the flattened neighbor table.
 */
static inline int* gnn_config_neighbors_mut(GnnConfig* config) {
    return config == NULL ? NULL : config->neighbor_index;
}

/**
 * @brief Return a read-only pointer to the flattened neighbor table.
 */
static inline const int* gnn_config_neighbors_view(const GnnConfig* config) {
    return config == NULL ? NULL : config->neighbor_index;
}

/**
 * @brief Return a writable pointer to one logical neighbor row.
 */
static inline int* gnn_config_neighbor_row_mut(GnnConfig* config, size_t node_index) {
    if (config == NULL || node_index >= config->node_count) {
        return NULL;
    }
    return config->neighbor_index + (node_index * config->slot_count);
}

/**
 * @brief Return a read-only pointer to one logical neighbor row.
 */
static inline const int* gnn_config_neighbor_row_view(const GnnConfig* config, size_t node_index) {
    if (config == NULL || node_index >= config->node_count) {
        return NULL;
    }
    return config->neighbor_index + (node_index * config->slot_count);
}

/**
 * @brief Allocate one profiler-ready GNN config blob on the heap.
 */
static inline GnnConfig* gnn_config_create(size_t node_count, size_t slot_count) {
    size_t total_size = gnn_config_size_for_topology(node_count, slot_count);
    return (GnnConfig*)calloc(1U, total_size);
}

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
