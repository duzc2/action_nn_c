#ifndef NETWORK_SPEC_H
#define NETWORK_SPEC_H

#include <stddef.h>

#include "config_user.h"

#define NETWORK_SPEC_MAX_LAYERS 32U
#define NETWORK_GRAPH_MAX_NODES 64U
#define NETWORK_GRAPH_MAX_EDGES 128U

typedef enum NetworkLayerKind {
    NETWORK_LAYER_TOKEN_STATE_LINEAR = 0,
    NETWORK_LAYER_TRANSFORMER_BLOCK = 1,
    NETWORK_LAYER_ATTENTION_HEAD = 2,
    NETWORK_LAYER_CNN = 3,
    NETWORK_LAYER_RNN = 4,
    NETWORK_LAYER_KNN = 5
} NetworkLayerKind;

typedef struct NetworkLayerConfig {
    NetworkLayerKind kind;
} NetworkLayerConfig;

typedef enum NetworkNodeType {
    NETWORK_NODE_INPUT = 0,
    NETWORK_NODE_LINEAR = 1,
    NETWORK_NODE_TRANSFORMER_BLOCK = 2,
    NETWORK_NODE_ATTENTION_HEAD = 3,
    NETWORK_NODE_CNN = 4,
    NETWORK_NODE_RNN = 5,
    NETWORK_NODE_KNN = 6,
    NETWORK_NODE_SELECT = 7,
    NETWORK_NODE_MERGE = 8
} NetworkNodeType;

typedef struct NetworkGraphNode {
    int id;
    int type;
    int selector_offset;
    int selector_size;
} NetworkGraphNode;

typedef struct NetworkGraphEdge {
    int from_id;
    int to_id;
} NetworkGraphEdge;

typedef struct NetworkGraph {
    size_t node_count;
    NetworkGraphNode nodes[NETWORK_GRAPH_MAX_NODES];
    size_t edge_count;
    NetworkGraphEdge edges[NETWORK_GRAPH_MAX_EDGES];
    int input_node_id;
    int output_node_id;
} NetworkGraph;

typedef struct NetworkSpec {
    size_t layer_count;
    NetworkLayerConfig layers[NETWORK_SPEC_MAX_LAYERS];
} NetworkSpec;

int network_spec_build_from_graph(NetworkSpec* spec, const NetworkGraph* graph);
int network_graph_build(NetworkGraph* graph,
                        const NetworkGraphNode* nodes,
                        size_t node_count,
                        const NetworkGraphEdge* edges,
                        size_t edge_count,
                        int input_node_id,
                        int output_node_id);
int network_graph_validate(const NetworkGraph* graph);
int network_spec_validate(const NetworkSpec* spec);
size_t network_spec_weight_count(const NetworkSpec* spec);

#endif
