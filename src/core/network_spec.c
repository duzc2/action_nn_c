#include "../include/network_spec.h"

enum {
    TOKEN_WEIGHT_COUNT = VOCAB_SIZE * OUTPUT_DIM,
    STATE_WEIGHT_COUNT = STATE_DIM * OUTPUT_DIM,
    BIAS_COUNT = OUTPUT_DIM
};

static int network_graph_find_node_index(const NetworkGraph* graph, int node_id) {
    size_t i = 0U;
    if (graph == NULL) {
        return -1;
    }
    for (i = 0U; i < graph->node_count; ++i) {
        if (graph->nodes[i].id == node_id) {
            return (int)i;
        }
    }
    return -1;
}

static int network_graph_toposort(const NetworkGraph* graph, size_t* out_order, size_t* out_count) {
    size_t indegree[NETWORK_GRAPH_MAX_NODES];
    size_t queue[NETWORK_GRAPH_MAX_NODES];
    size_t q_head = 0U;
    size_t q_tail = 0U;
    size_t order_count = 0U;
    size_t i = 0U;
    if (graph == NULL || out_order == NULL || out_count == NULL) {
        return -1;
    }
    for (i = 0U; i < graph->node_count; ++i) {
        indegree[i] = 0U;
    }
    for (i = 0U; i < graph->edge_count; ++i) {
        int to_index = network_graph_find_node_index(graph, graph->edges[i].to_id);
        if (to_index < 0) {
            return -1;
        }
        indegree[(size_t)to_index] += 1U;
    }
    for (i = 0U; i < graph->node_count; ++i) {
        if (indegree[i] == 0U) {
            queue[q_tail++] = i;
        }
    }
    while (q_head < q_tail) {
        size_t current = queue[q_head++];
        out_order[order_count++] = current;
        for (i = 0U; i < graph->edge_count; ++i) {
            int from_index = network_graph_find_node_index(graph, graph->edges[i].from_id);
            int to_index = network_graph_find_node_index(graph, graph->edges[i].to_id);
            if (from_index < 0 || to_index < 0) {
                return -1;
            }
            if ((size_t)from_index == current) {
                size_t to = (size_t)to_index;
                if (indegree[to] == 0U) {
                    return -1;
                }
                indegree[to] -= 1U;
                if (indegree[to] == 0U) {
                    queue[q_tail++] = to;
                }
            }
        }
    }
    *out_count = order_count;
    return (order_count == graph->node_count) ? 0 : -1;
}

static int network_spec_append_layer(NetworkSpec* spec, NetworkLayerKind kind) {
    if (spec == NULL || spec->layer_count >= NETWORK_SPEC_MAX_LAYERS) {
        return -1;
    }
    spec->layers[spec->layer_count].kind = kind;
    spec->layer_count += 1U;
    return 0;
}

static int network_node_type_is_compute(int type) {
    if (type == NETWORK_NODE_LINEAR ||
        type == NETWORK_NODE_TRANSFORMER_BLOCK ||
        type == NETWORK_NODE_ATTENTION_HEAD ||
        type == NETWORK_NODE_CNN ||
        type == NETWORK_NODE_RNN ||
        type == NETWORK_NODE_KNN) {
        return 1;
    }
    return 0;
}

int network_graph_build(NetworkGraph* graph,
                        const NetworkGraphNode* nodes,
                        size_t node_count,
                        const NetworkGraphEdge* edges,
                        size_t edge_count,
                        int input_node_id,
                        int output_node_id) {
    size_t i = 0U;
    if (graph == NULL || nodes == NULL || edges == NULL || node_count == 0U || edge_count == 0U ||
        node_count > NETWORK_GRAPH_MAX_NODES || edge_count > NETWORK_GRAPH_MAX_EDGES) {
        return -1;
    }
    graph->node_count = node_count;
    graph->edge_count = edge_count;
    graph->input_node_id = input_node_id;
    graph->output_node_id = output_node_id;
    for (i = 0U; i < node_count; ++i) {
        graph->nodes[i] = nodes[i];
    }
    for (i = 0U; i < edge_count; ++i) {
        graph->edges[i] = edges[i];
    }
    return network_graph_validate(graph);
}

int network_graph_validate(const NetworkGraph* graph) {
    size_t i = 0U;
    size_t j = 0U;
    size_t order[NETWORK_GRAPH_MAX_NODES];
    size_t order_count = 0U;
    if (graph == NULL || graph->node_count == 0U || graph->edge_count == 0U ||
        graph->node_count > NETWORK_GRAPH_MAX_NODES || graph->edge_count > NETWORK_GRAPH_MAX_EDGES) {
        return -1;
    }
    if (network_graph_find_node_index(graph, graph->input_node_id) < 0 ||
        network_graph_find_node_index(graph, graph->output_node_id) < 0) {
        return -1;
    }
    for (i = 0U; i < graph->node_count; ++i) {
        for (j = i + 1U; j < graph->node_count; ++j) {
            if (graph->nodes[i].id == graph->nodes[j].id) {
                return -1;
            }
        }
        if (graph->nodes[i].type == NETWORK_NODE_SELECT && graph->nodes[i].selector_size <= 0) {
            return -1;
        }
    }
    for (i = 0U; i < graph->edge_count; ++i) {
        if (network_graph_find_node_index(graph, graph->edges[i].from_id) < 0 ||
            network_graph_find_node_index(graph, graph->edges[i].to_id) < 0) {
            return -1;
        }
    }
    if (network_graph_toposort(graph, order, &order_count) != 0) {
        return -1;
    }
    return 0;
}

int network_spec_build_from_graph(NetworkSpec* spec, const NetworkGraph* graph) {
    size_t i = 0U;
    size_t order[NETWORK_GRAPH_MAX_NODES];
    size_t order_count = 0U;
    if (spec == NULL || graph == NULL) {
        return -1;
    }
    if (network_graph_validate(graph) != 0) {
        return -1;
    }
    if (network_graph_toposort(graph, order, &order_count) != 0) {
        return -1;
    }
    spec->layer_count = 0U;
    for (i = 0U; i < order_count; ++i) {
        const NetworkGraphNode* node = &graph->nodes[order[i]];
        if (!network_node_type_is_compute(node->type)) {
            continue;
        }
        if (node->type == NETWORK_NODE_LINEAR) {
            if (network_spec_append_layer(spec, NETWORK_LAYER_TOKEN_STATE_LINEAR) != 0) {
                return -1;
            }
            continue;
        }
        if (node->type == NETWORK_NODE_TRANSFORMER_BLOCK) {
            if (network_spec_append_layer(spec, NETWORK_LAYER_TRANSFORMER_BLOCK) != 0) {
                return -1;
            }
            continue;
        }
        if (node->type == NETWORK_NODE_ATTENTION_HEAD) {
            if (network_spec_append_layer(spec, NETWORK_LAYER_ATTENTION_HEAD) != 0) {
                return -1;
            }
            continue;
        }
        if (node->type == NETWORK_NODE_CNN) {
            if (network_spec_append_layer(spec, NETWORK_LAYER_CNN) != 0) {
                return -1;
            }
            continue;
        }
        if (node->type == NETWORK_NODE_RNN) {
            if (network_spec_append_layer(spec, NETWORK_LAYER_RNN) != 0) {
                return -1;
            }
            continue;
        }
        if (node->type == NETWORK_NODE_KNN) {
            if (network_spec_append_layer(spec, NETWORK_LAYER_KNN) != 0) {
                return -1;
            }
            continue;
        }
    }
    return network_spec_validate(spec);
}

int network_spec_validate(const NetworkSpec* spec) {
    size_t i = 0U;
    size_t linear_count = 0U;
    if (spec == NULL || spec->layer_count == 0U || spec->layer_count > NETWORK_SPEC_MAX_LAYERS) {
        return -1;
    }
    if (spec->layers[0].kind != NETWORK_LAYER_TOKEN_STATE_LINEAR) {
        return -1;
    }
    for (i = 0U; i < spec->layer_count; ++i) {
        if (spec->layers[i].kind == NETWORK_LAYER_TOKEN_STATE_LINEAR) {
            linear_count += 1U;
            continue;
        }
        if (spec->layers[i].kind == NETWORK_LAYER_TRANSFORMER_BLOCK ||
            spec->layers[i].kind == NETWORK_LAYER_ATTENTION_HEAD ||
            spec->layers[i].kind == NETWORK_LAYER_CNN ||
            spec->layers[i].kind == NETWORK_LAYER_RNN ||
            spec->layers[i].kind == NETWORK_LAYER_KNN) {
            continue;
        }
        return -1;
    }
    return (linear_count == 1U) ? 0 : -1;
}

size_t network_spec_weight_count(const NetworkSpec* spec) {
    if (network_spec_validate(spec) != 0) {
        return 0U;
    }
    return (size_t)TOKEN_WEIGHT_COUNT + (size_t)STATE_WEIGHT_COUNT + (size_t)BIAS_COUNT;
}
