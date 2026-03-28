/**
 * @file generate_main.c
 * @brief road_graph_nav demo code generation entry.
 */

#include "profiler.h"
#include "network_def.h"
#include "types/gnn/gnn_config.h"
#include "types/mlp/mlp_config.h"
#include "../demo_runtime_paths.h"
#include "road_graph_nav_scene.h"

#include <stdio.h>
#include <string.h>

#define ROAD_GRAPH_NAV_INPUT_SIZE (ROAD_GRAPH_NAV_NODE_COUNT * ROAD_GRAPH_NAV_NODE_FEATURE_SIZE)

/**
 * @brief Fill the fixed topology table shared by the graph-encoder leaf.
 */
static void road_graph_nav_fill_gnn_neighbors(GnnConfig* config) {
    road_graph_nav_fill_neighbors(config->neighbor_index);
}

/**
 * @brief Build the graph encoder leaf that uses the generic GNN backend.
 */
static NNSubnetDef* create_graph_encoder_leaf(void) {
    NNSubnetDef* subnet;
    GnnConfig infer_config;
    GnnTrainConfig train_config;
    size_t hidden_sizes[1] = {12U};

    subnet = nn_subnet_def_create(
        "graph_encoder",
        "gnn",
        ROAD_GRAPH_NAV_INPUT_SIZE,
        ROAD_GRAPH_NAV_GNN_OUTPUT_SIZE
    );
    if (subnet == NULL) {
        return NULL;
    }

    if (nn_subnet_def_set_hidden_layers(subnet, 1U, hidden_sizes) != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    (void)memset(&infer_config, 0, sizeof(infer_config));
    infer_config.node_count = ROAD_GRAPH_NAV_NODE_COUNT;
    infer_config.node_feature_size = ROAD_GRAPH_NAV_NODE_FEATURE_SIZE;
    infer_config.hidden_size = 12U;
    infer_config.output_size = ROAD_GRAPH_NAV_GNN_OUTPUT_SIZE;
    infer_config.message_passes = 2U;
    infer_config.slot_count = ROAD_GRAPH_NAV_SLOT_COUNT;
    infer_config.node_mask_feature_index = ROAD_GRAPH_NAV_FEATURE_OPEN;
    infer_config.primary_anchor_feature_index = ROAD_GRAPH_NAV_FEATURE_CURRENT;
    infer_config.secondary_anchor_feature_index = ROAD_GRAPH_NAV_FEATURE_TARGET;
    infer_config.aggregator_type = GNN_AGG_MEAN;
    infer_config.readout_type = GNN_READOUT_ANCHOR_SLOTS;
    infer_config.hidden_activation = GNN_ACT_TANH;
    infer_config.output_activation = GNN_ACT_NONE;
    infer_config.seed = 23U;
    road_graph_nav_fill_gnn_neighbors(&infer_config);

    (void)memset(&train_config, 0, sizeof(train_config));
    train_config.learning_rate = 0.0035f;
    train_config.weight_decay = 0.0001f;
    train_config.batch_size = 1U;
    train_config.seed = 41U;

    if (nn_subnet_def_set_infer_type_config(
            subnet,
            &infer_config,
            sizeof(infer_config),
            "types/gnn/gnn_config.h",
            "GnnConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    if (nn_subnet_def_set_train_type_config(
            subnet,
            &train_config,
            sizeof(train_config),
            "types/gnn/gnn_config.h",
            "GnnTrainConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    return subnet;
}

/**
 * @brief Build the policy head that maps graph embeddings to action scores.
 */
static NNSubnetDef* create_decision_head_leaf(void) {
    NNSubnetDef* subnet;
    MlpConfig infer_config;
    MlpTrainConfig train_config;
    size_t hidden_sizes[1] = {16U};

    subnet = nn_subnet_def_create(
        "decision_head",
        "mlp",
        ROAD_GRAPH_NAV_GNN_OUTPUT_SIZE,
        ROAD_GRAPH_NAV_ACTION_COUNT
    );
    if (subnet == NULL) {
        return NULL;
    }

    if (nn_subnet_def_set_hidden_layers(subnet, 1U, hidden_sizes) != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    (void)memset(&infer_config, 0, sizeof(infer_config));
    infer_config.input_size = ROAD_GRAPH_NAV_GNN_OUTPUT_SIZE;
    infer_config.hidden_layer_count = 1U;
    infer_config.hidden_sizes[0] = hidden_sizes[0];
    infer_config.output_size = ROAD_GRAPH_NAV_ACTION_COUNT;
    infer_config.hidden_activation = MLP_ACT_TANH;
    infer_config.output_activation = MLP_ACT_NONE;

    (void)memset(&train_config, 0, sizeof(train_config));
    train_config.learning_rate = 0.0050f;
    train_config.momentum = 0.9f;
    train_config.weight_decay = 0.0001f;
    train_config.optimizer = MLP_OPT_ADAM;
    train_config.loss_func = MLP_LOSS_MSE;
    train_config.batch_size = 1U;
    train_config.seed = 79U;

    if (nn_subnet_def_set_infer_type_config(
            subnet,
            &infer_config,
            sizeof(infer_config),
            "types/mlp/mlp_config.h",
            "MlpConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    if (nn_subnet_def_set_train_type_config(
            subnet,
            &train_config,
            sizeof(train_config),
            "types/mlp/mlp_config.h",
            "MlpTrainConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    return subnet;
}

/**
 * @brief Create the composed graph_encoder(gnn) -> decision_head(mlp) network.
 */
static NN_NetworkDef* create_road_graph_nav_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* root;
    NNSubnetDef* graph_group;
    NNSubnetDef* policy_group;
    NNSubnetDef* graph_encoder;
    NNSubnetDef* decision_head;
    size_t node_index;

    network = nn_network_def_create("road_graph_nav");
    if (network == NULL) {
        return NULL;
    }

    root = nn_subnet_def_create("agent", NULL, 0U, 0U);
    graph_group = nn_subnet_def_create("graph_group", NULL, 0U, 0U);
    policy_group = nn_subnet_def_create("policy_group", NULL, 0U, 0U);
    graph_encoder = create_graph_encoder_leaf();
    decision_head = create_decision_head_leaf();

    if (root == NULL || graph_group == NULL || policy_group == NULL ||
        graph_encoder == NULL || decision_head == NULL) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(graph_group);
        nn_subnet_def_free(policy_group);
        nn_subnet_def_free(graph_encoder);
        nn_subnet_def_free(decision_head);
        nn_network_def_free(network);
        return NULL;
    }

    if (nn_subnet_def_add_subnet(graph_group, graph_encoder) != 0) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(graph_group);
        nn_subnet_def_free(policy_group);
        nn_subnet_def_free(graph_encoder);
        nn_subnet_def_free(decision_head);
        nn_network_def_free(network);
        return NULL;
    }
    graph_encoder = NULL;

    if (nn_subnet_def_add_subnet(policy_group, decision_head) != 0) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(graph_group);
        nn_subnet_def_free(policy_group);
        nn_subnet_def_free(graph_encoder);
        nn_subnet_def_free(decision_head);
        nn_network_def_free(network);
        return NULL;
    }
    decision_head = NULL;

    if (nn_subnet_def_add_subnet(root, graph_group) != 0) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(graph_group);
        nn_subnet_def_free(policy_group);
        nn_subnet_def_free(graph_encoder);
        nn_subnet_def_free(decision_head);
        nn_network_def_free(network);
        return NULL;
    }
    graph_group = NULL;

    if (nn_subnet_def_add_subnet(root, policy_group) != 0) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(graph_group);
        nn_subnet_def_free(policy_group);
        nn_subnet_def_free(graph_encoder);
        nn_subnet_def_free(decision_head);
        nn_network_def_free(network);
        return NULL;
    }
    policy_group = NULL;

    if (nn_network_def_add_subnet(network, root) != 0) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(graph_group);
        nn_subnet_def_free(policy_group);
        nn_subnet_def_free(graph_encoder);
        nn_subnet_def_free(decision_head);
        nn_network_def_free(network);
        return NULL;
    }
    root = NULL;

    for (node_index = 0U; node_index < ROAD_GRAPH_NAV_GNN_OUTPUT_SIZE; ++node_index) {
        NNConnectionDef* connection = nn_connection_def_create(
            "graph_encoder",
            "out",
            node_index,
            "decision_head",
            "in",
            node_index
        );

        if (connection == NULL) {
            nn_network_def_free(network);
            return NULL;
        }
        if (nn_network_def_add_connection(network, connection) != 0) {
            nn_connection_def_free(connection);
            nn_network_def_free(network);
            return NULL;
        }
    }

    return network;
}

int main(void) {
    NN_NetworkDef* network;
    ProfGenerateRequest req;
    ProfGenerateResult result;
    ProfStatus st;
    char error_buffer[512];
    ProfOutputLayout layout = {
        .tokenizer = { .c_path = "../data/tokenizer.c", .h_path = "../data/tokenizer.h" },
        .network_init = { .c_path = "../data/network_init.c", .h_path = "../data/network_init.h" },
        .weights_load = { .c_path = "../data/weights_load.c", .h_path = "../data/weights_load.h" },
        .train = { .c_path = "../data/train.c", .h_path = "../data/train.h" },
        .weights_save = { .c_path = "../data/weights_save.c", .h_path = "../data/weights_save.h" },
        .infer = { .c_path = "../data/infer.c", .h_path = "../data/infer.h" },
        .metadata_path = "../data/network_metadata.h"
    };

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "failed to switch working directory to executable directory\n");
        return 1;
    }

    network = create_road_graph_nav_network();
    if (network == NULL) {
        fprintf(stderr, "failed to create road graph navigation network definition\n");
        return 1;
    }

    (void)memset(&req, 0, sizeof(req));
    req.network_def = (const void*)network;
    req.output_layout = layout;
    req.error.buffer = error_buffer;
    req.error.capacity = sizeof(error_buffer);

    st = profiler_generate_v2(&req, &result);
    if (st != PROF_STATUS_OK) {
        fprintf(stderr, "profiler_generate_v2 failed: %s\n", error_buffer);
        nn_network_def_free(network);
        return 1;
    }

    printf("road_graph_nav code generated\n");
    printf("network hash: 0x%016llx\n", (unsigned long long)result.network_hash);
    printf("metadata: %s\n", result.metadata_written_path);

    nn_network_def_free(network);
    return 0;
}

