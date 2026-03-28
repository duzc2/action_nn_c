/**
 * @file generate_main.c
 * @brief Hybrid transformer+MLP route demo code generation entry
 */

#include "profiler.h"
#include "network_def.h"
#include "types/mlp/mlp_config.h"
#include "types/transformer/transformer_config.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>

#define HYBRID_ROUTE_INPUT_SIZE 8U
#define HYBRID_ROUTE_EMBED_SIZE 8U

static NNSubnetDef* create_transformer_leaf(void) {
    NNSubnetDef* subnet;
    TransformerModelConfig infer_config;
    TransformerTrainConfig train_config;

    subnet = nn_subnet_def_create(
        "sequence_encoder",
        "transformer",
        HYBRID_ROUTE_INPUT_SIZE,
        HYBRID_ROUTE_EMBED_SIZE
    );
    if (subnet == NULL) {
        return NULL;
    }

    (void)memset(&infer_config, 0, sizeof(infer_config));
    infer_config.model_dim = HYBRID_ROUTE_EMBED_SIZE;
    infer_config.seed = 7U;

    (void)memset(&train_config, 0, sizeof(train_config));
    train_config.learning_rate = 0.003f;

    if (nn_subnet_def_set_infer_type_config(
            subnet,
            &infer_config,
            sizeof(infer_config),
            "types/transformer/transformer_config.h",
            "TransformerModelConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    if (nn_subnet_def_set_train_type_config(
            subnet,
            &train_config,
            sizeof(train_config),
            "types/transformer/transformer_config.h",
            "TransformerTrainConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    return subnet;
}

static NNSubnetDef* create_fusion_leaf(void) {
    NNSubnetDef* subnet;
    size_t hidden_sizes[1] = {12U};
    MlpConfig infer_config;
    MlpTrainConfig train_config;

    subnet = nn_subnet_def_create("decision_head", "mlp", HYBRID_ROUTE_EMBED_SIZE, 2U);
    if (subnet == NULL) {
        return NULL;
    }

    if (nn_subnet_def_set_hidden_layers(subnet, 1U, hidden_sizes) != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    (void)memset(&infer_config, 0, sizeof(infer_config));
    infer_config.input_size = HYBRID_ROUTE_EMBED_SIZE;
    infer_config.hidden_layer_count = 1U;
    infer_config.hidden_sizes[0] = hidden_sizes[0];
    infer_config.output_size = 2U;
    infer_config.hidden_activation = MLP_ACT_TANH;
    infer_config.output_activation = MLP_ACT_NONE;

    (void)memset(&train_config, 0, sizeof(train_config));
    train_config.learning_rate = 0.004f;
    train_config.momentum = 0.9f;
    train_config.weight_decay = 0.0001f;
    train_config.optimizer = MLP_OPT_ADAM;
    train_config.loss_func = MLP_LOSS_MSE;
    train_config.batch_size = 1U;
    train_config.seed = 19U;

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

static NN_NetworkDef* create_hybrid_route_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* root;
    NNSubnetDef* semantic_group;
    NNSubnetDef* planner_group;
    NNSubnetDef* transformer_leaf;
    NNSubnetDef* fusion_leaf;
    size_t node_index;

    network = nn_network_def_create("hybrid_route");
    if (network == NULL) {
        return NULL;
    }

    root = nn_subnet_def_create("agent", NULL, 0U, 0U);
    semantic_group = nn_subnet_def_create("semantic_group", NULL, 0U, 0U);
    planner_group = nn_subnet_def_create("planner_group", NULL, 0U, 0U);
    transformer_leaf = create_transformer_leaf();
    fusion_leaf = create_fusion_leaf();

    if (root == NULL || semantic_group == NULL || planner_group == NULL ||
        transformer_leaf == NULL || fusion_leaf == NULL) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(semantic_group);
        nn_subnet_def_free(planner_group);
        nn_subnet_def_free(transformer_leaf);
        nn_subnet_def_free(fusion_leaf);
        nn_network_def_free(network);
        return NULL;
    }

    if (nn_subnet_def_add_subnet(semantic_group, transformer_leaf) != 0) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(semantic_group);
        nn_subnet_def_free(planner_group);
        nn_subnet_def_free(transformer_leaf);
        nn_subnet_def_free(fusion_leaf);
        nn_network_def_free(network);
        return NULL;
    }
    transformer_leaf = NULL;

    if (nn_subnet_def_add_subnet(planner_group, fusion_leaf) != 0) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(semantic_group);
        nn_subnet_def_free(planner_group);
        nn_subnet_def_free(transformer_leaf);
        nn_subnet_def_free(fusion_leaf);
        nn_network_def_free(network);
        return NULL;
    }
    fusion_leaf = NULL;

    if (nn_subnet_def_add_subnet(root, semantic_group) != 0) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(semantic_group);
        nn_subnet_def_free(planner_group);
        nn_subnet_def_free(transformer_leaf);
        nn_subnet_def_free(fusion_leaf);
        nn_network_def_free(network);
        return NULL;
    }
    semantic_group = NULL;

    if (nn_subnet_def_add_subnet(root, planner_group) != 0) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(semantic_group);
        nn_subnet_def_free(planner_group);
        nn_subnet_def_free(transformer_leaf);
        nn_subnet_def_free(fusion_leaf);
        nn_network_def_free(network);
        return NULL;
    }
    planner_group = NULL;

    if (nn_network_def_add_subnet(network, root) != 0) {
        nn_subnet_def_free(root);
        nn_subnet_def_free(semantic_group);
        nn_subnet_def_free(planner_group);
        nn_subnet_def_free(transformer_leaf);
        nn_subnet_def_free(fusion_leaf);
        nn_network_def_free(network);
        return NULL;
    }
    root = NULL;

    for (node_index = 0U; node_index < HYBRID_ROUTE_EMBED_SIZE; ++node_index) {
        NNConnectionDef* connection = nn_connection_def_create(
            "sequence_encoder",
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
        .tokenizer = { .c_path = "../../data/tokenizer.c", .h_path = "../../data/tokenizer.h" },
        .network_init = { .c_path = "../../data/network_init.c", .h_path = "../../data/network_init.h" },
        .weights_load = { .c_path = "../../data/weights_load.c", .h_path = "../../data/weights_load.h" },
        .train = { .c_path = "../../data/train.c", .h_path = "../../data/train.h" },
        .weights_save = { .c_path = "../../data/weights_save.c", .h_path = "../../data/weights_save.h" },
        .infer = { .c_path = "../../data/infer.c", .h_path = "../../data/infer.h" },
        .metadata_path = "../../data/network_metadata.h"
    };

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "failed to switch working directory to executable directory\n");
        return 1;
    }

    network = create_hybrid_route_network();
    if (network == NULL) {
        fprintf(stderr, "failed to create hybrid route network definition\n");
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

    printf("hybrid_route code generated\n");
    printf("network hash: 0x%016llx\n", (unsigned long long)result.network_hash);
    printf("metadata: %s\n", result.metadata_written_path);

    nn_network_def_free(network);
    return 0;
}
