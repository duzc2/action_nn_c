/**
 * @file generate_main.c
 * @brief Nested navigation demo code generation entry
 */

#include "profiler.h"
#include "network_def.h"
#include "types/mlp/mlp_config.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>

#define NESTED_NAV_INPUT_SIZE 10U
#define TARGET_ENCODER_OUTPUT_SIZE 20U
#define OBSTACLE_ENCODER_OUTPUT_SIZE 24U
#define FUSION_INPUT_SIZE (TARGET_ENCODER_OUTPUT_SIZE + OBSTACLE_ENCODER_OUTPUT_SIZE)

static void fill_mlp_infer_config(
    MlpConfig* config,
    size_t input_size,
    size_t output_size,
    size_t hidden_layer_count,
    const size_t* hidden_sizes
) {
    size_t hidden_index;

    memset(config, 0, sizeof(*config));
    config->input_size = input_size;
    config->hidden_layer_count = hidden_layer_count;
    for (hidden_index = 0U; hidden_index < 4U; ++hidden_index) {
        config->hidden_sizes[hidden_index] =
            (hidden_sizes != NULL && hidden_index < hidden_layer_count) ? hidden_sizes[hidden_index] : 0U;
    }
    config->output_size = output_size;
    config->hidden_activation = MLP_ACT_TANH;
    config->output_activation = MLP_ACT_NONE;
}

static void fill_mlp_train_config(
    MlpTrainConfig* config,
    float learning_rate,
    uint32_t seed
) {
    memset(config, 0, sizeof(*config));
    config->learning_rate = learning_rate;
    config->momentum = 0.9f;
    config->weight_decay = 0.0001f;
    config->optimizer = MLP_OPT_ADAM;
    config->loss_func = MLP_LOSS_MSE;
    config->batch_size = 1U;
    config->seed = seed;
}

static NNSubnetDef* create_leaf_subnet(
    const char* subnet_id,
    size_t input_size,
    size_t output_size,
    size_t hidden_layer_count,
    const size_t* hidden_sizes,
    float learning_rate,
    uint32_t seed
) {
    NNSubnetDef* subnet;
    MlpConfig infer_config;
    MlpTrainConfig train_config;

    subnet = nn_subnet_def_create(subnet_id, "mlp", input_size, output_size);
    if (subnet == NULL) {
        return NULL;
    }

    nn_subnet_def_set_hidden_layers(subnet, hidden_layer_count, hidden_sizes);
    fill_mlp_infer_config(&infer_config, input_size, output_size, hidden_layer_count, hidden_sizes);
    fill_mlp_train_config(&train_config, learning_rate, seed);

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

static int add_full_connection_block(
    NN_NetworkDef* network,
    const char* source_subnet_id,
    size_t source_count,
    const char* target_subnet_id,
    size_t target_offset
) {
    size_t node_index;

    for (node_index = 0U; node_index < source_count; ++node_index) {
        NNConnectionDef* connection = nn_connection_def_create(
            source_subnet_id,
            "out",
            node_index,
            target_subnet_id,
            "in",
            target_offset + node_index
        );
        if (connection == NULL) {
            return -1;
        }
        nn_network_def_add_connection(network, connection);
    }

    return 0;
}

static NN_NetworkDef* create_nested_nav_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* agent;
    NNSubnetDef* perception;
    NNSubnetDef* planner;
    NNSubnetDef* target_encoder;
    NNSubnetDef* obstacle_encoder;
    NNSubnetDef* fusion_head;
    size_t target_hidden_sizes[1] = {16U};
    size_t obstacle_hidden_sizes[1] = {20U};
    size_t fusion_hidden_sizes[2] = {32U, 24U};

    network = nn_network_def_create("nested_nav");
    if (network == NULL) {
        return NULL;
    }

    agent = nn_subnet_def_create("agent", NULL, 0U, 0U);
    perception = nn_subnet_def_create("perception", NULL, 0U, 0U);
    planner = nn_subnet_def_create("planner", NULL, 0U, 0U);
    target_encoder = create_leaf_subnet(
        "target_encoder",
        NESTED_NAV_INPUT_SIZE,
        TARGET_ENCODER_OUTPUT_SIZE,
        1U,
        target_hidden_sizes,
        0.0030f,
        11U
    );
    obstacle_encoder = create_leaf_subnet(
        "obstacle_encoder",
        NESTED_NAV_INPUT_SIZE,
        OBSTACLE_ENCODER_OUTPUT_SIZE,
        1U,
        obstacle_hidden_sizes,
        0.0035f,
        29U
    );
    fusion_head = create_leaf_subnet(
        "fusion_head",
        FUSION_INPUT_SIZE,
        2U,
        2U,
        fusion_hidden_sizes,
        0.004f,
        97U
    );

    if (agent == NULL || perception == NULL || planner == NULL ||
        target_encoder == NULL || obstacle_encoder == NULL || fusion_head == NULL) {
        nn_subnet_def_free(agent);
        nn_subnet_def_free(perception);
        nn_subnet_def_free(planner);
        nn_subnet_def_free(target_encoder);
        nn_subnet_def_free(obstacle_encoder);
        nn_subnet_def_free(fusion_head);
        nn_network_def_free(network);
        return NULL;
    }

    nn_subnet_def_add_subnet(perception, target_encoder);
    nn_subnet_def_add_subnet(perception, obstacle_encoder);
    nn_subnet_def_add_subnet(planner, fusion_head);
    nn_subnet_def_add_subnet(agent, perception);
    nn_subnet_def_add_subnet(agent, planner);
    nn_network_def_add_subnet(network, agent);

    if (add_full_connection_block(
            network,
            "target_encoder",
            TARGET_ENCODER_OUTPUT_SIZE,
            "fusion_head",
            0U) != 0 ||
        add_full_connection_block(
            network,
            "obstacle_encoder",
            OBSTACLE_ENCODER_OUTPUT_SIZE,
            "fusion_head",
            TARGET_ENCODER_OUTPUT_SIZE) != 0) {
        nn_network_def_free(network);
        return NULL;
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

    network = create_nested_nav_network();
    if (network == NULL) {
        fprintf(stderr, "failed to create nested navigation network definition\n");
        return 1;
    }

    memset(&req, 0, sizeof(req));
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

    printf("nested_nav code generated\n");
    printf("network hash: 0x%016llx\n", (unsigned long long)result.network_hash);
    printf("metadata: %s\n", result.metadata_written_path);

    nn_network_def_free(network);
    return 0;
}
