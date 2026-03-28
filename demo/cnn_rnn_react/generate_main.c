/**
 * @file generate_main.c
 * @brief Code-generation entry for the "cross the road while avoiding cars" demo.
 *
 * The generated network is not a classifier. It is a nested controller:
 * - CNN reads each full 30x20 road frame and extracts where the goal/cars are;
 * - RNN fuses four frames and emits the immediate crossing response.
 */

#include "profiler.h"
#include "network_def.h"
#include "types/cnn/cnn_config.h"
#include "types/rnn/rnn_config.h"
#include "../demo_runtime_paths.h"
#include "cnn_rnn_react_scene.h"

#include <stdio.h>
#include <string.h>

/**
 * @brief Fill the typed CNN infer config used by the profiler.
 */
static void fill_cnn_infer_config(CnnConfig* config) {
    (void)memset(config, 0, sizeof(*config));
    config->total_input_size = CNN_RNN_REACT_INPUT_SIZE;
    config->sequence_length = CNN_RNN_REACT_SEQUENCE_LENGTH;
    config->frame_width = CNN_RNN_REACT_FRAME_WIDTH;
    config->frame_height = CNN_RNN_REACT_FRAME_HEIGHT;
    config->channel_count = CNN_RNN_REACT_CHANNEL_COUNT;
    config->kernel_size = 3U;
    config->filter_count = 8U;
    config->feature_size = CNN_RNN_REACT_CNN_FEATURE_SIZE;
    config->pooling_activation = CNN_ACT_TANH;
    config->output_activation = CNN_ACT_TANH;
    config->seed = 17U;
}

/**
 * @brief Fill the typed CNN train config used by the profiler.
 */
static void fill_cnn_train_config(CnnTrainConfig* config) {
    (void)memset(config, 0, sizeof(*config));
    config->learning_rate = 0.006f;
    config->momentum = 0.0f;
    config->weight_decay = 0.0002f;
    config->batch_size = 1U;
    config->seed = 17U;
}

/**
 * @brief Fill the typed RNN infer config used by the profiler.
 */
static void fill_rnn_infer_config(RnnConfig* config) {
    (void)memset(config, 0, sizeof(*config));
    config->sequence_length = CNN_RNN_REACT_SEQUENCE_LENGTH;
    config->input_feature_size = CNN_RNN_REACT_CNN_FEATURE_SIZE;
    config->hidden_size = CNN_RNN_REACT_RNN_HIDDEN_SIZE;
    config->output_size = CNN_RNN_REACT_OUTPUT_SIZE;
    config->hidden_activation = RNN_ACT_TANH;
    config->output_activation = RNN_ACT_TANH;
    config->seed = 29U;
}

/**
 * @brief Fill the typed RNN train config used by the profiler.
 */
static void fill_rnn_train_config(RnnTrainConfig* config) {
    (void)memset(config, 0, sizeof(*config));
    config->learning_rate = 0.008f;
    config->momentum = 0.0f;
    config->weight_decay = 0.0001f;
    config->batch_size = 1U;
    config->seed = 29U;
}

/**
 * @brief Create the CNN leaf that encodes each full-map road frame into compact cues.
 */
static NNSubnetDef* create_cnn_leaf(void) {
    NNSubnetDef* subnet;
    CnnConfig infer_config;
    CnnTrainConfig train_config;
    size_t hidden_sizes[1] = {8U};

    subnet = nn_subnet_def_create(
        "cnn_encoder",
        "cnn",
        CNN_RNN_REACT_INPUT_SIZE,
        CNN_RNN_REACT_CNN_OUTPUT_SIZE
    );
    if (subnet == NULL) {
        return NULL;
    }

    if (nn_subnet_def_set_hidden_layers(subnet, 1U, hidden_sizes) != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }
    fill_cnn_infer_config(&infer_config);
    fill_cnn_train_config(&train_config);

    if (nn_subnet_def_set_infer_type_config(
            subnet,
            &infer_config,
            sizeof(infer_config),
            "types/cnn/cnn_config.h",
            "CnnConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    if (nn_subnet_def_set_train_type_config(
            subnet,
            &train_config,
            sizeof(train_config),
            "types/cnn/cnn_config.h",
            "CnnTrainConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    return subnet;
}

/**
 * @brief Create the RNN leaf that turns four snapshot features into crossing actions.
 */
static NNSubnetDef* create_rnn_leaf(void) {
    NNSubnetDef* subnet;
    RnnConfig infer_config;
    RnnTrainConfig train_config;
    size_t hidden_sizes[1] = {CNN_RNN_REACT_RNN_HIDDEN_SIZE};

    subnet = nn_subnet_def_create(
        "rnn_reactor",
        "rnn",
        CNN_RNN_REACT_CNN_OUTPUT_SIZE,
        CNN_RNN_REACT_OUTPUT_SIZE
    );
    if (subnet == NULL) {
        return NULL;
    }

    if (nn_subnet_def_set_hidden_layers(subnet, 1U, hidden_sizes) != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }
    fill_rnn_infer_config(&infer_config);
    fill_rnn_train_config(&train_config);

    if (nn_subnet_def_set_infer_type_config(
            subnet,
            &infer_config,
            sizeof(infer_config),
            "types/rnn/rnn_config.h",
            "RnnConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    if (nn_subnet_def_set_train_type_config(
            subnet,
            &train_config,
            sizeof(train_config),
            "types/rnn/rnn_config.h",
            "RnnTrainConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    return subnet;
}

/**
 * @brief Create the nested CNN->RNN road-crossing controller definition.
 */
static NN_NetworkDef* create_cnn_rnn_react_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* agent;
    NNSubnetDef* perception;
    NNSubnetDef* controller;
    NNSubnetDef* cnn_leaf;
    NNSubnetDef* rnn_leaf;
    size_t feature_index;

    network = nn_network_def_create("cnn_rnn_react");
    if (network == NULL) {
        return NULL;
    }

    agent = nn_subnet_def_create("agent", NULL, 0U, 0U);
    perception = nn_subnet_def_create("perception", NULL, 0U, 0U);
    controller = nn_subnet_def_create("controller", NULL, 0U, 0U);
    cnn_leaf = create_cnn_leaf();
    rnn_leaf = create_rnn_leaf();

    if (agent == NULL || perception == NULL || controller == NULL ||
        cnn_leaf == NULL || rnn_leaf == NULL) {
        nn_subnet_def_free(agent);
        nn_subnet_def_free(perception);
        nn_subnet_def_free(controller);
        nn_subnet_def_free(cnn_leaf);
        nn_subnet_def_free(rnn_leaf);
        nn_network_def_free(network);
        return NULL;
    }

    if (nn_subnet_def_add_subnet(perception, cnn_leaf) != 0) {
        nn_subnet_def_free(agent);
        nn_subnet_def_free(perception);
        nn_subnet_def_free(controller);
        nn_subnet_def_free(cnn_leaf);
        nn_subnet_def_free(rnn_leaf);
        nn_network_def_free(network);
        return NULL;
    }
    cnn_leaf = NULL;

    if (nn_subnet_def_add_subnet(controller, rnn_leaf) != 0) {
        nn_subnet_def_free(agent);
        nn_subnet_def_free(perception);
        nn_subnet_def_free(controller);
        nn_subnet_def_free(cnn_leaf);
        nn_subnet_def_free(rnn_leaf);
        nn_network_def_free(network);
        return NULL;
    }
    rnn_leaf = NULL;

    if (nn_subnet_def_add_subnet(agent, perception) != 0) {
        nn_subnet_def_free(agent);
        nn_subnet_def_free(perception);
        nn_subnet_def_free(controller);
        nn_subnet_def_free(cnn_leaf);
        nn_subnet_def_free(rnn_leaf);
        nn_network_def_free(network);
        return NULL;
    }
    perception = NULL;

    if (nn_subnet_def_add_subnet(agent, controller) != 0) {
        nn_subnet_def_free(agent);
        nn_subnet_def_free(perception);
        nn_subnet_def_free(controller);
        nn_subnet_def_free(cnn_leaf);
        nn_subnet_def_free(rnn_leaf);
        nn_network_def_free(network);
        return NULL;
    }
    controller = NULL;

    if (nn_network_def_add_subnet(network, agent) != 0) {
        nn_subnet_def_free(agent);
        nn_subnet_def_free(perception);
        nn_subnet_def_free(controller);
        nn_subnet_def_free(cnn_leaf);
        nn_subnet_def_free(rnn_leaf);
        nn_network_def_free(network);
        return NULL;
    }
    agent = NULL;

    /* Feed every per-frame CNN feature into the matching RNN time-step slot. */
    for (feature_index = 0U; feature_index < CNN_RNN_REACT_CNN_OUTPUT_SIZE; ++feature_index) {
        NNConnectionDef* connection = nn_connection_def_create(
            "cnn_encoder",
            "out",
            feature_index,
            "rnn_reactor",
            "in",
            feature_index
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

    network = create_cnn_rnn_react_network();
    if (network == NULL) {
        fprintf(stderr, "failed to create cnn_rnn_react network definition\n");
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

    printf("cnn_rnn_react crossing demo code generated\n");
    printf("network hash: 0x%016llx\n", (unsigned long long)result.network_hash);
    printf("metadata: %s\n", result.metadata_written_path);

    nn_network_def_free(network);
    return 0;
}
