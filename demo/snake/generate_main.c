/**
 * @file generate_main.c
 * @brief Snake Demo code generation entry
 *
 * Creates an 8->[6]->4 MLP network definition for the snake game.
 * Follows the move/generate_main.c pattern exactly.
 */

#include "profiler.h"
#include "network_def.h"
#include "types/mlp/mlp_config.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Create Snake network definition
 * Architecture: 8 inputs -> [6 hidden] -> 4 outputs
 * Activations: TANH (hidden), SOFTMAX (output)
 */
static NN_NetworkDef* create_snake_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* subnet;
    size_t hidden_sizes[1] = {6U};
    MlpConfig* infer_config;
    MlpTrainConfig train_config;

    network = nn_network_def_create("snake");
    if (network == NULL) {
        return NULL;
    }

    subnet = nn_subnet_def_create("main", "mlp", 8U, 4U);
    if (subnet == NULL) {
        nn_network_def_free(network);
        return NULL;
    }

    if (nn_subnet_def_set_hidden_layers(subnet, 1U, hidden_sizes) != 0) {
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }

    infer_config = mlp_config_create(1U);
    if (infer_config == NULL ||
        mlp_config_init(
            infer_config,
            8U,
            1U,
            hidden_sizes,
            4U,
            MLP_ACT_TANH,
            MLP_ACT_SOFTMAX
        ) != 0) {
        free(infer_config);
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }

    train_config.learning_rate = 0.003f;
    train_config.momentum = 0.9f;
    train_config.weight_decay = 0.0001f;
    train_config.optimizer = MLP_OPT_ADAM;
    train_config.loss_func = MLP_LOSS_MSE;
    train_config.batch_size = 1U;
    train_config.seed = 42U;

    if (nn_subnet_def_set_infer_type_config(
            subnet,
            infer_config,
            mlp_config_size_for_hidden_layers(infer_config->hidden_layer_count),
            "types/mlp/mlp_config.h",
            "MlpConfig") != 0) {
        free(infer_config);
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }
    free(infer_config);

    if (nn_subnet_def_set_train_type_config(
            subnet,
            &train_config,
            sizeof(train_config),
            "types/mlp/mlp_config.h",
            "MlpTrainConfig") != 0) {
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }

    if (nn_network_def_add_subnet(network, subnet) != 0) {
        nn_subnet_def_free(subnet);
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

    network = create_snake_network();
    if (network == NULL) {
        fprintf(stderr, "failed to create snake network definition\n");
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

    printf("snake code generated\n");
    printf("network hash: 0x%016llx\n", (unsigned long long)result.network_hash);
    printf("metadata: %s\n", result.metadata_written_path);

    nn_network_def_free(network);
    return 0;
}
