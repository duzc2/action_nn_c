/**
 * @file generate_main.c
 * @brief MNIST demo code generation entry.
 *
 * The demo keeps user-side code limited to profiler/network-definition APIs and
 * the config-only MLP header, which matches the repository contract for demo
 * generators.
 */

#include "profiler.h"
#include "network_def.h"
#include "types/mlp/mlp_config.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MNIST_INPUT_SIZE 784U
#define MNIST_OUTPUT_SIZE 10U

/**
 * @brief Build one small MLP suitable for 28x28 handwritten digits.
 */
static NN_NetworkDef* create_mnist_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* subnet;
    size_t hidden_sizes[2] = {64U, 32U};
    MlpConfig* infer_config;
    MlpTrainConfig train_config;

    network = nn_network_def_create("mnist");
    if (network == NULL) {
        fprintf(stderr, "Failed to create MNIST network definition\n");
        return NULL;
    }

    subnet = nn_subnet_def_create("classifier", "mlp", MNIST_INPUT_SIZE, MNIST_OUTPUT_SIZE);
    if (subnet == NULL) {
        fprintf(stderr, "Failed to create MNIST subnet definition\n");
        nn_network_def_free(network);
        return NULL;
    }

    if (nn_subnet_def_set_hidden_layers(subnet, 2U, hidden_sizes) != 0) {
        fprintf(stderr, "Failed to attach hidden-layer metadata\n");
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }

    infer_config = mlp_config_create(2U);
    if (infer_config == NULL) {
        fprintf(stderr, "Failed to allocate MLP infer config\n");
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }

    if (mlp_config_init(
            infer_config,
            MNIST_INPUT_SIZE,
            2U,
            hidden_sizes,
            MNIST_OUTPUT_SIZE,
            MLP_ACT_RELU,
            MLP_ACT_SOFTMAX) != 0) {
        fprintf(stderr, "Failed to initialize MLP infer config\n");
        free(infer_config);
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }

    (void)memset(&train_config, 0, sizeof(train_config));
    train_config.learning_rate = 0.0015f;
    train_config.momentum = 0.9f;
    train_config.weight_decay = 0.00005f;
    train_config.optimizer = MLP_OPT_ADAM;
    train_config.loss_func = MLP_LOSS_CROSS_ENTROPY;
    train_config.batch_size = 1U;
    train_config.seed = 42U;

    if (nn_subnet_def_set_infer_type_config(
            subnet,
            infer_config,
            mlp_config_size_for_hidden_layers(infer_config->hidden_layer_count),
            "types/mlp/mlp_config.h",
            "MlpConfig") != 0) {
        fprintf(stderr, "Failed to store infer config on subnet\n");
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
        fprintf(stderr, "Failed to store train config on subnet\n");
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }

    if (nn_network_def_add_subnet(network, subnet) != 0) {
        fprintf(stderr, "Failed to append subnet to MNIST network\n");
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }

    return network;
}

int main(void) {
    NN_NetworkDef* network;
    ProfGenerateRequest request;
    ProfGenerateResult result;
    ProfStatus status;
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
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("MNIST Network Code Generator\n");
    printf("============================\n\n");

    network = create_mnist_network();
    if (network == NULL) {
        return 1;
    }

    printf("Network: %s\n", network->network_name);
    printf("Subnets: %zu\n", (size_t)network->subnet_count);

    (void)memset(&request, 0, sizeof(request));
    request.network_def = (const void*)network;
    request.error.buffer = error_buffer;
    request.error.capacity = sizeof(error_buffer);
    request.output_layout = layout;

    printf("\nGenerating code to ../data/ relative to executable ...\n");
    status = profiler_generate_v2(&request, &result);
    if (status != PROF_STATUS_OK) {
        fprintf(stderr, "Code generation failed: %s\n", error_buffer);
        nn_network_def_free(network);
        return 1;
    }

    printf("\nCode generation successful!\n");
    printf("Network Hash: 0x%016llx\n", (unsigned long long)result.network_hash);
    printf("Metadata: %s\n", result.metadata_written_path);

    nn_network_def_free(network);
    return 0;
}
