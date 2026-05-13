/**
 * @file generate_main.c
 * @brief Weather prediction demo — code generation entry.
 *
 * Defines one small MLP for next-day weather prediction using a 7-day
 * sliding window (28 inputs -> 8 hidden -> 4 outputs).
 * The same architecture is shared by all three cities (Beijing, Shanghai,
 * New York); each city trains its own weight file.
 */

#include "profiler.h"
#include "network_def.h"
#include "types/mlp/mlp_config.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WEATHER_INPUT_SIZE  28U
#define WEATHER_OUTPUT_SIZE 4U
#define HIDDEN_LAYER_COUNT  1U
#define HIDDEN_SIZE         8U

static NN_NetworkDef* create_weather_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* subnet;
    size_t hidden_sizes[1] = { HIDDEN_SIZE };
    MlpConfig* infer_config;
    MlpTrainConfig train_config;

    network = nn_network_def_create("weather");
    if (network == NULL) {
        fprintf(stderr, "Failed to create weather network definition\n");
        return NULL;
    }

    subnet = nn_subnet_def_create("predictor", "mlp",
        WEATHER_INPUT_SIZE, WEATHER_OUTPUT_SIZE);
    if (subnet == NULL) {
        fprintf(stderr, "Failed to create weather subnet definition\n");
        nn_network_def_free(network);
        return NULL;
    }

    if (nn_subnet_def_set_hidden_layers(subnet, HIDDEN_LAYER_COUNT, hidden_sizes) != 0) {
        fprintf(stderr, "Failed to attach hidden-layer metadata\n");
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }

    infer_config = mlp_config_create(HIDDEN_LAYER_COUNT);
    if (infer_config == NULL) {
        fprintf(stderr, "Failed to allocate MLP infer config\n");
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }

    /* Regression: ReLU hidden, linear (NONE) output. */
    if (mlp_config_init(
            infer_config,
            WEATHER_INPUT_SIZE,
            HIDDEN_LAYER_COUNT,
            hidden_sizes,
            WEATHER_OUTPUT_SIZE,
            MLP_ACT_RELU,
            MLP_ACT_NONE) != 0) {
        fprintf(stderr, "Failed to initialize MLP infer config\n");
        free(infer_config);
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }

    (void)memset(&train_config, 0, sizeof(train_config));
    train_config.learning_rate = 0.001f;
    train_config.momentum      = 0.9f;
    train_config.weight_decay  = 0.00001f;
    train_config.optimizer     = MLP_OPT_ADAM;
    train_config.loss_func     = MLP_LOSS_MSE;
    train_config.batch_size    = 1U;
    train_config.seed          = 42U;

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
        fprintf(stderr, "Failed to append subnet to weather network\n");
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

    printf("Weather Network Code Generator\n");
    printf("==============================\n\n");

    network = create_weather_network();
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

    nn_network_def_free(network);
    return 0;
}
