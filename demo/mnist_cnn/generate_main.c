/**
 * @file generate_main.c
 * @brief MNIST CNN demo code generation entry.
 *
 * This demo uses a small CNN encoder over four 14x14 quadrants, followed by an
 * MLP classification head. It is kept as a separate demo so the original MLP
 * MNIST flow remains untouched.
 */

#include "profiler.h"
#include "network_def.h"
#include "types/cnn_dual_pool/cnn_dual_pool_config.h"
#include "types/mlp/mlp_config.h"
#include "../demo_runtime_paths.h"
#include "mnist_cnn_dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MNIST_CNN_ENCODER_FEATURE_SIZE 12U
#define MNIST_CNN_ENCODER_OUTPUT_SIZE (MNIST_CNN_SEQUENCE_LENGTH * MNIST_CNN_ENCODER_FEATURE_SIZE)

static void fill_cnn_infer_config(CnnDualPoolConfig* config) {
    (void)memset(config, 0, sizeof(*config));
    config->total_input_size = MNIST_CNN_FEATURE_INPUT_SIZE;
    config->sequence_length = MNIST_CNN_SEQUENCE_LENGTH;
    config->frame_width = MNIST_CNN_QUADRANT_COLS;
    config->frame_height = MNIST_CNN_QUADRANT_ROWS;
    config->channel_count = 1U;
    config->kernel_size = 3U;
    config->filter_count = 16U;
    config->feature_size = MNIST_CNN_ENCODER_FEATURE_SIZE;
    config->pooling_activation = CNN_DUAL_POOL_ACT_RELU;
    config->output_activation = CNN_DUAL_POOL_ACT_RELU;
    config->seed = 31U;
}

static void fill_cnn_train_config(CnnDualPoolTrainConfig* config) {
    (void)memset(config, 0, sizeof(*config));
    config->learning_rate = 0.0035f;
    config->momentum = 0.0f;
    config->weight_decay = 0.00008f;
    config->batch_size = 1U;
    config->seed = 31U;
}

static NNSubnetDef* create_cnn_leaf(void) {
    NNSubnetDef* subnet;
    CnnDualPoolConfig infer_config;
    CnnDualPoolTrainConfig train_config;
    size_t hidden_sizes[1] = {16U};

    subnet = nn_subnet_def_create(
        "cnn_encoder",
        "cnn_dual_pool",
        MNIST_CNN_FEATURE_INPUT_SIZE,
        MNIST_CNN_ENCODER_OUTPUT_SIZE
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
            "types/cnn_dual_pool/cnn_dual_pool_config.h",
            "CnnDualPoolConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    if (nn_subnet_def_set_train_type_config(
            subnet,
            &train_config,
            sizeof(train_config),
            "types/cnn_dual_pool/cnn_dual_pool_config.h",
            "CnnDualPoolTrainConfig") != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    return subnet;
}

static NNSubnetDef* create_mlp_head(void) {
    NNSubnetDef* subnet;
    size_t hidden_sizes[1] = {32U};
    MlpConfig* infer_config;
    MlpTrainConfig train_config;

    subnet = nn_subnet_def_create(
        "mlp_head",
        "mlp",
        MNIST_CNN_ENCODER_OUTPUT_SIZE,
        MNIST_CNN_CLASS_COUNT
    );
    if (subnet == NULL) {
        return NULL;
    }

    if (nn_subnet_def_set_hidden_layers(subnet, 1U, hidden_sizes) != 0) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    infer_config = mlp_config_create(1U);
    if (infer_config == NULL) {
        nn_subnet_def_free(subnet);
        return NULL;
    }

    if (mlp_config_init(
            infer_config,
            MNIST_CNN_ENCODER_OUTPUT_SIZE,
            1U,
            hidden_sizes,
            MNIST_CNN_CLASS_COUNT,
            MLP_ACT_RELU,
            MLP_ACT_SOFTMAX) != 0) {
        free(infer_config);
        nn_subnet_def_free(subnet);
        return NULL;
    }

    (void)memset(&train_config, 0, sizeof(train_config));
    train_config.learning_rate = 0.0012f;
    train_config.momentum = 0.9f;
    train_config.weight_decay = 0.00005f;
    train_config.optimizer = MLP_OPT_ADAM;
    train_config.loss_func = MLP_LOSS_CROSS_ENTROPY;
    train_config.batch_size = 1U;
    train_config.seed = 37U;

    if (nn_subnet_def_set_infer_type_config(
            subnet,
            infer_config,
            mlp_config_size_for_hidden_layers(infer_config->hidden_layer_count),
            "types/mlp/mlp_config.h",
            "MlpConfig") != 0) {
        free(infer_config);
        nn_subnet_def_free(subnet);
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
        return NULL;
    }

    return subnet;
}

static NN_NetworkDef* create_mnist_cnn_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* encoder;
    NNSubnetDef* head;
    size_t feature_index;

    network = nn_network_def_create("mnist_cnn");
    if (network == NULL) {
        return NULL;
    }

    encoder = create_cnn_leaf();
    head = create_mlp_head();
    if (encoder == NULL || head == NULL) {
        nn_subnet_def_free(encoder);
        nn_subnet_def_free(head);
        nn_network_def_free(network);
        return NULL;
    }

    if (nn_network_def_add_subnet(network, encoder) != 0 ||
        nn_network_def_add_subnet(network, head) != 0) {
        nn_subnet_def_free(encoder);
        nn_subnet_def_free(head);
        nn_network_def_free(network);
        return NULL;
    }

    for (feature_index = 0U; feature_index < MNIST_CNN_ENCODER_OUTPUT_SIZE; ++feature_index) {
        NNConnectionDef* connection = nn_connection_def_create(
            "cnn_encoder",
            "out",
            feature_index,
            "mlp_head",
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

    printf("MNIST CNN+MLP Network Code Generator\n");
    printf("====================================\n\n");

    network = create_mnist_cnn_network();
    if (network == NULL) {
        fprintf(stderr, "Failed to create MNIST CNN network definition\n");
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
