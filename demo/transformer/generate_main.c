/**
 * @file generate_main.c
 * @brief Transformer Demo code generation entry
 */

#include "profiler.h"
#include "network_def.h"
#include "types/transformer/transformer_infer_ops.h"
#include "types/transformer/transformer_train_ops.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>

/**
 * @brief Create Transformer network definition
 */
static NN_NetworkDef* create_transformer_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* subnet;
    size_t hidden_sizes[2] = {32U, 32U};
    TransformerModelConfig infer_config;
    TransformerTrainConfig train_config;

    network = nn_network_def_create("transformer");
    if (network == NULL) {
        return NULL;
    }

    subnet = nn_subnet_def_create("dialogue", "transformer", 32U, 32U);
    if (subnet == NULL) {
        nn_network_def_free(network);
        return NULL;
    }

    nn_subnet_def_set_hidden_layers(subnet, 2U, hidden_sizes);
    infer_config.model_dim = 32U;
    infer_config.seed = 42U;
    train_config.learning_rate = 0.002f;
    if (nn_subnet_def_set_infer_type_config(
            subnet,
            &infer_config,
            sizeof(infer_config),
            "types/transformer/transformer_infer_ops.h",
            "TransformerModelConfig") != 0) {
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }
    if (nn_subnet_def_set_train_type_config(
            subnet,
            &train_config,
            sizeof(train_config),
            "types/transformer/transformer_train_ops.h",
            "TransformerTrainConfig") != 0) {
        nn_subnet_def_free(subnet);
        nn_network_def_free(network);
        return NULL;
    }
    nn_network_def_add_subnet(network, subnet);

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

    network = create_transformer_network();
    if (network == NULL) {
        fprintf(stderr, "failed to create transformer network definition\n");
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

    printf("transformer code generated\n");
    printf("network hash: 0x%016llx\n", (unsigned long long)result.network_hash);
    printf("metadata: %s\n", result.metadata_written_path);

    nn_network_def_free(network);
    return 0;
}
