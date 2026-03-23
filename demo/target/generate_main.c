/**
 * @file generate_main.c
 * @brief Target Demo code generation entry
 */

#include "profiler.h"
#include "network_def.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>

/**
 * @brief Create Target network definition
 */
static NN_NetworkDef* create_target_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* subnet;
    size_t hidden_sizes[2] = {16U, 8U};

    network = nn_network_def_create("target");
    if (network == NULL) {
        return NULL;
    }

    subnet = nn_subnet_def_create("main", "mlp", 4U, 2U);
    if (subnet == NULL) {
        nn_network_def_free(network);
        return NULL;
    }

    nn_subnet_def_set_hidden_layers(subnet, 2U, hidden_sizes);
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

    network = create_target_network();
    if (network == NULL) {
        fprintf(stderr, "failed to create target network definition\n");
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

    printf("target code generated\n");
    printf("network hash: 0x%016llx\n", (unsigned long long)result.network_hash);
    printf("metadata: %s\n", result.metadata_written_path);

    nn_network_def_free(network);
    return 0;
}
