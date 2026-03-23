/**
 * @file generate_main.c
 * @brief SevenSeg Demo code generation entry
 *
 * Demonstrates usage of profiler_generate_v2() with structured
 * network definition and modular output paths.
 */

#include "profiler.h"
#include "network_def.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Create SevenSeg network definition
 *
 * Network structure:
 * - Input: 10 nodes (one-hot encoding for digits 0-9)
 * - Hidden layers: [16, 8] with ReLU activation
 * - Output: 7 nodes (segment states)
 */
static NN_NetworkDef* create_sevenseg_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* subnet;
    size_t hidden_sizes[2] = {16U, 8U};

    network = nn_network_def_create("sevenseg");
    if (network == NULL) {
        fprintf(stderr, "Failed to create network definition\n");
        return NULL;
    }

    subnet = nn_subnet_def_create("main", "mlp", 10U, 7U);
    if (subnet == NULL) {
        fprintf(stderr, "Failed to create subnet definition\n");
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
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("SevenSeg Network Code Generator\n");
    printf("================================\n\n");

    network = create_sevenseg_network();
    if (network == NULL) {
        fprintf(stderr, "Failed to create SevenSeg network definition\n");
        return 1;
    }

    printf("Network: %s\n", network->network_name);
    printf("Subnets: %zu\n", (size_t)network->subnet_count);

    memset(&req, 0, sizeof(req));
    req.network_def = (const void*)network;
    req.error.buffer = error_buffer;
    req.error.capacity = sizeof(error_buffer);
    req.output_layout = layout;

    printf("\nGenerating code to ../../data/ relative to executable ...\n");

    st = profiler_generate_v2(&req, &result);
    if (st != PROF_STATUS_OK) {
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
