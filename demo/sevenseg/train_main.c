#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/include/config_user.h"
#include "../../src/include/weights_io.h"
#include "../../src/train/include/workflow_train.h"
#include "sevenseg_shared.h"

static void build_path(char* out_path, size_t cap, const char* dir, const char* file_name) {
    (void)snprintf(out_path, cap, "%s/%s", dir, file_name);
}

int main(int argc, char** argv) {
    const char* out_dir = "./data";
    char vocab_path[260];
    char bin_path[260];
    char weights_c_path[260];
    char network_fn_path[260];
    const int activations[OUTPUT_DIM] = IO_MAPPING_ACTIVATIONS;
    WorkflowTrainMemoryOptions train_options;
    WorkflowTrainSample train_samples[MAX_SEVENSEG_SAMPLES];
    char train_commands[MAX_SEVENSEG_SAMPLES][32];
    float train_states[MAX_SEVENSEG_SAMPLES][STATE_DIM];
    float train_targets[MAX_SEVENSEG_SAMPLES][OUTPUT_DIM];
    float* loaded = NULL;
    size_t loaded_count = 0U;
    size_t train_count = 0U;
    int rc = 0;
    if (argc >= 3 && strcmp(argv[1], "--export-only") == 0) {
        out_dir = argv[2];
    }
    memset(&train_options, 0, sizeof(train_options));
    build_path(vocab_path, sizeof(vocab_path), out_dir, "demo_vocab_sevenseg.txt");
    build_path(bin_path, sizeof(bin_path), out_dir, "demo_weights_sevenseg.bin");
    build_path(weights_c_path, sizeof(weights_c_path), out_dir, "demo_weights_sevenseg_export.c");
    build_path(network_fn_path, sizeof(network_fn_path), out_dir, "demo_network_sevenseg_functions.c");
    rc = sevenseg_ensure_dir("demo");
    if (rc != 0) {
        return 1;
    }
    rc = sevenseg_ensure_dir(out_dir);
    if (rc != 0) {
        return 3;
    }
    rc = sevenseg_write_vocab(vocab_path);
    if (rc != 0) {
        return 4;
    }
    rc = sevenseg_build_samples(train_samples,
                                train_commands,
                                train_states,
                                train_targets,
                                MAX_SEVENSEG_SAMPLES,
                                &train_count);
    if (rc != 0 || train_count == 0U) {
        return 5;
    }
    rc = sevenseg_verify_samples(train_samples, train_count);
    if (rc != 0) {
        return 6;
    }
    train_options.samples = train_samples;
    train_options.sample_count = train_count;
    train_options.vocab_path = vocab_path;
    train_options.out_weights_bin = bin_path;
    train_options.out_weights_c = weights_c_path;
    train_options.out_symbol = "g_demo_weights_sevenseg";
    train_options.epochs = 140U;
    train_options.learning_rate = 0.06f;
    rc = workflow_train_from_memory(&train_options);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_train_from_memory failed: %d\n", rc);
        return 7;
    }
    rc = weights_load_binary(bin_path, &loaded, &loaded_count);
    if (rc != WEIGHTS_IO_STATUS_OK || loaded == NULL || loaded_count != workflow_weights_count()) {
        free(loaded);
        return 8;
    }
    rc = weights_export_c_function_network(network_fn_path,
                                           "g_demo_network_sevenseg",
                                           loaded,
                                           VOCAB_SIZE,
                                           MAX_SEQ_LEN,
                                           STATE_DIM,
                                           OUTPUT_DIM,
                                           activations);
    free(loaded);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        return 9;
    }
    printf("sevenseg training/export done\n");
    return 0;
}
