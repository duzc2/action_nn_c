#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "../../src/include/config_user.h"
#include "../../src/include/weights_io.h"
#include "../../src/infer/include/workflow_infer.h"
#include "../../src/train/include/workflow_train.h"

enum {
    MAX_SEVENSEG_SAMPLES = 256
};

static const int kSevenSegTruth[10][8] = {
    { 1,1,1,1,1,1,0,0 },
    { 0,1,1,0,0,0,0,0 },
    { 1,1,0,1,1,0,1,0 },
    { 1,1,1,1,0,0,1,0 },
    { 0,1,1,0,0,1,1,0 },
    { 1,0,1,1,0,1,1,0 },
    { 1,0,1,1,1,1,1,0 },
    { 1,1,1,0,0,0,0,0 },
    { 1,1,1,1,1,1,1,0 },
    { 1,1,1,1,0,1,1,0 }
};

static int create_dir_if_missing(const char* path) {
#if defined(_WIN32)
    int rc = _mkdir(path);
#else
    int rc = mkdir(path, 0755);
#endif
    if (rc == 0 || errno == EEXIST) {
        return 0;
    }
    return -1;
}

static int ensure_sevenseg_data_dir(void) {
    if (create_dir_if_missing("demo") != 0) {
        return -1;
    }
    if (create_dir_if_missing("demo/sevenseg") != 0) {
        return -2;
    }
    if (create_dir_if_missing("demo/sevenseg/data") != 0) {
        return -3;
    }
    return 0;
}

static int write_rows_to_text(const char* file_path, const char* const* rows, size_t row_count) {
    FILE* fp = NULL;
    size_t i = 0U;
    if (file_path == NULL || rows == NULL || row_count == 0U) {
        return -1;
    }
    fp = fopen(file_path, "w");
    if (fp == NULL) {
        return -2;
    }
    for (i = 0U; i < row_count; ++i) {
        if (fprintf(fp, "%s\n", rows[i]) < 0) {
            fclose(fp);
            return -3;
        }
    }
    fclose(fp);
    return 0;
}

static int write_vocab(const char* file_path) {
    static const char* tokens[] = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    };
    return write_rows_to_text(file_path, tokens, sizeof(tokens) / sizeof(tokens[0]));
}

static int segbit(const int segs[8], size_t index) {
    return (segs[index] != 0) ? 1 : 0;
}

static float map_pin_to_target(size_t out_index, int pin_on) {
    if (out_index == 0U || out_index == 1U) {
        return pin_on ? 1.0f : 0.0f;
    }
    return pin_on ? 1.0f : -1.0f;
}

static void build_expected_bank_targets(int digit, int bank, float out_target[OUTPUT_DIM]) {
    size_t i = 0U;
    size_t base = (bank == 0) ? 0U : 4U;
    for (i = 0U; i < OUTPUT_DIM; ++i) {
        out_target[i] = map_pin_to_target(i, segbit(kSevenSegTruth[digit], base + i));
    }
}

static int build_sevenseg_samples(WorkflowTrainSample* out_samples,
                                  char commands[][32],
                                  float states[][STATE_DIM],
                                  float targets[][OUTPUT_DIM],
                                  size_t capacity,
                                  size_t* out_count) {
    size_t idx = 0U;
    int digit = 0;
    int repeat = 0;
    int bank = 0;
    if (out_samples == NULL || commands == NULL || states == NULL || targets == NULL || out_count == NULL) {
        return -1;
    }
    for (digit = 0; digit <= 9; ++digit) {
        for (repeat = 0; repeat < 10; ++repeat) {
            for (bank = 0; bank <= 1; ++bank) {
                if (idx >= capacity) {
                    return -2;
                }
                (void)snprintf(commands[idx], 32U, "%d", digit);
                memset(states[idx], 0, sizeof(float) * STATE_DIM);
                memset(targets[idx], 0, sizeof(float) * OUTPUT_DIM);
                states[idx][0] = (float)bank;
                states[idx][1] = (float)digit / 9.0f;
                build_expected_bank_targets(digit, bank, targets[idx]);
                out_samples[idx].command = commands[idx];
                out_samples[idx].state = states[idx];
                out_samples[idx].target = targets[idx];
                idx++;
            }
        }
    }
    *out_count = idx;
    return 0;
}

static int verify_generated_samples(const WorkflowTrainSample* samples, size_t sample_count) {
    size_t i = 0U;
    if (samples == NULL || sample_count == 0U) {
        return -1;
    }
    for (i = 0U; i < sample_count; ++i) {
        int digit = -1;
        int bank = 0;
        float expected[OUTPUT_DIM];
        size_t k = 0U;
        if (samples[i].command == NULL || samples[i].state == NULL || samples[i].target == NULL) {
            return -2;
        }
        if (sscanf(samples[i].command, "%d", &digit) != 1 || digit < 0 || digit > 9) {
            return -3;
        }
        bank = (samples[i].state[0] >= 0.5f) ? 1 : 0;
        build_expected_bank_targets(digit, bank, expected);
        for (k = 0U; k < OUTPUT_DIM; ++k) {
            if (fabsf(samples[i].target[k] - expected[k]) > 0.0001f) {
                return -4;
            }
        }
    }
    return 0;
}

static int decode_bank_output(const float* act, int* out_bits4) {
    if (act == NULL || out_bits4 == NULL) {
        return -1;
    }
    out_bits4[0] = (act[0] >= 0.5f) ? 1 : 0;
    out_bits4[1] = (act[1] >= 0.5f) ? 1 : 0;
    out_bits4[2] = (act[2] >= 0.0f) ? 1 : 0;
    out_bits4[3] = (act[3] >= 0.0f) ? 1 : 0;
    return 0;
}

static int infer_digit_segments(WorkflowRuntime* runtime, int digit, int out_seg[8]) {
    char command[32];
    float state[STATE_DIM];
    float act[OUTPUT_DIM];
    int bank_bits[4];
    int rc = 0;
    if (runtime == NULL || out_seg == NULL || digit < 0 || digit > 9) {
        return -1;
    }
    memset(out_seg, 0, sizeof(int) * 8);
    (void)snprintf(command, sizeof(command), "%d", digit);
    memset(state, 0, sizeof(state));
    memset(act, 0, sizeof(act));
    state[0] = 0.0f;
    state[1] = (float)digit / 9.0f;
    rc = workflow_run_step(runtime, command, state, act);
    if (rc != WORKFLOW_STATUS_OK) {
        return -2;
    }
    rc = decode_bank_output(act, bank_bits);
    if (rc != 0) {
        return -3;
    }
    out_seg[0] = bank_bits[0];
    out_seg[1] = bank_bits[1];
    out_seg[2] = bank_bits[2];
    out_seg[3] = bank_bits[3];
    memset(state, 0, sizeof(state));
    memset(act, 0, sizeof(act));
    state[0] = 1.0f;
    state[1] = (float)digit / 9.0f;
    rc = workflow_run_step(runtime, command, state, act);
    if (rc != WORKFLOW_STATUS_OK) {
        return -4;
    }
    rc = decode_bank_output(act, bank_bits);
    if (rc != 0) {
        return -5;
    }
    out_seg[4] = bank_bits[0];
    out_seg[5] = bank_bits[1];
    out_seg[6] = bank_bits[2];
    out_seg[7] = bank_bits[3];
    return 0;
}

static int verify_all_digits_outputs(WorkflowRuntime* runtime) {
    int digit = 0;
    int mismatches = 0;
    for (digit = 0; digit <= 9; ++digit) {
        int seg[8];
        int rc = infer_digit_segments(runtime, digit, seg);
        int i = 0;
        if (rc != 0) {
            return -1;
        }
        for (i = 0; i < 8; ++i) {
            if (seg[i] != kSevenSegTruth[digit][i]) {
                mismatches++;
            }
        }
    }
    return mismatches;
}

static void render_sevenseg_cli(int digit, const int seg[8]) {
    char a = seg[0] ? '-' : ' ';
    char b = seg[1] ? '|' : ' ';
    char c = seg[2] ? '|' : ' ';
    char d = seg[3] ? '-' : ' ';
    char e = seg[4] ? '|' : ' ';
    char f = seg[5] ? '|' : ' ';
    char g = seg[6] ? '-' : ' ';
    char dp = seg[7] ? '.' : ' ';
    printf("\ninput digit: %d\n", digit);
    printf("pins[a b c d e f g dp] = [%d %d %d %d %d %d %d %d]\n",
           seg[0], seg[1], seg[2], seg[3], seg[4], seg[5], seg[6], seg[7]);
    printf(" %c%c%c \n", a, a, a);
    printf("%c   %c\n", f, b);
    printf("%c   %c\n", f, b);
    printf(" %c%c%c \n", g, g, g);
    printf("%c   %c\n", e, c);
    printf("%c   %c\n", e, c);
    printf(" %c%c%c  %c\n", d, d, d, dp);
}

static int run_sevenseg_cli_demo(WorkflowRuntime* runtime) {
    char line[64];
    if (runtime == NULL) {
        return -1;
    }
    printf("SevenSeg CLI Demo: input one digit (0-9), 'q' to quit.\n");
    for (;;) {
        int digit = -1;
        int seg[8];
        int rc = 0;
        printf("digit> ");
        fflush(stdout);
        if (fgets(line, sizeof(line), stdin) == NULL) {
            clearerr(stdin);
            return 0;
        }
        if (line[0] == 'q' || line[0] == 'Q') {
            return 0;
        }
        if (sscanf(line, "%d", &digit) != 1 || digit < 0 || digit > 9) {
            printf("invalid input, please enter 0-9 or q.\n");
            continue;
        }
        rc = infer_digit_segments(runtime, digit, seg);
        if (rc != 0) {
            return -2;
        }
        render_sevenseg_cli(digit, seg);
    }
}

int main(void) {
    const char* vocab_path = "demo/sevenseg/data/demo_vocab_sevenseg.txt";
    const char* weight_bin_path = "demo/sevenseg/data/demo_weights_sevenseg.bin";
    const char* weight_c_path = "demo/sevenseg/data/demo_weights_sevenseg_export.c";
    const char* weight_fn_c_path = "demo/sevenseg/data/demo_network_sevenseg_functions.c";
    const char* symbol = "g_demo_weights_sevenseg";
    const int activations[OUTPUT_DIM] = IO_MAPPING_ACTIVATIONS;
    WorkflowRuntime runtime;
    WorkflowTrainMemoryOptions train_options;
    WorkflowTrainSample train_samples[MAX_SEVENSEG_SAMPLES];
    char train_commands[MAX_SEVENSEG_SAMPLES][32];
    float train_states[MAX_SEVENSEG_SAMPLES][STATE_DIM];
    float train_targets[MAX_SEVENSEG_SAMPLES][OUTPUT_DIM];
    size_t train_count = 0U;
    float* loaded_weights = NULL;
    size_t loaded_count = 0U;
    int rc = 0;
    memset(&runtime, 0, sizeof(runtime));
    memset(&train_options, 0, sizeof(train_options));
    rc = ensure_sevenseg_data_dir();
    if (rc != 0) {
        fprintf(stderr, "ensure_sevenseg_data_dir failed: %d\n", rc);
        return 1;
    }
    rc = write_vocab(vocab_path);
    if (rc != 0) {
        fprintf(stderr, "write_vocab failed: %d\n", rc);
        return 2;
    }
    rc = build_sevenseg_samples(train_samples,
                                train_commands,
                                train_states,
                                train_targets,
                                MAX_SEVENSEG_SAMPLES,
                                &train_count);
    if (rc != 0 || train_count == 0U) {
        fprintf(stderr, "build_sevenseg_samples failed: %d\n", rc);
        return 3;
    }
    rc = verify_generated_samples(train_samples, train_count);
    if (rc != 0) {
        fprintf(stderr, "verify_generated_samples failed: %d\n", rc);
        return 4;
    }
    train_options.samples = train_samples;
    train_options.sample_count = train_count;
    train_options.vocab_path = vocab_path;
    train_options.out_weights_bin = weight_bin_path;
    train_options.out_weights_c = weight_c_path;
    train_options.out_symbol = symbol;
    train_options.epochs = 120U;
    train_options.learning_rate = 0.08f;
    rc = workflow_train_from_memory(&train_options);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_train_from_memory failed: %d\n", rc);
        return 5;
    }
    rc = weights_load_binary(weight_bin_path, &loaded_weights, &loaded_count);
    if (rc != WEIGHTS_IO_STATUS_OK || loaded_weights == NULL || loaded_count != workflow_weights_count()) {
        fprintf(stderr, "weights_load_binary failed: rc=%d count=%zu\n", rc, loaded_count);
        free(loaded_weights);
        return 6;
    }
    rc = weights_export_c_function_network(weight_fn_c_path,
                                           "g_demo_network_sevenseg",
                                           loaded_weights,
                                           VOCAB_SIZE,
                                           MAX_SEQ_LEN,
                                           STATE_DIM,
                                           OUTPUT_DIM,
                                           activations);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        fprintf(stderr, "weights_export_c_function_network failed: %d\n", rc);
        free(loaded_weights);
        return 7;
    }
    rc = workflow_runtime_init(&runtime, vocab_path, weight_bin_path);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_runtime_init failed: %d\n", rc);
        free(loaded_weights);
        return 8;
    }
    rc = verify_all_digits_outputs(&runtime);
    if (rc < 0) {
        fprintf(stderr, "verify_all_digits_outputs failed: %d\n", rc);
        workflow_runtime_shutdown(&runtime);
        free(loaded_weights);
        return 9;
    }
    printf("sevenseg self-check: mismatched segments=%d\n", rc);
    rc = run_sevenseg_cli_demo(&runtime);
    if (rc != 0) {
        fprintf(stderr, "run_sevenseg_cli_demo failed: %d\n", rc);
        workflow_runtime_shutdown(&runtime);
        free(loaded_weights);
        return 10;
    }
    workflow_runtime_shutdown(&runtime);
    free(loaded_weights);
    printf("sevenseg demo completed\n");
    return 0;
}
