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
    MAX_SEVENSEG_SAMPLES = 512
};

static const int kSevenSegTruth[10][7] = {
    { 1,1,1,1,1,1,0 },
    { 0,1,1,0,0,0,0 },
    { 1,1,0,1,1,0,1 },
    { 1,1,1,1,0,0,1 },
    { 0,1,1,0,0,1,1 },
    { 1,0,1,1,0,1,1 },
    { 1,0,1,1,1,1,1 },
    { 1,1,1,0,0,0,0 },
    { 1,1,1,1,1,1,1 },
    { 1,1,1,1,0,1,1 }
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

static int ensure_data_dir(void) {
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

static int segbit(int digit, int idx) {
    return (kSevenSegTruth[digit][idx] != 0) ? 1 : 0;
}

static float map_target(int pin_on) {
    return pin_on ? 1.0f : 0.0f;
}

static int build_samples(WorkflowTrainSample* out_samples,
                         char commands[][32],
                         float states[][STATE_DIM],
                         float targets[][OUTPUT_DIM],
                         size_t capacity,
                         size_t* out_count) {
    size_t idx = 0U;
    int digit = 0;
    int repeat = 0;
    if (out_samples == NULL || commands == NULL || states == NULL || targets == NULL || out_count == NULL) {
        return -1;
    }
    for (digit = 0; digit <= 9; ++digit) {
        for (repeat = 0; repeat < 32; ++repeat) {
            int i = 0;
            if (idx >= capacity) {
                return -2;
            }
            (void)snprintf(commands[idx], 32U, "%d", digit);
            memset(states[idx], 0, sizeof(float) * STATE_DIM);
            memset(targets[idx], 0, sizeof(float) * OUTPUT_DIM);
            states[idx][0] = (float)digit / 9.0f;
            for (i = 0; i < 7; ++i) {
                targets[idx][i] = map_target(segbit(digit, i));
            }
            out_samples[idx].command = commands[idx];
            out_samples[idx].state = states[idx];
            out_samples[idx].target = targets[idx];
            idx++;
        }
    }
    *out_count = idx;
    return 0;
}

static int verify_samples(const WorkflowTrainSample* samples, size_t sample_count) {
    size_t i = 0U;
    if (samples == NULL || sample_count == 0U) {
        return -1;
    }
    for (i = 0U; i < sample_count; ++i) {
        int digit = -1;
        int k = 0;
        if (samples[i].command == NULL || samples[i].state == NULL || samples[i].target == NULL) {
            return -2;
        }
        if (sscanf(samples[i].command, "%d", &digit) != 1 || digit < 0 || digit > 9) {
            return -3;
        }
        for (k = 0; k < 7; ++k) {
            float expected = map_target(segbit(digit, k));
            if (fabsf(samples[i].target[k] - expected) > 0.0001f) {
                return -4;
            }
        }
    }
    return 0;
}

static int infer_digit(WorkflowRuntime* runtime, int digit, int out_seg[7]) {
    char command[32];
    float state[STATE_DIM];
    float act[OUTPUT_DIM];
    int i = 0;
    int rc = 0;
    if (runtime == NULL || out_seg == NULL || digit < 0 || digit > 9) {
        return -1;
    }
    (void)snprintf(command, sizeof(command), "%d", digit);
    memset(state, 0, sizeof(state));
    memset(act, 0, sizeof(act));
    state[0] = (float)digit / 9.0f;
    rc = workflow_run_step(runtime, command, state, act);
    if (rc != WORKFLOW_STATUS_OK) {
        return -2;
    }
    for (i = 0; i < 7; ++i) {
        out_seg[i] = (act[i] >= 0.5f) ? 1 : 0;
    }
    return 0;
}

static int self_check(WorkflowRuntime* runtime) {
    int digit = 0;
    int mismatches = 0;
    for (digit = 0; digit <= 9; ++digit) {
        int seg[7];
        int i = 0;
        if (infer_digit(runtime, digit, seg) != 0) {
            return -1;
        }
        for (i = 0; i < 7; ++i) {
            if (seg[i] != kSevenSegTruth[digit][i]) {
                mismatches++;
            }
        }
    }
    return mismatches;
}

static void render_cli(int digit, const int seg[7]) {
    char a = seg[0] ? '-' : ' ';
    char b = seg[1] ? '|' : ' ';
    char c = seg[2] ? '|' : ' ';
    char d = seg[3] ? '-' : ' ';
    char e = seg[4] ? '|' : ' ';
    char f = seg[5] ? '|' : ' ';
    char g = seg[6] ? '-' : ' ';
    printf("\ninput digit: %d\n", digit);
    printf("pins[a b c d e f g] = [%d %d %d %d %d %d %d]\n",
           seg[0], seg[1], seg[2], seg[3], seg[4], seg[5], seg[6]);
    printf(" %c%c%c \n", a, a, a);
    printf("%c   %c\n", f, b);
    printf("%c   %c\n", f, b);
    printf(" %c%c%c \n", g, g, g);
    printf("%c   %c\n", e, c);
    printf("%c   %c\n", e, c);
    printf(" %c%c%c \n", d, d, d);
}

int main(void) {
    const char* vocab_path = "demo/sevenseg/data/demo_vocab_sevenseg.txt";
    const char* weight_bin = "demo/sevenseg/data/demo_weights_sevenseg.bin";
    const char* weight_c = "demo/sevenseg/data/demo_weights_sevenseg_export.c";
    const char* weight_fn_c = "demo/sevenseg/data/demo_network_sevenseg_functions.c";
    const int activations[OUTPUT_DIM] = IO_MAPPING_ACTIVATIONS;
    WorkflowRuntime runtime;
    WorkflowTrainMemoryOptions train_options;
    WorkflowTrainSample train_samples[MAX_SEVENSEG_SAMPLES];
    char train_commands[MAX_SEVENSEG_SAMPLES][32];
    float train_states[MAX_SEVENSEG_SAMPLES][STATE_DIM];
    float train_targets[MAX_SEVENSEG_SAMPLES][OUTPUT_DIM];
    float* loaded = NULL;
    size_t loaded_count = 0U;
    size_t train_count = 0U;
    char line[64];
    int rc = 0;
    memset(&runtime, 0, sizeof(runtime));
    memset(&train_options, 0, sizeof(train_options));
    rc = ensure_data_dir();
    if (rc != 0) {
        return 1;
    }
    rc = write_vocab(vocab_path);
    if (rc != 0) {
        return 2;
    }
    rc = build_samples(train_samples, train_commands, train_states, train_targets, MAX_SEVENSEG_SAMPLES, &train_count);
    if (rc != 0 || train_count == 0U) {
        return 3;
    }
    rc = verify_samples(train_samples, train_count);
    if (rc != 0) {
        return 4;
    }
    train_options.samples = train_samples;
    train_options.sample_count = train_count;
    train_options.vocab_path = vocab_path;
    train_options.out_weights_bin = weight_bin;
    train_options.out_weights_c = weight_c;
    train_options.out_symbol = "g_demo_weights_sevenseg";
    train_options.epochs = 140U;
    train_options.learning_rate = 0.06f;
    rc = workflow_train_from_memory(&train_options);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_train_from_memory failed: %d\n", rc);
        return 5;
    }
    rc = weights_load_binary(weight_bin, &loaded, &loaded_count);
    if (rc != WEIGHTS_IO_STATUS_OK || loaded == NULL || loaded_count != workflow_weights_count()) {
        free(loaded);
        return 6;
    }
    rc = weights_export_c_function_network(weight_fn_c,
                                           "g_demo_network_sevenseg",
                                           loaded,
                                           VOCAB_SIZE,
                                           MAX_SEQ_LEN,
                                           STATE_DIM,
                                           OUTPUT_DIM,
                                           activations);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        free(loaded);
        return 7;
    }
    rc = workflow_runtime_init(&runtime, vocab_path, weight_bin);
    if (rc != WORKFLOW_STATUS_OK) {
        free(loaded);
        return 8;
    }
    rc = self_check(&runtime);
    printf("sevenseg self-check: mismatched segments=%d\n", rc);
    printf("SevenSeg CLI Demo: input one digit (0-9), 'q' to quit.\n");
    for (;;) {
        int d = -1;
        int seg[7];
        printf("digit> ");
        fflush(stdout);
        if (fgets(line, sizeof(line), stdin) == NULL) {
            clearerr(stdin);
            break;
        }
        if (line[0] == 'q' || line[0] == 'Q') {
            break;
        }
        if (sscanf(line, "%d", &d) != 1 || d < 0 || d > 9) {
            printf("invalid input, please enter 0-9 or q.\n");
            continue;
        }
        if (infer_digit(&runtime, d, seg) != 0) {
            break;
        }
        render_cli(d, seg);
    }
    workflow_runtime_shutdown(&runtime);
    free(loaded);
    printf("sevenseg demo completed\n");
    return 0;
}
