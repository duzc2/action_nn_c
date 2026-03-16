#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/infer/include/workflow_infer.h"
#include "sevenseg_shared.h"

static void build_path(char* out_path, size_t cap, const char* dir, const char* file_name) {
    (void)snprintf(out_path, cap, "%s/%s", dir, file_name);
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
            if (seg[i] != g_sevenseg_truth[digit][i]) {
                mismatches++;
            }
        }
    }
    return mismatches;
}

int main(int argc, char** argv) {
    const char* data_dir = "demo/sevenseg/data";
    char vocab_path[260];
    char bin_path[260];
    WorkflowRuntime runtime;
    char line[64];
    int rc = 0;
    memset(&runtime, 0, sizeof(runtime));
    if (argc >= 2) {
        data_dir = argv[1];
    }
    build_path(vocab_path, sizeof(vocab_path), data_dir, "demo_vocab_sevenseg.txt");
    build_path(bin_path, sizeof(bin_path), data_dir, "demo_weights_sevenseg.bin");
    rc = workflow_runtime_init(&runtime, vocab_path, bin_path);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_runtime_init failed: %d\n", rc);
        return 1;
    }
    rc = self_check(&runtime);
    printf("sevenseg bin self-check: mismatched segments=%d\n", rc);
    printf("SevenSeg Infer(BIN): input one digit (0-9), 'q' to quit.\n");
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
        sevenseg_render_cli(d, seg);
    }
    workflow_runtime_shutdown(&runtime);
    return 0;
}
