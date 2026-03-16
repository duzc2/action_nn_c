#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/infer/include/workflow_infer.h"
#include "sevenseg_shared.h"

size_t g_demo_weights_sevenseg_count(void);
int g_demo_weights_sevenseg_copy(float* out, size_t out_capacity);

/**
 * @brief 从编译期数组初始化运行时权重与分词器。
 *
 * 关键保护点：
 * - 任一步失败都释放已申请内存，避免泄漏。
 */
static int runtime_init_from_array(WorkflowRuntime* runtime, const char* vocab_path) {
    size_t count = g_demo_weights_sevenseg_count();
    int rc = 0;
    if (runtime == NULL || vocab_path == NULL) {
        return -1;
    }
    memset(runtime, 0, sizeof(*runtime));
    runtime->weights = (float*)malloc(sizeof(float) * count);
    if (runtime->weights == NULL) {
        return -2;
    }
    rc = g_demo_weights_sevenseg_copy(runtime->weights, count);
    if (rc != 0) {
        free(runtime->weights);
        runtime->weights = NULL;
        return -3;
    }
    runtime->weight_count = count;
    rc = workflow_prepare_tokenizer(vocab_path, &runtime->vocab, &runtime->tokenizer);
    if (rc != WORKFLOW_STATUS_OK) {
        free(runtime->weights);
        runtime->weights = NULL;
        runtime->weight_count = 0U;
        return -4;
    }
    runtime->ready = 1;
    return 0;
}

/**
 * @brief 拼接目录与文件名，构造数据文件路径。
 */
static void build_path(char* out_path, size_t cap, const char* dir, const char* file_name) {
    (void)snprintf(out_path, cap, "%s/%s", dir, file_name);
}

/**
 * @brief 单数字推理。
 */
static int infer_digit(WorkflowRuntime* runtime, int digit, int out_seg[7]) {
    char command[32];
    float state[STATE_DIM];
    float act[OUTPUT_DIM];
    int i = 0;
    int rc = 0;
    (void)snprintf(command, sizeof(command), "%d", digit);
    memset(state, 0, sizeof(state));
    memset(act, 0, sizeof(act));
    state[0] = (float)digit / 9.0f;
    rc = workflow_run_step(runtime, command, state, act);
    if (rc != WORKFLOW_STATUS_OK) {
        return -1;
    }
    for (i = 0; i < 7; ++i) {
        out_seg[i] = (act[i] >= 0.5f) ? 1 : 0;
    }
    return 0;
}

/**
 * @brief 基于 C 数组权重的交互式推理入口。
 */
int main(int argc, char** argv) {
    const char* data_dir = "demo/sevenseg/data";
    char vocab_path[260];
    WorkflowRuntime runtime;
    char line[64];
    memset(&runtime, 0, sizeof(runtime));
    if (argc >= 2) {
        data_dir = argv[1];
    }
    build_path(vocab_path, sizeof(vocab_path), data_dir, "demo_vocab_sevenseg.txt");
    if (runtime_init_from_array(&runtime, vocab_path) != 0) {
        return 1;
    }
    printf("SevenSeg Infer(C Array): input one digit (0-9), 'q' to quit.\n");
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
