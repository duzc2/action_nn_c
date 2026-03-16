#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/infer/include/workflow_infer.h"
#include "../../src/include/tokenizer.h"
#include "sevenseg_shared.h"

size_t g_demo_network_sevenseg_vocab_size(void);
size_t g_demo_network_sevenseg_state_dim(void);
size_t g_demo_network_sevenseg_output_dim(void);
size_t g_demo_network_sevenseg_token_slots(void);
void g_demo_network_sevenseg_forward(const float* token_onehot, size_t token_count, const float* state, float* out);

/**
 * @brief 拼接目录与文件名，构造数据文件路径。
 */
static void build_path(char* out_path, size_t cap, const char* dir, const char* file_name) {
    (void)snprintf(out_path, cap, "%s/%s", dir, file_name);
}

/**
 * @brief 使用导出的 C 前向函数执行单数字推理。
 *
 * 关键算法：
 * - 先将 token id 展开为 one-hot，再调用导出的网络函数完成前向计算。
 */
static int infer_digit_with_cfunc(Tokenizer* tokenizer, int digit, int out_seg[7]) {
    char command[32];
    int ids[MAX_SEQ_LEN];
    size_t count = 0U;
    float token_onehot[MAX_SEQ_LEN * VOCAB_SIZE];
    float state[STATE_DIM];
    float act[OUTPUT_DIM];
    size_t t = 0U;
    int i = 0;
    int rc = 0;
    if (tokenizer == NULL || out_seg == NULL) {
        return -1;
    }
    memset(ids, 0, sizeof(ids));
    memset(token_onehot, 0, sizeof(token_onehot));
    memset(state, 0, sizeof(state));
    memset(act, 0, sizeof(act));
    (void)snprintf(command, sizeof(command), "%d", digit);
    rc = tokenizer_encode(tokenizer, command, ids, MAX_SEQ_LEN, &count);
    if (rc != TOKENIZER_STATUS_OK || count == 0U) {
        return -2;
    }
    for (t = 0U; t < count && t < MAX_SEQ_LEN; ++t) {
        size_t id = (ids[t] >= 0 && (size_t)ids[t] < VOCAB_SIZE) ? (size_t)ids[t] : 0U;
        token_onehot[t * VOCAB_SIZE + id] = 1.0f;
    }
    state[0] = (float)digit / 9.0f;
    g_demo_network_sevenseg_forward(token_onehot, count, state, act);
    for (i = 0; i < 7; ++i) {
        out_seg[i] = (act[i] >= 0.5f) ? 1 : 0;
    }
    return 0;
}

/**
 * @brief 基于 C 函数前向网络的交互式推理入口。
 */
int main(int argc, char** argv) {
    const char* data_dir = "demo/sevenseg/data";
    char vocab_path[260];
    Vocabulary vocab;
    Tokenizer tokenizer;
    char line[64];
    int rc = 0;
    memset(&vocab, 0, sizeof(vocab));
    memset(&tokenizer, 0, sizeof(tokenizer));
    if (argc >= 2) {
        data_dir = argv[1];
    }
    build_path(vocab_path, sizeof(vocab_path), data_dir, "demo_vocab_sevenseg.txt");
    rc = workflow_prepare_tokenizer(vocab_path, &vocab, &tokenizer);
    if (rc != WORKFLOW_STATUS_OK) {
        return 1;
    }
    if (g_demo_network_sevenseg_vocab_size() != VOCAB_SIZE ||
        g_demo_network_sevenseg_state_dim() != STATE_DIM ||
        g_demo_network_sevenseg_output_dim() != OUTPUT_DIM ||
        g_demo_network_sevenseg_token_slots() != MAX_SEQ_LEN) {
        vocab_free(&vocab);
        return 2;
    }
    printf("SevenSeg Infer(C Function): input one digit (0-9), 'q' to quit.\n");
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
        if (infer_digit_with_cfunc(&tokenizer, d, seg) != 0) {
            break;
        }
        sevenseg_render_cli(d, seg);
    }
    vocab_free(&vocab);
    return 0;
}
