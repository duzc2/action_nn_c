#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../src/include/config_user.h"
#include "../../src/infer/include/workflow_infer.h"
#include "../../src/include/tokenizer.h"

enum {
    BENCH_WARMUP_ITERS = 2000,
    BENCH_ITERS = 20000
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

typedef struct BenchStat {
    const char* mode_name;
    double total_ms;
    double us_per_call;
    double calls_per_sec;
    int mismatches;
    unsigned long long calls;
} BenchStat;

size_t g_demo_weights_sevenseg_count(void);
int g_demo_weights_sevenseg_copy(float* out, size_t out_capacity);
size_t g_demo_network_sevenseg_vocab_size(void);
size_t g_demo_network_sevenseg_state_dim(void);
size_t g_demo_network_sevenseg_output_dim(void);
size_t g_demo_network_sevenseg_token_slots(void);
void g_demo_network_sevenseg_forward(const float* token_onehot, size_t token_count, const float* state, float* out);

static void build_path(char* out_path, size_t cap, const char* dir, const char* file_name) {
    (void)snprintf(out_path, cap, "%s/%s", dir, file_name);
}

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

static int infer_digit_bin_style(WorkflowRuntime* runtime, int digit, int out_seg[7]) {
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

static int infer_digit_cfunc_style(Tokenizer* tokenizer, int digit, int out_seg[7]) {
    char command[32];
    int ids[MAX_SEQ_LEN];
    size_t count = 0U;
    float token_onehot[MAX_SEQ_LEN * VOCAB_SIZE];
    float state[STATE_DIM];
    float act[OUTPUT_DIM];
    size_t t = 0U;
    int i = 0;
    int rc = 0;
    memset(ids, 0, sizeof(ids));
    memset(token_onehot, 0, sizeof(token_onehot));
    memset(state, 0, sizeof(state));
    memset(act, 0, sizeof(act));
    (void)snprintf(command, sizeof(command), "%d", digit);
    rc = tokenizer_encode(tokenizer, command, ids, MAX_SEQ_LEN, &count);
    if (rc != TOKENIZER_STATUS_OK || count == 0U) {
        return -1;
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

static double clock_diff_ms(clock_t start_tick, clock_t end_tick) {
    return ((double)(end_tick - start_tick) * 1000.0) / (double)CLOCKS_PER_SEC;
}

static BenchStat bench_runtime_mode(const char* name,
                                    WorkflowRuntime* runtime,
                                    int warmup_iters,
                                    int bench_iters) {
    BenchStat st;
    clock_t t0 = 0;
    clock_t t1 = 0;
    int iter = 0;
    memset(&st, 0, sizeof(st));
    st.mode_name = name;
    for (iter = 0; iter < warmup_iters; ++iter) {
        int d = iter % 10;
        int seg[7];
        (void)infer_digit_bin_style(runtime, d, seg);
    }
    t0 = clock();
    for (iter = 0; iter < bench_iters; ++iter) {
        int d = iter % 10;
        int seg[7];
        int i = 0;
        if (infer_digit_bin_style(runtime, d, seg) != 0) {
            st.mismatches += 7;
        } else {
            for (i = 0; i < 7; ++i) {
                if (seg[i] != kSevenSegTruth[d][i]) {
                    st.mismatches++;
                }
            }
        }
    }
    t1 = clock();
    st.calls = (unsigned long long)bench_iters;
    st.total_ms = clock_diff_ms(t0, t1);
    st.us_per_call = (st.calls > 0U) ? (st.total_ms * 1000.0 / (double)st.calls) : 0.0;
    st.calls_per_sec = (st.total_ms > 0.0) ? ((double)st.calls * 1000.0 / st.total_ms) : 0.0;
    return st;
}

static BenchStat bench_cfunc_mode(const char* name,
                                  Tokenizer* tokenizer,
                                  int warmup_iters,
                                  int bench_iters) {
    BenchStat st;
    clock_t t0 = 0;
    clock_t t1 = 0;
    int iter = 0;
    memset(&st, 0, sizeof(st));
    st.mode_name = name;
    for (iter = 0; iter < warmup_iters; ++iter) {
        int d = iter % 10;
        int seg[7];
        (void)infer_digit_cfunc_style(tokenizer, d, seg);
    }
    t0 = clock();
    for (iter = 0; iter < bench_iters; ++iter) {
        int d = iter % 10;
        int seg[7];
        int i = 0;
        if (infer_digit_cfunc_style(tokenizer, d, seg) != 0) {
            st.mismatches += 7;
        } else {
            for (i = 0; i < 7; ++i) {
                if (seg[i] != kSevenSegTruth[d][i]) {
                    st.mismatches++;
                }
            }
        }
    }
    t1 = clock();
    st.calls = (unsigned long long)bench_iters;
    st.total_ms = clock_diff_ms(t0, t1);
    st.us_per_call = (st.calls > 0U) ? (st.total_ms * 1000.0 / (double)st.calls) : 0.0;
    st.calls_per_sec = (st.total_ms > 0.0) ? ((double)st.calls * 1000.0 / st.total_ms) : 0.0;
    return st;
}

static int write_markdown_report(const char* path, const BenchStat* stats, size_t n, int warmup_iters, int bench_iters) {
    FILE* fp = NULL;
    size_t i = 0U;
    if (path == NULL || stats == NULL || n == 0U) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -2;
    }
    fprintf(fp, "# SevenSeg Inference Benchmark\n\n");
    fprintf(fp, "- warmup_iters: %d\n", warmup_iters);
    fprintf(fp, "- bench_iters: %d\n\n", bench_iters);
    fprintf(fp, "| Mode | Calls | Total ms | us/call | calls/sec | Mismatches |\n");
    fprintf(fp, "| --- | ---: | ---: | ---: | ---: | ---: |\n");
    for (i = 0U; i < n; ++i) {
        fprintf(fp, "| %s | %llu | %.3f | %.3f | %.2f | %d |\n",
                stats[i].mode_name,
                stats[i].calls,
                stats[i].total_ms,
                stats[i].us_per_call,
                stats[i].calls_per_sec,
                stats[i].mismatches);
    }
    fclose(fp);
    return 0;
}

int main(int argc, char** argv) {
    const char* data_dir = "demo/sevenseg/data";
    char vocab_path[260];
    char bin_path[260];
    char report_path[260];
    WorkflowRuntime runtime_bin;
    WorkflowRuntime runtime_c_array;
    Vocabulary vocab;
    Tokenizer tokenizer;
    BenchStat stats[3];
    int rc = 0;
    memset(&runtime_bin, 0, sizeof(runtime_bin));
    memset(&runtime_c_array, 0, sizeof(runtime_c_array));
    memset(&vocab, 0, sizeof(vocab));
    memset(&tokenizer, 0, sizeof(tokenizer));
    if (argc >= 2) {
        data_dir = argv[1];
    }
    build_path(vocab_path, sizeof(vocab_path), data_dir, "demo_vocab_sevenseg.txt");
    build_path(bin_path, sizeof(bin_path), data_dir, "demo_weights_sevenseg.bin");
    build_path(report_path, sizeof(report_path), data_dir, "benchmark_report.md");
    rc = workflow_runtime_init(&runtime_bin, vocab_path, bin_path);
    if (rc != WORKFLOW_STATUS_OK) {
        return 1;
    }
    rc = runtime_init_from_array(&runtime_c_array, vocab_path);
    if (rc != 0) {
        workflow_runtime_shutdown(&runtime_bin);
        return 2;
    }
    rc = workflow_prepare_tokenizer(vocab_path, &vocab, &tokenizer);
    if (rc != WORKFLOW_STATUS_OK) {
        workflow_runtime_shutdown(&runtime_bin);
        workflow_runtime_shutdown(&runtime_c_array);
        return 3;
    }
    if (g_demo_network_sevenseg_vocab_size() != VOCAB_SIZE ||
        g_demo_network_sevenseg_state_dim() != STATE_DIM ||
        g_demo_network_sevenseg_output_dim() != OUTPUT_DIM ||
        g_demo_network_sevenseg_token_slots() != MAX_SEQ_LEN) {
        vocab_free(&vocab);
        workflow_runtime_shutdown(&runtime_bin);
        workflow_runtime_shutdown(&runtime_c_array);
        return 4;
    }
    stats[0] = bench_runtime_mode("bin_runtime", &runtime_bin, BENCH_WARMUP_ITERS, BENCH_ITERS);
    stats[1] = bench_runtime_mode("c_array_runtime", &runtime_c_array, BENCH_WARMUP_ITERS, BENCH_ITERS);
    stats[2] = bench_cfunc_mode("c_function_forward", &tokenizer, BENCH_WARMUP_ITERS, BENCH_ITERS);
    rc = write_markdown_report(report_path, stats, 3U, BENCH_WARMUP_ITERS, BENCH_ITERS);
    if (rc != 0) {
        vocab_free(&vocab);
        workflow_runtime_shutdown(&runtime_bin);
        workflow_runtime_shutdown(&runtime_c_array);
        return 5;
    }
    printf("benchmark done, report: %s\n", report_path);
    printf("| Mode | Calls | Total ms | us/call | calls/sec | Mismatches |\n");
    printf("| --- | ---: | ---: | ---: | ---: | ---: |\n");
    printf("| %s | %llu | %.3f | %.3f | %.2f | %d |\n",
           stats[0].mode_name, stats[0].calls, stats[0].total_ms, stats[0].us_per_call, stats[0].calls_per_sec, stats[0].mismatches);
    printf("| %s | %llu | %.3f | %.3f | %.2f | %d |\n",
           stats[1].mode_name, stats[1].calls, stats[1].total_ms, stats[1].us_per_call, stats[1].calls_per_sec, stats[1].mismatches);
    printf("| %s | %llu | %.3f | %.3f | %.2f | %d |\n",
           stats[2].mode_name, stats[2].calls, stats[2].total_ms, stats[2].us_per_call, stats[2].calls_per_sec, stats[2].mismatches);
    vocab_free(&vocab);
    workflow_runtime_shutdown(&runtime_bin);
    workflow_runtime_shutdown(&runtime_c_array);
    return 0;
}
