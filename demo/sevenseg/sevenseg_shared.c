#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "sevenseg_shared.h"

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

const int g_sevenseg_truth[10][7] = {
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

int sevenseg_ensure_dir(const char* path) {
    int rc = 0;
    if (path == NULL) {
        return -1;
    }
#if defined(_WIN32)
    rc = _mkdir(path);
#else
    rc = mkdir(path, 0755);
#endif
    if (rc == 0 || errno == EEXIST) {
        return 0;
    }
    return -1;
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

int sevenseg_write_vocab(const char* file_path) {
    static const char* tokens[] = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    };
    return write_rows_to_text(file_path, tokens, sizeof(tokens) / sizeof(tokens[0]));
}

static int segbit(int digit, int idx) {
    return (g_sevenseg_truth[digit][idx] != 0) ? 1 : 0;
}

static float map_target(int pin_on) {
    return pin_on ? 1.0f : 0.0f;
}

int sevenseg_build_samples(WorkflowTrainSample* out_samples,
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

int sevenseg_verify_samples(const WorkflowTrainSample* samples, size_t sample_count) {
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

void sevenseg_render_cli(int digit, const int seg[7]) {
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
