#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/csv_loader.h"

static int parse_float_token(const char* token, float* out_value) {
    char* end = NULL;
    double value = 0.0;
    if (token == NULL || out_value == NULL) {
        return -1;
    }
    value = strtod(token, &end);
    if (end == token) {
        return -1;
    }
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') {
        ++end;
    }
    if (*end != '\0') {
        return -1;
    }
    *out_value = (float)value;
    return 0;
}

static int parse_csv_line(const char* line, CsvSample* out_item) {
    char buffer[512];
    char* token = NULL;
    int i = 0;
    if (line == NULL || out_item == NULL) {
        return -1;
    }
    memset(out_item, 0, sizeof(*out_item));
    (void)snprintf(buffer, sizeof(buffer), "%s", line);
    token = strtok(buffer, ",");
    if (token == NULL) {
        return -1;
    }
    (void)snprintf(out_item->command, sizeof(out_item->command), "%s", token);
    for (i = 0; i < STATE_DIM; ++i) {
        token = strtok(NULL, ",");
        if (token == NULL || parse_float_token(token, &out_item->state[i]) != 0) {
            return -1;
        }
    }
    for (i = 0; i < OUTPUT_DIM; ++i) {
        token = strtok(NULL, ",");
        if (token == NULL || parse_float_token(token, &out_item->target[i]) != 0) {
            return -1;
        }
    }
    return 0;
}

/**
 * @brief 读取 CSV 文件并构建内存数据集。
 *
 * 关键保护点：
 * - 动态扩容失败时立即释放已分配内存并返回错误，避免泄漏。
 * - 字段数不足的行会被跳过，保证输出样本结构完整。
 */
int csv_load_dataset(const char* file_path, CsvDataset* out_dataset) {
    FILE* fp = NULL;
    CsvSample* samples = NULL;
    size_t count = 0U;
    size_t cap = 0U;
    char line[512];
    if (file_path == NULL || out_dataset == NULL) {
        return -1;
    }
    out_dataset->samples = NULL;
    out_dataset->count = 0U;
    fp = fopen(file_path, "r");
    if (fp == NULL) {
        return -2;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        CsvSample item;
        if (parse_csv_line(line, &item) != 0) {
            continue;
        }
        if (count == cap) {
            size_t new_cap = (cap == 0U) ? 16U : cap * 2U;
            CsvSample* tmp = (CsvSample*)realloc(samples, sizeof(CsvSample) * new_cap);
            if (tmp == NULL) {
                free(samples);
                fclose(fp);
                return -3;
            }
            samples = tmp;
            cap = new_cap;
        }
        samples[count++] = item;
    }
    fclose(fp);
    out_dataset->samples = samples;
    out_dataset->count = count;
    return 0;
}

/**
 * @brief 释放 csv_load_dataset 分配的数据集资源。
 */
void csv_free_dataset(CsvDataset* dataset) {
    if (dataset == NULL) {
        return;
    }
    free(dataset->samples);
    dataset->samples = NULL;
    dataset->count = 0U;
}
