#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/csv_loader.h"

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
        int n = 0;
        memset(&item, 0, sizeof(item));
        n = sscanf(line,
                   "%127[^,],%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                   item.command,
                   &item.state[0], &item.state[1], &item.state[2], &item.state[3],
                   &item.state[4], &item.state[5], &item.state[6], &item.state[7],
                   &item.target[0], &item.target[1], &item.target[2], &item.target[3]);
        if (n < 13) {
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

void csv_free_dataset(CsvDataset* dataset) {
    if (dataset == NULL) {
        return;
    }
    free(dataset->samples);
    dataset->samples = NULL;
    dataset->count = 0U;
}
