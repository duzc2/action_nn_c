#ifndef CSV_LOADER_H
#define CSV_LOADER_H

#include <stddef.h>

typedef struct CsvSample {
    char command[128];
    float state[8];
    float target[4];
} CsvSample;

typedef struct CsvDataset {
    CsvSample* samples;
    size_t count;
} CsvDataset;

int csv_load_dataset(const char* file_path, CsvDataset* out_dataset);
void csv_free_dataset(CsvDataset* dataset);

#endif
