/**
 * @file weather_dataset.c
 * @brief Weather CSV loader and sliding-window dataset builder.
 */

#include "weather_dataset.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── CSV record parser ─────────────────────────────────────────────────── */

/* Maximum line length (date + 4 floats + commas + newline). */
#define WEATHER_LINE_BUF_SIZE 256U

/**
 * @brief Parse one CSV line (skip date, read 4 floats).
 *
 * @return Number of floats successfully parsed (should be 4).
 */
static int parse_csv_line(const char* line, WeatherRecord* record) {
    const char* p = line;
    float vals[4];
    int count;

    /* Skip the date column (first field). */
    while (*p != '\0' && *p != ',') {
        ++p;
    }
    if (*p == ',') {
        ++p;
    }

    /* Parse 4 comma-separated float fields. */
    for (count = 0; count < 4; ++count) {
        char* end;
        vals[count] = strtof(p, &end);
        if (end == p) {
            /* Parse failed — treat as 0.0f (should not happen with our data). */
            vals[count] = 0.0f;
        }
        p = end;
        while (*p == ',' || *p == ' ') {
            ++p;
        }
        if (*p == '\0' || *p == '\n' || *p == '\r') {
            ++count;  /* Count this last field before breaking. */
            break;
        }
    }

    if (count < 4) {
        return count;
    }

    record->temp_max      = vals[0];
    record->temp_min      = vals[1];
    record->precipitation = vals[2];
    record->wind_speed    = vals[3];
    return 4;
}

/* ── File I/O (MSVC-compatible) ─────────────────────────────────────────── */

static int weather_fopen(FILE** out_file, const char* path, const char* mode) {
#ifdef _WIN32
    return fopen_s(out_file, path, mode);
#else
    *out_file = fopen(path, mode);
    return (*out_file == NULL) ? -1 : 0;
#endif
}

/* ── Public: load CSV ───────────────────────────────────────────────────── */

int weather_dataset_load(const char* csv_path,
                         WeatherDataset* out_dataset,
                         char* error_buffer,
                         size_t error_buffer_size) {
    FILE* file;
    char line[WEATHER_LINE_BUF_SIZE];
    size_t capacity;
    size_t line_num;
    size_t i;
    int fields_parsed;

    if (csv_path == NULL || out_dataset == NULL) {
        if (error_buffer != NULL && error_buffer_size > 0U) {
            snprintf(error_buffer, error_buffer_size, "NULL pointer argument");
        }
        return -1;
    }

    if (weather_fopen(&file, csv_path, "rb") != 0) {
        if (error_buffer != NULL && error_buffer_size > 0U) {
            snprintf(error_buffer, error_buffer_size,
                     "Cannot open %s", csv_path);
        }
        return -1;
    }

    /* First pass: count records and allocate. */
    capacity = 0U;
    out_dataset->record_count = 0U;

    /* Skip header. */
    if (fgets(line, (int)sizeof(line), file) == NULL) {
        fclose(file);
        if (error_buffer != NULL && error_buffer_size > 0U) {
            snprintf(error_buffer, error_buffer_size,
                     "Empty or missing CSV header in %s", csv_path);
        }
        return -1;
    }

    /* Count lines (skip empty lines at end). */
    while (fgets(line, (int)sizeof(line), file) != NULL) {
        if (line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        ++out_dataset->record_count;
    }

    if (out_dataset->record_count == 0U) {
        fclose(file);
        if (error_buffer != NULL && error_buffer_size > 0U) {
            snprintf(error_buffer, error_buffer_size,
                     "No data records in %s", csv_path);
        }
        return -1;
    }

    capacity = out_dataset->record_count;
    out_dataset->records = (WeatherRecord*)calloc(capacity, sizeof(WeatherRecord));
    if (out_dataset->records == NULL) {
        fclose(file);
        if (error_buffer != NULL && error_buffer_size > 0U) {
            snprintf(error_buffer, error_buffer_size,
                     "Failed to allocate %zu records", capacity);
        }
        return -1;
    }

    /* Second pass: rewind and parse records. */
    rewind(file);
    /* Skip header again. */
    (void)fgets(line, (int)sizeof(line), file);

    /* Initialize per-feature min/max. */
    for (i = 0U; i < WEATHER_NUM_FEATURES; ++i) {
        out_dataset->feature_min[i] = 1e30f;
        out_dataset->feature_max[i] = -1e30f;
    }

    for (line_num = 0U; line_num < capacity; ++line_num) {
        WeatherRecord rec;
        if (fgets(line, (int)sizeof(line), file) == NULL) {
            break;
        }
        if (line[0] == '\n' || line[0] == '\r') {
            /* Should not happen, but skip just in case. */
            continue;
        }

        fields_parsed = parse_csv_line(line, &rec);
        if (fields_parsed < 4) {
            fclose(file);
            if (error_buffer != NULL && error_buffer_size > 0U) {
                snprintf(error_buffer, error_buffer_size,
                         "Failed to parse line %zu in %s", line_num + 2U, csv_path);
            }
            return -1;
        }

        out_dataset->records[line_num] = rec;

        /* Update per-feature min/max. */
        {
            float feat[4];
            feat[0] = rec.temp_max;
            feat[1] = rec.temp_min;
            feat[2] = rec.precipitation;
            feat[3] = rec.wind_speed;
            for (i = 0U; i < WEATHER_NUM_FEATURES; ++i) {
                if (feat[i] < out_dataset->feature_min[i]) {
                    out_dataset->feature_min[i] = feat[i];
                }
                if (feat[i] > out_dataset->feature_max[i]) {
                    out_dataset->feature_max[i] = feat[i];
                }
            }
        }
    }

    fclose(file);
    return 0;
}

/* ── Public: free dataset ──────────────────────────────────────────────── */

void weather_dataset_free(WeatherDataset* dataset) {
    if (dataset == NULL) {
        return;
    }
    free(dataset->records);
    dataset->records = NULL;
    dataset->record_count = 0U;
}

/* ── Normalization helpers ──────────────────────────────────────────────── */

static float weather_normalize_value(float val, float vmin, float vmax) {
    float range = vmax - vmin;
    if (range < 1e-8f) {
        return 0.5f;  /* All values are the same — map to midpoint. */
    }
    return (val - vmin) / range;
}

void weather_normalize_input(const float* raw_input,
                             const float* feature_min,
                             const float* feature_max,
                             float* normalized_input) {
    size_t day;
    for (day = 0U; day < WEATHER_WINDOW_SIZE; ++day) {
        size_t base = day * WEATHER_NUM_FEATURES;
        size_t feat;
        for (feat = 0U; feat < WEATHER_NUM_FEATURES; ++feat) {
            normalized_input[base + feat] = weather_normalize_value(
                raw_input[base + feat],
                feature_min[feat],
                feature_max[feat]);
        }
    }
}

static float weather_denormalize_value(float val, float vmin, float vmax) {
    return val * (vmax - vmin) + vmin;
}

void weather_denormalize_output(const float* normalized_output,
                                const float* feature_min,
                                const float* feature_max,
                                float* denormalized_output) {
    size_t feat;
    for (feat = 0U; feat < WEATHER_NUM_FEATURES; ++feat) {
        denormalized_output[feat] = weather_denormalize_value(
            normalized_output[feat],
            feature_min[feat],
            feature_max[feat]);
    }
}

/* ── Public: build sliding windows ─────────────────────────────────────── */

int weather_window_build(const WeatherDataset* source,
                         float train_ratio,
                         WeatherWindowDataset* out_train,
                         WeatherWindowDataset* out_test,
                         char* error_buffer,
                         size_t error_buffer_size) {
    size_t total_samples;
    size_t train_count;
    size_t test_count;
    size_t sample_idx;
    size_t feat_idx;
    size_t day_idx;

    if (source == NULL || out_train == NULL || out_test == NULL) {
        if (error_buffer != NULL && error_buffer_size > 0U) {
            snprintf(error_buffer, error_buffer_size, "NULL pointer argument");
        }
        return -1;
    }

    if (source->record_count <= WEATHER_WINDOW_SIZE) {
        if (error_buffer != NULL && error_buffer_size > 0U) {
            snprintf(error_buffer, error_buffer_size,
                     "Not enough records (%zu) for window size %u",
                     source->record_count, (unsigned)WEATHER_WINDOW_SIZE);
        }
        return -1;
    }

    /* Each sample: (records[i..i+6]) -> records[i+7]. */
    total_samples = source->record_count - WEATHER_WINDOW_SIZE;
    train_count = (size_t)((float)total_samples * train_ratio);
    if (train_count == 0U) {
        train_count = 1U;
    }
    test_count = total_samples - train_count;

    out_train->sample_count = train_count;
    out_train->inputs  = (float*)calloc(train_count * WEATHER_INPUT_SIZE,  sizeof(float));
    out_train->targets = (float*)calloc(train_count * WEATHER_OUTPUT_SIZE, sizeof(float));

    out_test->sample_count = test_count;
    out_test->inputs  = (float*)calloc(test_count * WEATHER_INPUT_SIZE,  sizeof(float));
    out_test->targets = (float*)calloc(test_count * WEATHER_OUTPUT_SIZE, sizeof(float));

    if (out_train->inputs == NULL || out_train->targets == NULL ||
        out_test->inputs  == NULL || out_test->targets  == NULL) {
        weather_window_free(out_train);
        weather_window_free(out_test);
        if (error_buffer != NULL && error_buffer_size > 0U) {
            snprintf(error_buffer, error_buffer_size,
                     "Failed to allocate sliding window memory");
        }
        return -1;
    }

    /* Copy normalization params. */
    for (feat_idx = 0U; feat_idx < WEATHER_NUM_FEATURES; ++feat_idx) {
        out_train->feature_min[feat_idx] = source->feature_min[feat_idx];
        out_train->feature_max[feat_idx] = source->feature_max[feat_idx];
        out_test->feature_min[feat_idx]  = source->feature_min[feat_idx];
        out_test->feature_max[feat_idx]  = source->feature_max[feat_idx];
    }

    /* Build all samples sequentially: first train_count go to train,
     * remaining go to test. */
    for (sample_idx = 0U; sample_idx < total_samples; ++sample_idx) {
        WeatherWindowDataset* dest;
        size_t dest_offset;

        if (sample_idx < train_count) {
            dest = out_train;
            dest_offset = sample_idx;
        } else {
            dest = out_test;
            dest_offset = sample_idx - train_count;
        }

        /* Input: flatten WEATHER_WINDOW_SIZE consecutive records. */
        for (day_idx = 0U; day_idx < WEATHER_WINDOW_SIZE; ++day_idx) {
            const WeatherRecord* rec = &source->records[sample_idx + day_idx];
            size_t in_base = dest_offset * WEATHER_INPUT_SIZE + day_idx * WEATHER_NUM_FEATURES;

            dest->inputs[in_base + 0] = weather_normalize_value(
                rec->temp_max, source->feature_min[0], source->feature_max[0]);
            dest->inputs[in_base + 1] = weather_normalize_value(
                rec->temp_min, source->feature_min[1], source->feature_max[1]);
            dest->inputs[in_base + 2] = weather_normalize_value(
                rec->precipitation, source->feature_min[2], source->feature_max[2]);
            dest->inputs[in_base + 3] = weather_normalize_value(
                rec->wind_speed, source->feature_min[3], source->feature_max[3]);
        }

        /* Target: the next day's record (normalized). */
        {
            const WeatherRecord* tgt = &source->records[sample_idx + WEATHER_WINDOW_SIZE];
            size_t tgt_base = dest_offset * WEATHER_OUTPUT_SIZE;

            dest->targets[tgt_base + 0] = weather_normalize_value(
                tgt->temp_max, source->feature_min[0], source->feature_max[0]);
            dest->targets[tgt_base + 1] = weather_normalize_value(
                tgt->temp_min, source->feature_min[1], source->feature_max[1]);
            dest->targets[tgt_base + 2] = weather_normalize_value(
                tgt->precipitation, source->feature_min[2], source->feature_max[2]);
            dest->targets[tgt_base + 3] = weather_normalize_value(
                tgt->wind_speed, source->feature_min[3], source->feature_max[3]);
        }
    }

    return 0;
}

/* ── Public: free window dataset ───────────────────────────────────────── */

void weather_window_free(WeatherWindowDataset* dataset) {
    if (dataset == NULL) {
        return;
    }
    free(dataset->inputs);
    free(dataset->targets);
    dataset->inputs  = NULL;
    dataset->targets = NULL;
    dataset->sample_count = 0U;
}
