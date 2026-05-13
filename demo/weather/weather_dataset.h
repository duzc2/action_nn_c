/**
 * @file weather_dataset.h
 * @brief Weather time-series dataset loader with sliding window support.
 *
 * Loads CSV weather records for a city and builds normalized sliding-window
 * datasets suitable for next-day prediction with a small MLP model.
 */

#ifndef DEMO_WEATHER_DATASET_H
#define DEMO_WEATHER_DATASET_H

#include <stddef.h>

#define WEATHER_WINDOW_SIZE      7U
#define WEATHER_NUM_FEATURES     4U
#define WEATHER_INPUT_SIZE       (WEATHER_WINDOW_SIZE * WEATHER_NUM_FEATURES)  /* 28 */
#define WEATHER_OUTPUT_SIZE      WEATHER_NUM_FEATURES                         /* 4  */

/* Raw daily weather record (one row from CSV). */
typedef struct {
    float temp_max;
    float temp_min;
    float precipitation;
    float wind_speed;
} WeatherRecord;

/* Full time-series dataset loaded from CSV. */
typedef struct {
    WeatherRecord* records;
    size_t          record_count;
    /* Per-feature min/max computed during loading for normalization. */
    float feature_min[WEATHER_NUM_FEATURES];
    float feature_max[WEATHER_NUM_FEATURES];
} WeatherDataset;

/* Pre-built sliding window dataset with normalized I/O. */
typedef struct {
    size_t   sample_count;
    float*   inputs;   /* [sample_count * WEATHER_INPUT_SIZE] */
    float*   targets;  /* [sample_count * WEATHER_OUTPUT_SIZE] */
    /* Normalization params copied from the source WeatherDataset. */
    float    feature_min[WEATHER_NUM_FEATURES];
    float    feature_max[WEATHER_NUM_FEATURES];
} WeatherWindowDataset;

/**
 * @brief Load CSV weather data and compute per-feature min/max.
 *
 * @param csv_path         Path to the CSV file.
 * @param out_dataset      Output dataset (caller must zero-initialize).
 * @param error_buffer     Buffer for error messages.
 * @param error_buffer_size Size of error_buffer.
 * @return 0 on success, non-zero on failure.
 */
int weather_dataset_load(const char* csv_path,
                         WeatherDataset* out_dataset,
                         char* error_buffer,
                         size_t error_buffer_size);

/** @brief Free resources owned by a WeatherDataset. */
void weather_dataset_free(WeatherDataset* dataset);

/**
 * @brief Build sliding windows from loaded records with min-max normalization.
 *
 * The first (1.0 - train_ratio) fraction of data is reserved for testing;
 * the rest is used for training. No shuffling is performed to preserve
 * temporal order and avoid look-ahead bias.
 *
 * @param source           Loaded raw weather dataset.
 * @param train_ratio      Fraction of windows to use for training (e.g. 0.8f).
 * @param out_train        Output training window dataset (caller zero-inits).
 * @param out_test         Output test window dataset (caller zero-inits).
 * @param error_buffer     Buffer for error messages.
 * @param error_buffer_size Size of error_buffer.
 * @return 0 on success, non-zero on failure.
 */
int weather_window_build(const WeatherDataset* source,
                         float train_ratio,
                         WeatherWindowDataset* out_train,
                         WeatherWindowDataset* out_test,
                         char* error_buffer,
                         size_t error_buffer_size);

/** @brief Free resources owned by a WeatherWindowDataset. */
void weather_window_free(WeatherWindowDataset* dataset);

/**
 * @brief Apply stored min-max normalization to a raw input vector.
 *
 * @param raw_input   Raw feature vector (28 floats, 7 days x 4 features).
 * @param feature_min Per-feature minimum values.
 * @param feature_max Per-feature maximum values.
 * @param normalized_input  Output normalized vector (28 floats).
 */
void weather_normalize_input(const float* raw_input,
                             const float* feature_min,
                             const float* feature_max,
                             float* normalized_input);

/**
 * @brief Denormalize model output back to original scale.
 *
 * @param normalized_output  Model output in [0, 1] range (4 floats).
 * @param feature_min        Per-feature minimum values.
 * @param feature_max        Per-feature maximum values.
 * @param denormalized_output  Output in original scale (4 floats).
 */
void weather_denormalize_output(const float* normalized_output,
                                const float* feature_min,
                                const float* feature_max,
                                float* denormalized_output);

#endif /* DEMO_WEATHER_DATASET_H */
