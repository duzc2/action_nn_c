/**
 * @file train_main.c
 * @brief Weather prediction demo — training entry.
 *
 * Trains three identical-architecture MLP models (one per city) using
 * a 7-day sliding window approach. Each city gets its own weight file.
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "weather_dataset.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>

#define WEATHER_EPOCH_COUNT 50U

/*
 * MSVC multi-config generators place the executable inside a Debug/ (or
 * Release/) subdirectory, so we need one more "../" to reach the project
 * root than single-config generators (Ninja, Make).
 */
#ifdef _MSC_VER
#define DATASET_ROOT "../../../../../demo/weather/dataset"
#else
#define DATASET_ROOT "../../../../demo/weather/dataset"
#endif

typedef struct {
    const char* name;
    const char* csv_path;
    const char* weights_output;
} CityConfig;

static const CityConfig CITIES[] = {
    { "Beijing",  DATASET_ROOT "/beijing.csv",  "../data/weights_beijing.bin"  },
    { "Shanghai", DATASET_ROOT "/shanghai.csv", "../data/weights_shanghai.bin" },
    { "New York", DATASET_ROOT "/new_york.csv", "../data/weights_new_york.bin" },
};

static const size_t CITY_COUNT = sizeof(CITIES) / sizeof(CITIES[0]);

static int train_city(const CityConfig* cfg) {
    WeatherDataset raw_data;
    WeatherWindowDataset train_set;
    WeatherWindowDataset test_set;
    void* infer_ctx;
    void* train_ctx;
    char error_buffer[256];
    size_t epoch;
    size_t sample;
    float epoch_loss_sum;
    float avg_loss;

    (void)memset(&raw_data, 0, sizeof(raw_data));
    (void)memset(&train_set, 0, sizeof(train_set));
    (void)memset(&test_set, 0, sizeof(test_set));

    printf("\n=== Training: %s ===\n\n", cfg->name);

    if (weather_dataset_load(cfg->csv_path, &raw_data,
            error_buffer, sizeof(error_buffer)) != 0) {
        fprintf(stderr, "Failed to load %s: %s\n", cfg->name, error_buffer);
        return -1;
    }
    printf("Loaded %zu weather records\n", raw_data.record_count);

    if (weather_window_build(&raw_data, 0.8f, &train_set, &test_set,
            error_buffer, sizeof(error_buffer)) != 0) {
        fprintf(stderr, "Failed to build windows for %s: %s\n",
            cfg->name, error_buffer);
        weather_dataset_free(&raw_data);
        return -1;
    }
    printf("Train samples: %zu, Test samples: %zu\n",
        train_set.sample_count, test_set.sample_count);

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        weather_window_free(&test_set);
        weather_window_free(&train_set);
        weather_dataset_free(&raw_data);
        return -1;
    }

    train_ctx = train_create(infer_ctx);
    if (train_ctx == NULL) {
        fprintf(stderr, "Failed to create training context\n");
        infer_destroy(infer_ctx);
        weather_window_free(&test_set);
        weather_window_free(&train_set);
        weather_dataset_free(&raw_data);
        return -1;
    }

    for (epoch = 0U; epoch < WEATHER_EPOCH_COUNT; ++epoch) {
        epoch_loss_sum = 0.0f;
        for (sample = 0U; sample < train_set.sample_count; ++sample) {
            const float* input  = &train_set.inputs[
                sample * WEATHER_INPUT_SIZE];
            const float* target = &train_set.targets[
                sample * WEATHER_OUTPUT_SIZE];

            if (train_step(train_ctx, input, target) != 0) {
                fprintf(stderr, "Training step failed at epoch %zu sample %zu\n",
                    epoch + 1U, sample + 1U);
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                weather_window_free(&test_set);
                weather_window_free(&train_set);
                weather_dataset_free(&raw_data);
                return -1;
            }
            epoch_loss_sum += train_get_loss(train_ctx);
        }
        avg_loss = epoch_loss_sum / (float)train_set.sample_count;
        printf("  Epoch %zu/%u - avg loss: %.6f\n",
            epoch + 1U, (unsigned)WEATHER_EPOCH_COUNT, avg_loss);
    }

    if (weights_save_to_file(infer_ctx, cfg->weights_output) != 0) {
        fprintf(stderr, "Failed to save weights to %s\n", cfg->weights_output);
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        weather_window_free(&test_set);
        weather_window_free(&train_set);
        weather_dataset_free(&raw_data);
        return -1;
    }
    printf("\nWeights saved: %s\n", cfg->weights_output);

    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    weather_window_free(&test_set);
    weather_window_free(&train_set);
    weather_dataset_free(&raw_data);
    return 0;
}

int main(void) {
    size_t city_idx;
    int any_failed;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("Weather Prediction - Training\n");
    printf("=============================\n");
    printf("Architecture: MLP %u -> [%u] -> %u\n",
        (unsigned)WEATHER_INPUT_SIZE, 8U, (unsigned)WEATHER_OUTPUT_SIZE);
    printf("Epochs per city: %u\n\n", (unsigned)WEATHER_EPOCH_COUNT);

    any_failed = 0;
    for (city_idx = 0U; city_idx < CITY_COUNT; ++city_idx) {
        if (train_city(&CITIES[city_idx]) != 0) {
            any_failed = 1;
        }
    }

    if (any_failed) {
        fprintf(stderr, "\n=== One or more cities failed training ===\n");
        return 1;
    }

    printf("\n=== All cities trained ===\n");
    return 0;
}
