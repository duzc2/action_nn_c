/**
 * @file infer_main.c
 * @brief Weather prediction demo — inference entry.
 *
 * Loads per-city trained weight files, evaluates on test data, and reports
 * MAE / RMSE metrics along with sample predictions for each city.
 */

#include "infer.h"
#include "weights_load.h"
#include "weather_dataset.h"
#include "../demo_runtime_paths.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

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

#define PREVIEW_COUNT 5U

typedef struct {
    const char* name;
    const char* csv_path;
    const char* weights_path;
} CityConfig;

static const CityConfig CITIES[] = {
    { "Beijing",  DATASET_ROOT "/beijing.csv",  "../data/weights_beijing.bin"  },
    { "Shanghai", DATASET_ROOT "/shanghai.csv", "../data/weights_shanghai.bin" },
    { "New York", DATASET_ROOT "/new_york.csv", "../data/weights_new_york.bin" },
};

static const size_t CITY_COUNT = sizeof(CITIES) / sizeof(CITIES[0]);

static const char* FEATURE_NAMES[] = {
    "TempMax", "TempMin", "Precip", "WindSpd"
};

static float compute_mae(const float* pred, const float* actual, size_t count) {
    float sum = 0.0f;
    size_t i;
    for (i = 0U; i < count; ++i) {
        sum += fabsf(pred[i] - actual[i]);
    }
    return sum / (float)count;
}

static float compute_mse(const float* pred, const float* actual, size_t count) {
    float sum = 0.0f;
    size_t i;
    for (i = 0U; i < count; ++i) {
        float diff = pred[i] - actual[i];
        sum += diff * diff;
    }
    return sum / (float)count;
}

static int infer_city(const CityConfig* cfg) {
    WeatherDataset raw_data;
    WeatherWindowDataset train_set;  /* discarded, only test matters */
    WeatherWindowDataset test_set;
    void* infer_ctx;
    char error_buffer[256];
    size_t sample;
    float total_mae = 0.0f;
    float total_mse = 0.0f;

    (void)memset(&raw_data, 0, sizeof(raw_data));
    (void)memset(&train_set, 0, sizeof(train_set));
    (void)memset(&test_set, 0, sizeof(test_set));

    printf("\n=== Inference: %s ===\n\n", cfg->name);

    if (weather_dataset_load(cfg->csv_path, &raw_data,
            error_buffer, sizeof(error_buffer)) != 0) {
        fprintf(stderr, "Failed to load %s: %s\n", cfg->name, error_buffer);
        return -1;
    }

    if (weather_window_build(&raw_data, 0.8f, &train_set, &test_set,
            error_buffer, sizeof(error_buffer)) != 0) {
        fprintf(stderr, "Failed to build windows: %s\n", error_buffer);
        weather_dataset_free(&raw_data);
        return -1;
    }
    weather_window_free(&train_set);  /* only need test */

    if (test_set.sample_count == 0U) {
        fprintf(stderr, "No test samples for %s\n", cfg->name);
        weather_window_free(&test_set);
        weather_dataset_free(&raw_data);
        return -1;
    }

    printf("Test samples: %zu\n", test_set.sample_count);

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        weather_window_free(&test_set);
        weather_dataset_free(&raw_data);
        return -1;
    }

    if (weights_load_from_file(infer_ctx, cfg->weights_path) != 0) {
        fprintf(stderr, "Failed to load weights from %s\n", cfg->weights_path);
        infer_destroy(infer_ctx);
        weather_window_free(&test_set);
        weather_dataset_free(&raw_data);
        return -1;
    }

    for (sample = 0U; sample < test_set.sample_count; ++sample) {
        float output[WEATHER_OUTPUT_SIZE];
        float denorm_pred[WEATHER_OUTPUT_SIZE];
        float denorm_actual[WEATHER_OUTPUT_SIZE];
        size_t feat_idx;

        const float* input  = &test_set.inputs[sample * WEATHER_INPUT_SIZE];
        const float* target = &test_set.targets[sample * WEATHER_OUTPUT_SIZE];

        if (infer_auto_run(infer_ctx, input, output) != 0) {
            fprintf(stderr, "Inference failed at sample %zu\n", sample);
            continue;
        }

        weather_denormalize_output(output, test_set.feature_min,
            test_set.feature_max, denorm_pred);
        weather_denormalize_output(target, test_set.feature_min,
            test_set.feature_max, denorm_actual);

        total_mae += compute_mae(denorm_pred, denorm_actual, WEATHER_OUTPUT_SIZE);
        total_mse += compute_mse(denorm_pred, denorm_actual, WEATHER_OUTPUT_SIZE);

        if (sample < PREVIEW_COUNT) {
            printf("Sample %zu:\n", sample);
            for (feat_idx = 0U; feat_idx < WEATHER_NUM_FEATURES; ++feat_idx) {
                printf("  %s: pred=%.2f actual=%.2f\n",
                    FEATURE_NAMES[feat_idx],
                    (double)denorm_pred[feat_idx],
                    (double)denorm_actual[feat_idx]);
            }
        }
    }

    printf("\nMetrics on %zu test samples:\n", test_set.sample_count);
    printf("  MAE:  %.4f\n",
        (double)(total_mae / (float)test_set.sample_count));
    printf("  RMSE: %.4f\n",
        (double)(sqrtf(total_mse / (float)test_set.sample_count)));

    infer_destroy(infer_ctx);
    weather_window_free(&test_set);
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

    printf("Weather Prediction - Inference\n");
    printf("===============================\n");

    any_failed = 0;
    for (city_idx = 0U; city_idx < CITY_COUNT; ++city_idx) {
        if (infer_city(&CITIES[city_idx]) != 0) {
            any_failed = 1;
        }
    }

    if (any_failed) {
        fprintf(stderr, "\n=== One or more cities failed inference ===\n");
        return 1;
    }

    printf("\n=== All cities evaluated ===\n");
    return 0;
}
