/**
 * @file infer_main.c
 * @brief MNIST demo inference entry.
 */

#include "infer.h"
#include "weights_load.h"
#include "mnist_dataset.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>

#define MNIST_TEST_SAMPLE_LIMIT 200U
#define MNIST_PREVIEW_COUNT 5U

#define MNIST_DATASET_ROOT "../../../../demo/mnist/dataset"
#define MNIST_TEST_IMAGES MNIST_DATASET_ROOT "/t10k-images.idx3-ubyte"
#define MNIST_TEST_LABELS MNIST_DATASET_ROOT "/t10k-labels.idx1-ubyte"

/**
 * @brief Evaluate one dataset and print the first few predictions.
 */
static float preview_predictions(void* infer_ctx, const MnistDataset* dataset) {
    size_t sample_index;
    size_t correct_count;
    float output[MNIST_CLASS_COUNT];

    correct_count = 0U;
    for (sample_index = 0U; sample_index < dataset->sample_count; ++sample_index) {
        const float* input = &dataset->images[sample_index * dataset->image_size];
        int predicted_label;
        size_t class_index;

        if (infer_auto_run(infer_ctx, input, output) != 0) {
            fprintf(stderr, "Inference failed at sample %zu\n", sample_index);
            return 0.0f;
        }

        predicted_label = mnist_dataset_argmax(output, MNIST_CLASS_COUNT);
        if ((uint8_t)predicted_label == dataset->labels[sample_index]) {
            correct_count++;
        }

        if (sample_index < MNIST_PREVIEW_COUNT) {
            printf("Sample %zu - expected %u, predicted %d\n",
                sample_index,
                (unsigned)dataset->labels[sample_index],
                predicted_label);
            mnist_dataset_render_ascii(input, MNIST_IMAGE_ROWS, MNIST_IMAGE_COLS);
            printf("Scores:");
            for (class_index = 0U; class_index < MNIST_CLASS_COUNT; ++class_index) {
                printf(" %zu:%.3f", class_index, output[class_index]);
            }
            printf("\n\n");
        }
    }

    return ((float)correct_count * 100.0f) / (float)dataset->sample_count;
}

int main(void) {
    MnistDataset test_dataset;
    void* infer_ctx;
    char error_buffer[256];
    const char* weights_file = "../data/weights.bin";
    float accuracy;

    (void)memset(&test_dataset, 0, sizeof(test_dataset));

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    if (mnist_dataset_load(
            MNIST_TEST_IMAGES,
            MNIST_TEST_LABELS,
            MNIST_TEST_SAMPLE_LIMIT,
            &test_dataset,
            error_buffer,
            sizeof(error_buffer)) != 0) {
        fprintf(stderr, "Failed to load test dataset: %s\n", error_buffer);
        return 1;
    }

    printf("=== MNIST Network Inference ===\n\n");
    printf("Loaded %zu test samples from source dataset copy.\n\n", test_dataset.sample_count);

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        mnist_dataset_free(&test_dataset);
        return 1;
    }

    if (weights_load_from_file(infer_ctx, weights_file) != 0) {
        fprintf(stderr, "Failed to load weights from %s\n", weights_file);
        infer_destroy(infer_ctx);
        mnist_dataset_free(&test_dataset);
        return 1;
    }

    accuracy = preview_predictions(infer_ctx, &test_dataset);
    printf("Accuracy on %zu test samples: %.2f%%\n", test_dataset.sample_count, accuracy);

    infer_destroy(infer_ctx);
    mnist_dataset_free(&test_dataset);
    return 0;
}
