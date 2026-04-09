/**
 * @file train_main.c
 * @brief MNIST demo training entry based on profiler-generated APIs.
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "mnist_dataset.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>

#define MNIST_TRAIN_SAMPLE_LIMIT 1500U
#define MNIST_EVAL_SAMPLE_LIMIT 300U
#define MNIST_EPOCH_COUNT 5U

#define MNIST_DATASET_ROOT "../../../../demo/mnist/dataset"
#define MNIST_TRAIN_IMAGES MNIST_DATASET_ROOT "/train-images.idx3-ubyte"
#define MNIST_TRAIN_LABELS MNIST_DATASET_ROOT "/train-labels.idx1-ubyte"
#define MNIST_TEST_IMAGES MNIST_DATASET_ROOT "/t10k-images.idx3-ubyte"
#define MNIST_TEST_LABELS MNIST_DATASET_ROOT "/t10k-labels.idx1-ubyte"

/**
 * @brief Run one quick accuracy pass on a dataset subset.
 */
static float evaluate_accuracy(void* infer_ctx, const MnistDataset* dataset) {
    size_t sample_index;
    size_t correct_count;
    float output[MNIST_CLASS_COUNT];

    if (infer_ctx == NULL || dataset == NULL || dataset->sample_count == 0U) {
        return 0.0f;
    }

    correct_count = 0U;
    for (sample_index = 0U; sample_index < dataset->sample_count; ++sample_index) {
        const float* input = &dataset->images[sample_index * dataset->image_size];
        if (infer_auto_run(infer_ctx, input, output) != 0) {
            return 0.0f;
        }
        if ((uint8_t)mnist_dataset_argmax(output, MNIST_CLASS_COUNT) == dataset->labels[sample_index]) {
            correct_count++;
        }
    }

    return ((float)correct_count * 100.0f) / (float)dataset->sample_count;
}

int main(void) {
    MnistDataset train_dataset;
    MnistDataset eval_dataset;
    void* infer_ctx;
    void* train_ctx;
    char error_buffer[256];
    const char* output_file = "../data/weights.bin";
    size_t epoch_index;
    size_t sample_index;
    float target[MNIST_CLASS_COUNT];
    float average_loss;

    (void)memset(&train_dataset, 0, sizeof(train_dataset));
    (void)memset(&eval_dataset, 0, sizeof(eval_dataset));

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    if (mnist_dataset_load(
            MNIST_TRAIN_IMAGES,
            MNIST_TRAIN_LABELS,
            MNIST_TRAIN_SAMPLE_LIMIT,
            &train_dataset,
            error_buffer,
            sizeof(error_buffer)) != 0) {
        fprintf(stderr, "Failed to load train dataset: %s\n", error_buffer);
        return 1;
    }

    if (mnist_dataset_load(
            MNIST_TEST_IMAGES,
            MNIST_TEST_LABELS,
            MNIST_EVAL_SAMPLE_LIMIT,
            &eval_dataset,
            error_buffer,
            sizeof(error_buffer)) != 0) {
        fprintf(stderr, "Failed to load eval dataset: %s\n", error_buffer);
        mnist_dataset_free(&train_dataset);
        return 1;
    }

    printf("=== MNIST Network Training ===\n\n");
    printf("Train samples: %zu\n", train_dataset.sample_count);
    printf("Eval samples:  %zu\n", eval_dataset.sample_count);
    printf("Epochs:        %u\n\n", (unsigned)MNIST_EPOCH_COUNT);

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        mnist_dataset_free(&eval_dataset);
        mnist_dataset_free(&train_dataset);
        return 1;
    }

    train_ctx = train_create(infer_ctx);
    if (train_ctx == NULL) {
        fprintf(stderr, "Failed to create training context\n");
        infer_destroy(infer_ctx);
        mnist_dataset_free(&eval_dataset);
        mnist_dataset_free(&train_dataset);
        return 1;
    }

    for (epoch_index = 0U; epoch_index < MNIST_EPOCH_COUNT; ++epoch_index) {
        float epoch_loss_sum = 0.0f;
        float eval_accuracy;

        for (sample_index = 0U; sample_index < train_dataset.sample_count; ++sample_index) {
            const float* input = &train_dataset.images[sample_index * train_dataset.image_size];
            mnist_dataset_make_one_hot(train_dataset.labels[sample_index], target, MNIST_CLASS_COUNT);

            if (train_step(train_ctx, input, target) != 0) {
                fprintf(stderr, "Training step failed at epoch %zu sample %zu\n",
                    epoch_index + 1U,
                    sample_index + 1U);
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                mnist_dataset_free(&eval_dataset);
                mnist_dataset_free(&train_dataset);
                return 1;
            }
            epoch_loss_sum += train_get_loss(train_ctx);
        }

        average_loss = epoch_loss_sum / (float)train_dataset.sample_count;
        eval_accuracy = evaluate_accuracy(infer_ctx, &eval_dataset);
        printf("Epoch %zu/%u - avg loss: %.4f - eval accuracy: %.2f%%\n",
            epoch_index + 1U,
            (unsigned)MNIST_EPOCH_COUNT,
            average_loss,
            eval_accuracy);
    }

    if (weights_save_to_file(infer_ctx, output_file) != 0) {
        fprintf(stderr, "Failed to save weights to %s\n", output_file);
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        mnist_dataset_free(&eval_dataset);
        mnist_dataset_free(&train_dataset);
        return 1;
    }

    printf("\nWeights saved to: %s\n", output_file);

    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    mnist_dataset_free(&eval_dataset);
    mnist_dataset_free(&train_dataset);
    return 0;
}
