/**
 * @file train_main.c
 * @brief MNIST CNN+MLP demo training entry.
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "mnist_cnn_dataset.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>

#define MNIST_CNN_TRAIN_SAMPLE_LIMIT 4000U
#define MNIST_CNN_EVAL_SAMPLE_LIMIT 1000U
#define MNIST_CNN_EPOCH_COUNT 20U

#define MNIST_CNN_DATASET_ROOT "../../../../demo/mnist_cnn/dataset"
#define MNIST_CNN_TRAIN_IMAGES MNIST_CNN_DATASET_ROOT "/train-images.idx3-ubyte"
#define MNIST_CNN_TRAIN_LABELS MNIST_CNN_DATASET_ROOT "/train-labels.idx1-ubyte"
#define MNIST_CNN_TEST_IMAGES MNIST_CNN_DATASET_ROOT "/t10k-images.idx3-ubyte"
#define MNIST_CNN_TEST_LABELS MNIST_CNN_DATASET_ROOT "/t10k-labels.idx1-ubyte"

static float evaluate_accuracy(void* infer_ctx, const MnistCnnDataset* dataset) {
    size_t sample_index;
    size_t correct_count;
    float packed_input[MNIST_CNN_FEATURE_INPUT_SIZE];
    float output[MNIST_CNN_CLASS_COUNT];

    if (infer_ctx == NULL || dataset == NULL || dataset->sample_count == 0U) {
        return 0.0f;
    }

    correct_count = 0U;
    for (sample_index = 0U; sample_index < dataset->sample_count; ++sample_index) {
        const float* image = &dataset->images[sample_index * dataset->image_size];
        mnist_cnn_pack_quadrants(image, packed_input);
        if (infer_auto_run(infer_ctx, packed_input, output) != 0) {
            return 0.0f;
        }
        if ((uint8_t)mnist_cnn_dataset_argmax(output, MNIST_CNN_CLASS_COUNT) == dataset->labels[sample_index]) {
            correct_count++;
        }
    }

    return ((float)correct_count * 100.0f) / (float)dataset->sample_count;
}

int main(void) {
    MnistCnnDataset train_dataset;
    MnistCnnDataset eval_dataset;
    void* infer_ctx;
    void* train_ctx;
    char error_buffer[256];
    const char* output_file = "../data/weights.bin";
    size_t epoch_index;
    size_t sample_index;
    float packed_input[MNIST_CNN_FEATURE_INPUT_SIZE];
    float target[MNIST_CNN_CLASS_COUNT];
    float average_loss;

    (void)memset(&train_dataset, 0, sizeof(train_dataset));
    (void)memset(&eval_dataset, 0, sizeof(eval_dataset));

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    if (mnist_cnn_dataset_load(
            MNIST_CNN_TRAIN_IMAGES,
            MNIST_CNN_TRAIN_LABELS,
            MNIST_CNN_TRAIN_SAMPLE_LIMIT,
            &train_dataset,
            error_buffer,
            sizeof(error_buffer)) != 0) {
        fprintf(stderr, "Failed to load train dataset: %s\n", error_buffer);
        return 1;
    }

    if (mnist_cnn_dataset_load(
            MNIST_CNN_TEST_IMAGES,
            MNIST_CNN_TEST_LABELS,
            MNIST_CNN_EVAL_SAMPLE_LIMIT,
            &eval_dataset,
            error_buffer,
            sizeof(error_buffer)) != 0) {
        fprintf(stderr, "Failed to load eval dataset: %s\n", error_buffer);
        mnist_cnn_dataset_free(&train_dataset);
        return 1;
    }

    printf("=== MNIST CNN+MLP Network Training ===\n\n");
    printf("Train samples: %zu\n", train_dataset.sample_count);
    printf("Eval samples:  %zu\n", eval_dataset.sample_count);
    printf("Epochs:        %u\n", (unsigned)MNIST_CNN_EPOCH_COUNT);
    printf("Input packing: 4 quadrants of 14x14 each -> CNN encoder -> MLP head\n\n");

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        mnist_cnn_dataset_free(&eval_dataset);
        mnist_cnn_dataset_free(&train_dataset);
        return 1;
    }

    train_ctx = train_create(infer_ctx);
    if (train_ctx == NULL) {
        fprintf(stderr, "Failed to create training context\n");
        infer_destroy(infer_ctx);
        mnist_cnn_dataset_free(&eval_dataset);
        mnist_cnn_dataset_free(&train_dataset);
        return 1;
    }

    for (epoch_index = 0U; epoch_index < MNIST_CNN_EPOCH_COUNT; ++epoch_index) {
        float epoch_loss_sum = 0.0f;
        float eval_accuracy;

        for (sample_index = 0U; sample_index < train_dataset.sample_count; ++sample_index) {
            const float* image = &train_dataset.images[sample_index * train_dataset.image_size];
            mnist_cnn_pack_quadrants(image, packed_input);
            mnist_cnn_dataset_make_one_hot(train_dataset.labels[sample_index], target, MNIST_CNN_CLASS_COUNT);

            if (train_step(train_ctx, packed_input, target) != 0) {
                fprintf(stderr, "Training step failed at epoch %zu sample %zu\n",
                    epoch_index + 1U,
                    sample_index + 1U);
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                mnist_cnn_dataset_free(&eval_dataset);
                mnist_cnn_dataset_free(&train_dataset);
                return 1;
            }
            epoch_loss_sum += train_get_loss(train_ctx);
        }

        average_loss = epoch_loss_sum / (float)train_dataset.sample_count;
        eval_accuracy = evaluate_accuracy(infer_ctx, &eval_dataset);
        printf("Epoch %zu/%u - avg loss: %.4f - eval accuracy: %.2f%%\n",
            epoch_index + 1U,
            (unsigned)MNIST_CNN_EPOCH_COUNT,
            average_loss,
            eval_accuracy);
    }

    if (weights_save_to_file(infer_ctx, output_file) != 0) {
        fprintf(stderr, "Failed to save weights to %s\n", output_file);
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        mnist_cnn_dataset_free(&eval_dataset);
        mnist_cnn_dataset_free(&train_dataset);
        return 1;
    }

    printf("\nWeights saved to: %s\n", output_file);

    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    mnist_cnn_dataset_free(&eval_dataset);
    mnist_cnn_dataset_free(&train_dataset);
    return 0;
}
