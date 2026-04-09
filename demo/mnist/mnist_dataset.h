/**
 * @file mnist_dataset.h
 * @brief Small IDX loader shared by the MNIST demo train/infer executables.
 */

#ifndef DEMO_MNIST_DATASET_H
#define DEMO_MNIST_DATASET_H

#include <stddef.h>
#include <stdint.h>

#define MNIST_IMAGE_ROWS 28U
#define MNIST_IMAGE_COLS 28U
#define MNIST_IMAGE_SIZE (MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS)
#define MNIST_CLASS_COUNT 10U

typedef struct {
    size_t sample_count;
    size_t image_size;
    float* images;
    uint8_t* labels;
} MnistDataset;

int mnist_dataset_load(
    const char* images_path,
    const char* labels_path,
    size_t max_samples,
    MnistDataset* out_dataset,
    char* error_buffer,
    size_t error_buffer_size);

void mnist_dataset_free(MnistDataset* dataset);
void mnist_dataset_make_one_hot(uint8_t label, float* target, size_t class_count);
int mnist_dataset_argmax(const float* values, size_t count);
void mnist_dataset_render_ascii(const float* image, size_t rows, size_t cols);

#endif
