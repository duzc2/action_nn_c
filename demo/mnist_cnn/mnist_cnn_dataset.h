/**
 * @file mnist_cnn_dataset.h
 * @brief MNIST IDX loader and CNN-specific input packing helpers.
 */

#ifndef DEMO_MNIST_CNN_DATASET_H
#define DEMO_MNIST_CNN_DATASET_H

#include <stddef.h>
#include <stdint.h>

#define MNIST_CNN_IMAGE_ROWS 28U
#define MNIST_CNN_IMAGE_COLS 28U
#define MNIST_CNN_IMAGE_SIZE (MNIST_CNN_IMAGE_ROWS * MNIST_CNN_IMAGE_COLS)
#define MNIST_CNN_CLASS_COUNT 10U
#define MNIST_CNN_QUADRANT_ROWS 14U
#define MNIST_CNN_QUADRANT_COLS 14U
#define MNIST_CNN_SEQUENCE_LENGTH 4U
#define MNIST_CNN_FEATURE_INPUT_SIZE (MNIST_CNN_QUADRANT_ROWS * MNIST_CNN_QUADRANT_COLS * MNIST_CNN_SEQUENCE_LENGTH)

typedef struct {
    size_t sample_count;
    size_t image_size;
    float* images;
    uint8_t* labels;
} MnistCnnDataset;

int mnist_cnn_dataset_load(
    const char* images_path,
    const char* labels_path,
    size_t max_samples,
    MnistCnnDataset* out_dataset,
    char* error_buffer,
    size_t error_buffer_size);

void mnist_cnn_dataset_free(MnistCnnDataset* dataset);
void mnist_cnn_dataset_make_one_hot(uint8_t label, float* target, size_t class_count);
int mnist_cnn_dataset_argmax(const float* values, size_t count);
void mnist_cnn_dataset_render_ascii(const float* image, size_t rows, size_t cols);
void mnist_cnn_pack_quadrants(const float* image, float* packed_input);

#endif
