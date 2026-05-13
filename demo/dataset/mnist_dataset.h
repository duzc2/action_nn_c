/**
 * @file mnist_dataset.h
 * @brief Unified MNIST IDX loader shared by all MNIST-based demos.
 *
 * Loads the standard MNIST binary format (big-endian IDX), provides
 * one-hot encoding, argmax, ASCII rendering, and a CNN quadrant-packing
 * helper for the 28x28-to-4x(14x14) transform.
 */

#ifndef DEMO_DATASET_MNIST_DATASET_H
#define DEMO_DATASET_MNIST_DATASET_H

#include <stddef.h>
#include <stdint.h>

#define MNIST_IMAGE_ROWS 28U
#define MNIST_IMAGE_COLS 28U
#define MNIST_IMAGE_SIZE (MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS)
#define MNIST_CLASS_COUNT 10U

/* ── CNN quadrant packing constants ── */
#define MNIST_QUADRANT_ROWS 14U
#define MNIST_QUADRANT_COLS 14U
#define MNIST_SEQUENCE_LENGTH 4U
#define MNIST_FEATURE_INPUT_SIZE \
    (MNIST_QUADRANT_ROWS * MNIST_QUADRANT_COLS * MNIST_SEQUENCE_LENGTH)

typedef struct {
    size_t   sample_count;
    size_t   image_size;
    float*   images;
    uint8_t* labels;
} MnistDataset;

int  mnist_dataset_load(const char* images_path,
                        const char* labels_path,
                        size_t max_samples,
                        MnistDataset* out_dataset,
                        char* error_buffer,
                        size_t error_buffer_size);

void mnist_dataset_free(MnistDataset* dataset);
void mnist_dataset_make_one_hot(uint8_t label, float* target, size_t class_count);
int  mnist_dataset_argmax(const float* values, size_t count);
void mnist_dataset_render_ascii(const float* image, size_t rows, size_t cols);

/**
 * @brief Split a 28x28 MNIST image into four 14x14 quadrants.
 *
 * The CNN backend operates on multi-frame sequences. This helper
 * repacks a single flat 784-element image into a 4-step sequence
 * where each step is one 196-element quadrant (TL, TR, BL, BR).
 *
 * @param image        Flat 784-element float image [MNIST_IMAGE_SIZE].
 * @param packed_input  Output buffer sized [MNIST_FEATURE_INPUT_SIZE].
 */
void mnist_pack_quadrants(const float* image, float* packed_input);

#endif /* DEMO_DATASET_MNIST_DATASET_H */
