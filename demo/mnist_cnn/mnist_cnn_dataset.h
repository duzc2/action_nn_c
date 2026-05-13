/**
 * @file mnist_cnn_dataset.h
 * @brief Compatibility shim — delegates to the unified dataset loader.
 *
 * Provides legacy type/macro aliases so existing calling code
 * continues to compile without changes.
 */
#ifndef DEMO_MNIST_CNN_DATASET_H
#define DEMO_MNIST_CNN_DATASET_H
#include "../dataset/mnist_dataset.h"

/* Legacy type alias */
typedef MnistDataset MnistCnnDataset;

/* Legacy API aliases */
#define mnist_cnn_dataset_load(...)      mnist_dataset_load(__VA_ARGS__)
#define mnist_cnn_dataset_free           mnist_dataset_free
#define mnist_cnn_dataset_make_one_hot   mnist_dataset_make_one_hot
#define mnist_cnn_dataset_argmax         mnist_dataset_argmax
#define mnist_cnn_dataset_render_ascii   mnist_dataset_render_ascii
#define mnist_cnn_pack_quadrants         mnist_pack_quadrants

/* Legacy constant aliases */
#define MNIST_CNN_IMAGE_ROWS      MNIST_IMAGE_ROWS
#define MNIST_CNN_IMAGE_COLS      MNIST_IMAGE_COLS
#define MNIST_CNN_IMAGE_SIZE      MNIST_IMAGE_SIZE
#define MNIST_CNN_CLASS_COUNT     MNIST_CLASS_COUNT
#define MNIST_CNN_QUADRANT_ROWS   MNIST_QUADRANT_ROWS
#define MNIST_CNN_QUADRANT_COLS   MNIST_QUADRANT_COLS
#define MNIST_CNN_SEQUENCE_LENGTH MNIST_SEQUENCE_LENGTH
#define MNIST_CNN_FEATURE_INPUT_SIZE MNIST_FEATURE_INPUT_SIZE

#endif
