/**
 * @file mnist_cnn_dataset.c
 * @brief MNIST dataset helpers for the CNN+MLP demo.
 *
 * The CNN backend in this repository operates on one or more frames followed
 * by global pooling. To preserve coarse spatial information for handwritten
 * digits, this helper repacks each 28x28 digit into four 14x14 quadrants that
 * are fed as a 4-step sequence to the CNN encoder.
 */

#include "mnist_cnn_dataset.h"

#include <stdio.h>
#include <stdlib.h>

static void mnist_cnn_set_error(char* error_buffer, size_t error_buffer_size, const char* message) {
    if (error_buffer != NULL && error_buffer_size > 0U) {
        (void)snprintf(error_buffer, error_buffer_size, "%s", message);
    }
}

static int mnist_cnn_read_u32(FILE* file, uint32_t* out_value) {
    unsigned char bytes[4];

    if (file == NULL || out_value == NULL) {
        return -1;
    }
    if (fread(bytes, 1U, sizeof(bytes), file) != sizeof(bytes)) {
        return -1;
    }

    *out_value = ((uint32_t)bytes[0] << 24U) |
                 ((uint32_t)bytes[1] << 16U) |
                 ((uint32_t)bytes[2] << 8U) |
                 (uint32_t)bytes[3];
    return 0;
}

static FILE* mnist_cnn_open_binary_read(const char* path) {
#ifdef _WIN32
    FILE* file = NULL;
    if (path == NULL || fopen_s(&file, path, "rb") != 0) {
        return NULL;
    }
    return file;
#else
    return path == NULL ? NULL : fopen(path, "rb");
#endif
}

static void mnist_cnn_dataset_reset(MnistCnnDataset* dataset) {
    if (dataset != NULL) {
        dataset->sample_count = 0U;
        dataset->image_size = 0U;
        dataset->images = NULL;
        dataset->labels = NULL;
    }
}

int mnist_cnn_dataset_load(
    const char* images_path,
    const char* labels_path,
    size_t max_samples,
    MnistCnnDataset* out_dataset,
    char* error_buffer,
    size_t error_buffer_size) {
    FILE* images_file;
    FILE* labels_file;
    uint32_t image_magic;
    uint32_t image_count;
    uint32_t row_count;
    uint32_t column_count;
    uint32_t label_magic;
    uint32_t label_count;
    size_t sample_count;
    size_t sample_index;
    size_t pixel_index;
    size_t image_size;
    unsigned char image_bytes[MNIST_CNN_IMAGE_SIZE];

    if (images_path == NULL || labels_path == NULL || out_dataset == NULL) {
        mnist_cnn_set_error(error_buffer, error_buffer_size, "invalid dataset arguments");
        return -1;
    }

    mnist_cnn_dataset_reset(out_dataset);

    images_file = mnist_cnn_open_binary_read(images_path);
    if (images_file == NULL) {
        mnist_cnn_set_error(error_buffer, error_buffer_size, "failed to open image file");
        return -1;
    }

    labels_file = mnist_cnn_open_binary_read(labels_path);
    if (labels_file == NULL) {
        fclose(images_file);
        mnist_cnn_set_error(error_buffer, error_buffer_size, "failed to open label file");
        return -1;
    }

    if (mnist_cnn_read_u32(images_file, &image_magic) != 0 ||
        mnist_cnn_read_u32(images_file, &image_count) != 0 ||
        mnist_cnn_read_u32(images_file, &row_count) != 0 ||
        mnist_cnn_read_u32(images_file, &column_count) != 0) {
        fclose(images_file);
        fclose(labels_file);
        mnist_cnn_set_error(error_buffer, error_buffer_size, "failed to read image IDX header");
        return -1;
    }

    if (mnist_cnn_read_u32(labels_file, &label_magic) != 0 ||
        mnist_cnn_read_u32(labels_file, &label_count) != 0) {
        fclose(images_file);
        fclose(labels_file);
        mnist_cnn_set_error(error_buffer, error_buffer_size, "failed to read label IDX header");
        return -1;
    }

    if (image_magic != 2051U || label_magic != 2049U) {
        fclose(images_file);
        fclose(labels_file);
        mnist_cnn_set_error(error_buffer, error_buffer_size, "unexpected IDX magic number");
        return -1;
    }

    if (row_count != MNIST_CNN_IMAGE_ROWS || column_count != MNIST_CNN_IMAGE_COLS) {
        fclose(images_file);
        fclose(labels_file);
        mnist_cnn_set_error(error_buffer, error_buffer_size, "unexpected MNIST image dimensions");
        return -1;
    }

    if (image_count != label_count) {
        fclose(images_file);
        fclose(labels_file);
        mnist_cnn_set_error(error_buffer, error_buffer_size, "image and label count mismatch");
        return -1;
    }

    sample_count = (size_t)image_count;
    if (max_samples > 0U && max_samples < sample_count) {
        sample_count = max_samples;
    }
    image_size = (size_t)row_count * (size_t)column_count;

    out_dataset->images = (float*)calloc(sample_count * image_size, sizeof(float));
    out_dataset->labels = (uint8_t*)calloc(sample_count, sizeof(uint8_t));
    if (out_dataset->images == NULL || out_dataset->labels == NULL) {
        mnist_cnn_dataset_free(out_dataset);
        fclose(images_file);
        fclose(labels_file);
        mnist_cnn_set_error(error_buffer, error_buffer_size, "failed to allocate dataset buffers");
        return -1;
    }

    for (sample_index = 0U; sample_index < sample_count; ++sample_index) {
        if (fread(image_bytes, 1U, image_size, images_file) != image_size) {
            mnist_cnn_dataset_free(out_dataset);
            fclose(images_file);
            fclose(labels_file);
            mnist_cnn_set_error(error_buffer, error_buffer_size, "failed to read image payload");
            return -1;
        }
        if (fread(&out_dataset->labels[sample_index], 1U, 1U, labels_file) != 1U) {
            mnist_cnn_dataset_free(out_dataset);
            fclose(images_file);
            fclose(labels_file);
            mnist_cnn_set_error(error_buffer, error_buffer_size, "failed to read label payload");
            return -1;
        }
        for (pixel_index = 0U; pixel_index < image_size; ++pixel_index) {
            out_dataset->images[(sample_index * image_size) + pixel_index] =
                (float)image_bytes[pixel_index] / 255.0f;
        }
    }

    fclose(images_file);
    fclose(labels_file);

    out_dataset->sample_count = sample_count;
    out_dataset->image_size = image_size;
    mnist_cnn_set_error(error_buffer, error_buffer_size, "");
    return 0;
}

void mnist_cnn_dataset_free(MnistCnnDataset* dataset) {
    if (dataset == NULL) {
        return;
    }

    free(dataset->images);
    free(dataset->labels);
    mnist_cnn_dataset_reset(dataset);
}

void mnist_cnn_dataset_make_one_hot(uint8_t label, float* target, size_t class_count) {
    size_t class_index;

    if (target == NULL) {
        return;
    }

    for (class_index = 0U; class_index < class_count; ++class_index) {
        target[class_index] = 0.0f;
    }
    if ((size_t)label < class_count) {
        target[label] = 1.0f;
    }
}

int mnist_cnn_dataset_argmax(const float* values, size_t count) {
    size_t index;
    size_t best_index;

    if (values == NULL || count == 0U) {
        return -1;
    }

    best_index = 0U;
    for (index = 1U; index < count; ++index) {
        if (values[index] > values[best_index]) {
            best_index = index;
        }
    }

    return (int)best_index;
}

void mnist_cnn_dataset_render_ascii(const float* image, size_t rows, size_t cols) {
    static const char kRamp[] = " .:-=+*#%@";
    size_t row;
    size_t col;
    size_t last_index;

    if (image == NULL || rows == 0U || cols == 0U) {
        return;
    }

    last_index = sizeof(kRamp) - 2U;
    for (row = 0U; row < rows; ++row) {
        for (col = 0U; col < cols; ++col) {
            float value = image[(row * cols) + col];
            size_t ramp_index = (size_t)(value * (float)last_index + 0.5f);
            if (ramp_index > last_index) {
                ramp_index = last_index;
            }
            putchar((int)kRamp[ramp_index]);
        }
        putchar('\n');
    }
}

void mnist_cnn_pack_quadrants(const float* image, float* packed_input) {
    static const size_t kRowOffsets[MNIST_CNN_SEQUENCE_LENGTH] = {0U, 0U, 14U, 14U};
    static const size_t kColOffsets[MNIST_CNN_SEQUENCE_LENGTH] = {0U, 14U, 0U, 14U};
    size_t quadrant_index;

    if (image == NULL || packed_input == NULL) {
        return;
    }

    for (quadrant_index = 0U; quadrant_index < MNIST_CNN_SEQUENCE_LENGTH; ++quadrant_index) {
        size_t row;
        size_t row_offset = kRowOffsets[quadrant_index];
        size_t col_offset = kColOffsets[quadrant_index];
        size_t quadrant_base = quadrant_index * (MNIST_CNN_QUADRANT_ROWS * MNIST_CNN_QUADRANT_COLS);

        for (row = 0U; row < MNIST_CNN_QUADRANT_ROWS; ++row) {
            size_t col;
            for (col = 0U; col < MNIST_CNN_QUADRANT_COLS; ++col) {
                size_t source_index = ((row + row_offset) * MNIST_CNN_IMAGE_COLS) + (col + col_offset);
                size_t target_index = quadrant_base + (row * MNIST_CNN_QUADRANT_COLS) + col;
                packed_input[target_index] = image[source_index];
            }
        }
    }
}
