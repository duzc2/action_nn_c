/**
 * @file cifar10_dataset.h
 * @brief CIFAR-10 binary dataset loader for the action_c CNN framework.
 *
 * CIFAR-10 stores images in a planar format: [R×1024][G×1024][B×1024].
 * The CNN backend expects interleaved RGB: [R0,G0,B0, R1,G1,B1, ...].
 * Conversion happens during loading; pixel values are normalized to [0,1].
 *
 * @see https://www.cs.toronto.edu/~kriz/cifar.html
 */

#ifndef DEMO_EDGE_VIDEO_PREPROCESS_CIFAR10_DATASET_H
#define DEMO_EDGE_VIDEO_PREPROCESS_CIFAR10_DATASET_H

#include <stddef.h>
#include <stdint.h>

/* CIFAR-10 image dimensions */
#define CIFAR10_IMAGE_WIDTH  32U
#define CIFAR10_IMAGE_HEIGHT 32U
#define CIFAR10_CHANNELS      3U
#define CIFAR10_IMAGE_SIZE   (CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT * CIFAR10_CHANNELS)
#define CIFAR10_CLASS_COUNT  10U

/**
 * @brief CIFAR-10 class labels (human-readable names).
 */
extern const char* const cifar10_class_names[CIFAR10_CLASS_COUNT];

/**
 * @brief CIFAR-10 dataset structure.
 */
typedef struct {
    size_t   sample_count;  /**< Number of loaded samples. */
    size_t   image_size;    /**< Pixels per sample (3072 = 32*32*3). */
    float*   images;        /**< Flat interleaved RGB float array [0,1]. */
    uint8_t* labels;        /**< Class labels (0-9). */
} Cifar10Dataset;

/**
 * @brief Load CIFAR-10 binary data from a single batch file.
 *
 * Reads the CIFAR-10 binary format (1-byte label + 3072 bytes planar RGB),
 * converts to interleaved RGB, normalizes to [0,1].
 *
 * @param path              Path to *.bin file (e.g. data_batch_1.bin).
 * @param max_samples       Maximum samples to load, 0 for all (10000).
 * @param out_dataset       Output dataset struct.
 * @param error_buffer      Error message buffer.
 * @param error_buffer_size Size of error buffer.
 * @return 0 on success, negative on failure.
 */
int cifar10_dataset_load_batch(
    const char* path,
    size_t max_samples,
    Cifar10Dataset* out_dataset,
    char* error_buffer,
    size_t error_buffer_size);

/**
 * @brief Load CIFAR-10 from multiple batch files and concatenate.
 *
 * @param batch_paths       Array of batch file paths.
 * @param batch_count       Number of batch files.
 * @param max_per_batch     Max samples per batch (0 = all).
 * @param out_dataset       Output combined dataset.
 * @param error_buffer      Error message buffer.
 * @param error_buffer_size Size of error buffer.
 * @return 0 on success, negative on failure.
 */
int cifar10_dataset_load_multiple(
    const char* const* batch_paths,
    size_t batch_count,
    size_t max_per_batch,
    Cifar10Dataset* out_dataset,
    char* error_buffer,
    size_t error_buffer_size);

/**
 * @brief Free dataset memory.
 * @param dataset Dataset to free.
 */
void cifar10_dataset_free(Cifar10Dataset* dataset);

/**
 * @brief Create a one-hot encoded target vector for a label.
 * @param label      Class label (0-9).
 * @param target     Output target vector (size CIFAR10_CLASS_COUNT).
 * @param class_count Number of classes.
 */
void cifar10_make_one_hot(uint8_t label, float* target, size_t class_count);

/**
 * @brief Find the index of the maximum value in an array.
 * @param values Array of values.
 * @param count  Number of elements.
 * @return Index of maximum value, -1 on error.
 */
int cifar10_argmax(const float* values, size_t count);

/**
 * @brief In-place Fisher-Yates shuffle of dataset (images + labels together).
 * @param dataset Dataset to shuffle.
 * @param seed    Random seed for reproducibility.
 */
void cifar10_dataset_shuffle(Cifar10Dataset* dataset, uint32_t seed);

/* ========================================================================
 *  Data augmentation
 * ======================================================================== */

/**
 * @brief Augmentation configuration for a single training sample.
 */
typedef struct {
    int horizontal_flip; /**< 0 or 1 (p=0.5 when applied randomly). */
    int pad_crop;        /**< 0 or 2 (random 2px pad + crop back to 32x32). */
} Cifar10Augment;

/**
 * @brief Apply data augmentation to a single CIFAR-10 sample.
 *
 * Performs horizontal flip (p=0.5) and random pad-crop (pad 2px, random
 * crop back to original size) when the corresponding flags are set.
 * The augmented image is written to dst; src is unchanged.
 *
 * @param dst     Output buffer of size w*h*c (interleaved RGB, float [0,1]).
 * @param src     Input sample (interleaved RGB, float [0,1]).
 * @param w       Image width (32 for CIFAR-10).
 * @param h       Image height (32 for CIFAR-10).
 * @param c       Channels (3 for RGB).
 * @param aug     Augmentation config (non-NULL).
 * @param seed    PRNG state pointer — updated across calls for reproducibility.
 */
void cifar10_augment_sample(
    float* dst, const float* src,
    size_t w, size_t h, size_t c,
    const Cifar10Augment* aug, uint32_t* seed);

#endif /* DEMO_EDGE_VIDEO_PREPROCESS_CIFAR10_DATASET_H */
