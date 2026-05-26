/**
 * @file cifar10_dataset.c
 * @brief CIFAR-10 binary dataset loader implementation.
 *
 * Reads the CIFAR-10 binary format where each record is:
 *   [1 byte label][1024 bytes R][1024 bytes G][1024 bytes B]
 * Converts planar RGB to interleaved RGB and normalizes to [0,1] float.
 */

#include "cifar10_dataset.h"
#include "demo_edge_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* const cifar10_class_names[CIFAR10_CLASS_COUNT] = {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
};

int cifar10_dataset_load_batch(
    const char* path,
    size_t max_samples,
    Cifar10Dataset* out_dataset,
    char* error_buffer,
    size_t error_buffer_size)
{
    FILE* file;
    const size_t kRecordSize = 1U + CIFAR10_IMAGE_SIZE;  /* 1 label + 3072 pixels */
    const size_t kPlaneSize = CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT;
    unsigned char record[1U + CIFAR10_IMAGE_SIZE];
    size_t bytes_read;
    size_t sample_index;
    size_t pixel_index;

    if (path == NULL || out_dataset == NULL) {
        demo_set_error(error_buffer, error_buffer_size, "invalid arguments");
        return -1;
    }

    (void)memset(out_dataset, 0, sizeof(*out_dataset));

#ifdef _WIN32
    if (fopen_s(&file, path, "rb") != 0) file = NULL;
#else
    file = fopen(path, "rb");
#endif
    if (file == NULL) {
        demo_set_error(error_buffer, error_buffer_size, "failed to open batch file");
        return -1;
    }

    /* Determine how many records to read. */
    {
        long file_size;
        if (fseek(file, 0, SEEK_END) != 0) {
            fclose(file);
            demo_set_error(error_buffer, error_buffer_size, "failed to seek batch file");
            return -1;
        }
        file_size = ftell(file);
        if (file_size < 0 || (size_t)file_size < kRecordSize) {
            fclose(file);
            demo_set_error(error_buffer, error_buffer_size, "batch file too small");
            return -1;
        }
        rewind(file);

        size_t total_records = (size_t)file_size / kRecordSize;
        if (max_samples > 0U && max_samples < total_records) {
            total_records = max_samples;
        }

        /* Allocate memory. */
        out_dataset->images = (float*)calloc(total_records * CIFAR10_IMAGE_SIZE, sizeof(float));
        out_dataset->labels = (uint8_t*)calloc(total_records, sizeof(uint8_t));
        if (out_dataset->images == NULL || out_dataset->labels == NULL) {
            cifar10_dataset_free(out_dataset);
            fclose(file);
            demo_set_error(error_buffer, error_buffer_size, "failed to allocate dataset");
            return -1;
        }
        out_dataset->sample_count = total_records;
        out_dataset->image_size = CIFAR10_IMAGE_SIZE;
    }

    /* Read and convert each record. */
    for (sample_index = 0U; sample_index < out_dataset->sample_count; ++sample_index) {
        bytes_read = fread(record, 1U, kRecordSize, file);
        if (bytes_read != kRecordSize) {
            cifar10_dataset_free(out_dataset);
            fclose(file);
            demo_set_error(error_buffer, error_buffer_size, "failed to read record");
            return -1;
        }

        out_dataset->labels[sample_index] = record[0];

        /* Convert planar RGB to interleaved RGB, normalize to [0,1] */
        for (pixel_index = 0U; pixel_index < kPlaneSize; ++pixel_index) {
            size_t base = sample_index * CIFAR10_IMAGE_SIZE + pixel_index * CIFAR10_CHANNELS;
            out_dataset->images[base + 0U] = (float)record[1U + 0U * kPlaneSize + pixel_index] / 255.0f;
            out_dataset->images[base + 1U] = (float)record[1U + 1U * kPlaneSize + pixel_index] / 255.0f;
            out_dataset->images[base + 2U] = (float)record[1U + 2U * kPlaneSize + pixel_index] / 255.0f;
        }
    }

    fclose(file);
    demo_set_error(error_buffer, error_buffer_size, "");
    return 0;
}

int cifar10_dataset_load_multiple(
    const char* const* batch_paths,
    size_t batch_count,
    size_t max_per_batch,
    Cifar10Dataset* out_dataset,
    char* error_buffer,
    size_t error_buffer_size)
{
    Cifar10Dataset cumulative;
    Cifar10Dataset batch;
    size_t batch_index;
    size_t sample_index;
    size_t max_samples;
    size_t total_samples;

    if (batch_paths == NULL || out_dataset == NULL || batch_count == 0U) {
        demo_set_error(error_buffer, error_buffer_size, "invalid arguments");
        return -1;
    }

    (void)memset(out_dataset, 0, sizeof(*out_dataset));
    (void)memset(&cumulative, 0, sizeof(cumulative));
    cumulative.image_size = CIFAR10_IMAGE_SIZE;

    /* Pre-allocate for the worst-case total to avoid per-batch realloc. */
    max_samples = batch_count * max_per_batch;
    cumulative.images = (float*)calloc(max_samples * CIFAR10_IMAGE_SIZE, sizeof(float));
    cumulative.labels = (uint8_t*)calloc(max_samples, sizeof(uint8_t));
    if (cumulative.images == NULL || cumulative.labels == NULL) {
        cifar10_dataset_free(&cumulative);
        demo_set_error(error_buffer, error_buffer_size, "failed to allocate dataset");
        return -1;
    }

    total_samples = 0U;
    for (batch_index = 0U; batch_index < batch_count; ++batch_index) {
        if (cifar10_dataset_load_batch(
                batch_paths[batch_index],
                max_per_batch,
                &batch,
                error_buffer,
                error_buffer_size) != 0) {
            cifar10_dataset_free(&cumulative);
            return -1;
        }

        /* Copy batch data to pre-allocated buffer. */
        for (sample_index = 0U; sample_index < batch.sample_count; ++sample_index) {
            size_t src_offset = sample_index * CIFAR10_IMAGE_SIZE;
            size_t dst_offset = total_samples * CIFAR10_IMAGE_SIZE;
            size_t p;
            for (p = 0U; p < CIFAR10_IMAGE_SIZE; ++p) {
                cumulative.images[dst_offset + p] = batch.images[src_offset + p];
            }
            cumulative.labels[total_samples] = batch.labels[sample_index];
            total_samples++;
        }

        cifar10_dataset_free(&batch);
    }

    cumulative.sample_count = total_samples;
    *out_dataset = cumulative;
    demo_set_error(error_buffer, error_buffer_size, "");
    return 0;
}

void cifar10_dataset_free(Cifar10Dataset* dataset) {
    if (dataset == NULL) return;

    free(dataset->images);
    free(dataset->labels);
    dataset->images = NULL;
    dataset->labels = NULL;
    dataset->sample_count = 0U;
    dataset->image_size = 0U;
}

void cifar10_make_one_hot(uint8_t label, float* target, size_t class_count) {
    size_t i;

    if (target == NULL || class_count == 0U) return;

    for (i = 0U; i < class_count; ++i) {
        target[i] = 0.0f;
    }

    /* Out-of-range label: fall back to class 0 instead of all-zeros */
    if ((size_t)label < class_count) {
        target[label] = 1.0f;
    } else {
        target[0U] = 1.0f;
    }
}

void cifar10_dataset_shuffle(Cifar10Dataset* dataset, uint32_t seed) {
    size_t i;

    if (dataset == NULL || dataset->sample_count <= 1U || dataset->images == NULL) return;

    /* Simple LCG PRNG: constants from glibc */
    for (i = dataset->sample_count - 1U; i > 0U; --i) {
        size_t j;
        size_t src_off, dst_off;
        size_t p;
        uint8_t temp_label;

        seed = seed * 1103515245U + 12345U;
        j = (size_t)(((uint64_t)seed >> 16U) % (i + 1U));

        /* Swap images (element by element, 3072 floats each) */
        src_off = i * CIFAR10_IMAGE_SIZE;
        dst_off = j * CIFAR10_IMAGE_SIZE;
        for (p = 0U; p < CIFAR10_IMAGE_SIZE; ++p) {
            float tmp = dataset->images[src_off + p];
            dataset->images[src_off + p] = dataset->images[dst_off + p];
            dataset->images[dst_off + p] = tmp;
        }

        /* Swap labels */
        temp_label = dataset->labels[i];
        dataset->labels[i] = dataset->labels[j];
        dataset->labels[j] = temp_label;
    }
}

void cifar10_augment_sample(
    float* dst, const float* src,
    size_t w, size_t h, size_t c,
    const Cifar10Augment* aug, uint32_t* seed)
{
    size_t x, y, ch;
    uint32_t rng = (seed != NULL) ? *seed : 12345U;
    int do_flip = 0;
    int pad;

    if (dst == NULL || src == NULL || aug == NULL || w == 0U || h == 0U || c == 0U) return;

    /* Decide whether to flip horizontally (p=0.5) */
    if (aug->horizontal_flip) {
        rng = rng * 1103515245U + 12345U;
        do_flip = ((rng >> 16U) & 1U) ? 1 : 0;
    }

    /* Pad size */
    pad = aug->pad_crop;
    if (pad < 0) pad = 0;

    if (pad == 0) {
        /* Simple copy with optional horizontal flip — no pad/crop. */
        for (y = 0U; y < h; ++y) {
            for (x = 0U; x < w; ++x) {
                size_t src_x = do_flip ? (w - 1U - x) : x;
                size_t src_off = (y * w + src_x) * c;
                size_t dst_off = (y * w + x) * c;
                for (ch = 0U; ch < c; ++ch) {
                    dst[dst_off + ch] = src[src_off + ch];
                }
            }
        }
    } else {
        /* Pad-and-crop: pad with edge replication, then random crop.
         * Pad the source image to (w+2*pad, h+2*pad), then crop back to (w,h). */
        size_t pw = w + (size_t)(2U * (unsigned)pad);
        size_t ph = h + (size_t)(2U * (unsigned)pad);
        int crop_x, crop_y;

        rng = rng * 1103515245U + 12345U;
        crop_x = (int)((rng >> 16U) % ((unsigned)(pw - w + 1U)));
        rng = rng * 1103515245U + 12345U;
        crop_y = (int)((rng >> 16U) % ((unsigned)(ph - h + 1U)));

        for (y = 0U; y < h; ++y) {
            for (x = 0U; x < w; ++x) {
                int sx, sy;
                size_t src_x, src_y;

                /* Map crop coordinate to source coordinate */
                sx = (int)x + crop_x - pad;
                sy = (int)y + crop_y - pad;

                /* Clamp to source boundaries (edge replication) */
                if (sx < 0) sx = 0;
                if (sx >= (int)w) sx = (int)w - 1;
                if (sy < 0) sy = 0;
                if (sy >= (int)h) sy = (int)h - 1;

                src_x = do_flip ? (w - 1U - (size_t)sx) : (size_t)sx;
                src_y = (size_t)sy;

                {
                    size_t src_off = (src_y * w + src_x) * c;
                    size_t dst_off = (y * w + x) * c;
                    for (ch = 0U; ch < c; ++ch) {
                        dst[dst_off + ch] = src[src_off + ch];
                    }
                }
            }
        }
    }

    if (seed != NULL) *seed = rng;
}

int cifar10_argmax(const float* values, size_t count) {
    size_t i;
    int best_index;

    if (values == NULL || count == 0U) return -1;

    best_index = 0;
    for (i = 1U; i < count; ++i) {
        if (values[i] > values[(size_t)best_index]) {
            best_index = (int)i;
        }
    }

    return best_index;
}
