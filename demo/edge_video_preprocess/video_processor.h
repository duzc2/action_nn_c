/**
 * @file video_processor.h
 * @brief Video frame reader and motion detector for edge preprocessing demo.
 *
 * Reads pre-extracted 32x32x3 raw float32 frames from a binary file
 * and computes SAD (sum of absolute differences) with the previous frame
 * to detect motion. Frames below the threshold are skipped — this
 * simulates edge-node bandwidth-saving logic.
 */

#ifndef DEMO_EDGE_VIDEO_PREPROCESS_VIDEO_PROCESSOR_H
#define DEMO_EDGE_VIDEO_PREPROCESS_VIDEO_PROCESSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define VIDEO_FRAME_WIDTH  32U
#define VIDEO_FRAME_HEIGHT 32U
#define VIDEO_FRAME_CHANNELS 3U
#define VIDEO_FRAME_SIZE (VIDEO_FRAME_WIDTH * VIDEO_FRAME_HEIGHT * VIDEO_FRAME_CHANNELS)

/* video_frame_read return codes */
#define VIDEO_FRAME_OK     0   /**< Frame read successfully */
#define VIDEO_FRAME_ERR   -1   /**< I/O error */
#define VIDEO_FRAME_EOF   -2   /**< End of file reached */

/**
 * @brief Video frame metadata.
 */
typedef struct {
    size_t frame_count;   /**< Total number of frames. */
    double fps;           /**< Frames per second. */
    size_t width;         /**< Frame width. */
    size_t height;        /**< Frame height. */
    size_t channels;      /**< Color channels (3 for RGB). */
} VideoMeta;

/**
 * @brief Motion detection statistics accumulated during processing.
 */
typedef struct {
    size_t frames_processed;  /**< Total frames processed. */
    size_t motion_detected;   /**< Frames where SAD > threshold. */
    size_t frames_skipped;    /**< Frames where SAD <= threshold. */
    double elapsed_us;        /**< Total inference time in microseconds. */
    size_t class_counts[10];  /**< Per-class classification counts. */
} VideoStats;

/**
 * @brief Load video metadata from video_meta.txt.
 *
 * @param meta_path Path to metadata file.
 * @param out_meta  Output metadata struct.
 * @param error_buffer Error message buffer.
 * @param error_buffer_size Size of error buffer.
 * @return 0 on success, negative on failure.
 */
int video_meta_load(
    const char* meta_path,
    VideoMeta* out_meta,
    char* error_buffer,
    size_t error_buffer_size);

/**
 * @brief Read one frame from the raw binary file.
 *
 * @param file  Open file handle positioned at the start of the next frame.
 * @param frame Output buffer of size VIDEO_FRAME_SIZE.
 * @return 0 on success, negative on EOF or error.
 */
int video_frame_read(FILE* file, float* frame);

/**
 * @brief Compute SAD (sum of absolute differences) between two frames.
 *
 * @param a First frame.
 * @param b Second frame.
 * @param count Number of pixels (VIDEO_FRAME_SIZE).
 * @return SAD value.
 */
float video_frame_sad(const float* a, const float* b, size_t count);

#endif /* DEMO_EDGE_VIDEO_PREPROCESS_VIDEO_PROCESSOR_H */
