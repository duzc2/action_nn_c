/**
 * @file infer_main.c
 * @brief Edge Video Preprocessing demo — surveillance video inference pipeline.
 *
 * Demonstrates edge-level intelligent preprocessing:
 *   1. Read pre-extracted 32x32x3 frames from video_frames.dat
 *   2. Compute SAD (sum of absolute differences) vs previous frame
 *   3. Adaptive threshold (EMA-based) + temporal smoothing (debouncing)
 *   4. If static: skip CNN inference (bandwidth saved)
 *   5. If motion: run CNN+MLP inference, log classification
 *   6. Print final report: frames filtered, bandwidth saved, class distribution
 */

#include "infer.h"
#include "weights_load.h"
#include "cifar10_dataset.h"
#include "video_processor.h"
#include "../demo_runtime_paths.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ── Motion detection tunables ── */
#define EVP_ADAPT_BASE              50.0f  /* Minimum (floor) SAD threshold */
#define EVP_ADAPT_FACTOR            2.0f   /* Multiplier on rolling mean */
#define EVP_EMA_ALPHA               0.1f   /* EMA smoothing factor */
#define EVP_MOTION_CONFIRM_FRAMES   3      /* Consecutive motion frames to trigger */
#define EVP_STATIC_CONFIRM_FRAMES   10     /* Consecutive static frames to go idle */
#define EVP_MOTION_COOLDOWN_FRAMES  0      /* Min frames between inference triggers (0=off) */

/* ── Logging ── */
#define EVP_VERBOSE_LOG 0  /* set to 1 for per-motion-frame classification log */

#define EVP_VIDEO_ROOT "../../../../demo/edge_video_preprocess/video_frames"

#ifdef _WIN32
#include <windows.h>
static double get_time_us(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000000.0 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000000.0 + (double)tv.tv_usec;
}
#endif

static const char* kMetaPath  = EVP_VIDEO_ROOT "/video_meta.txt";
static const char* kFramesPath = EVP_VIDEO_ROOT "/video_frames.dat";
static const char* kWeightsPath = "../data/weights.bin";

/* ── Check output validity ── */
static int output_is_valid(const float* output, size_t count) {
    size_t i;
    if (output == NULL) return 0;
    for (i = 0U; i < count; ++i) {
        if (isnan(output[i]) || isinf(output[i])) return 0;
    }
    return 1;
}

/* ── Print final pipeline report ── */
static void print_report(const VideoMeta* meta, const VideoStats* stats) {
    double total_gb;
    double filtered_gb;
    double saved_gb;
    double skip_pct;
    double motion_pct;
    size_t class_index;
    double avg_latency_us;
    double total_seconds;

    if (meta == NULL || stats == NULL) return;

    skip_pct = 0.0;
    motion_pct = 0.0;
    if (stats->frames_processed > 0U) {
        skip_pct   = (double)stats->frames_skipped  * 100.0 / (double)stats->frames_processed;
        motion_pct = (double)stats->motion_detected  * 100.0 / (double)stats->frames_processed;
    }

    /* Each "upload" is a 32x32x3 float32 frame = 3072 * 4 = 12288 bytes */
    total_gb    = (double)stats->frames_processed * 12288.0 / (1024.0 * 1024.0 * 1024.0);
    filtered_gb = (double)stats->motion_detected  * 12288.0 / (1024.0 * 1024.0 * 1024.0);
    saved_gb    = total_gb - filtered_gb;

    avg_latency_us = 0.0;
    if (stats->motion_detected > 0U) {
        avg_latency_us = stats->elapsed_us / (double)stats->motion_detected;
    }

    total_seconds = meta->frame_count > 0U
        ? (double)meta->frame_count / meta->fps
        : 0.0;

    printf("\n");
    printf("============================================================\n");
    printf("=== Edge Video Preprocessing v47 — Pipeline Report ===\n");
    printf("============================================================\n\n");

    printf("Video: %zu frames @ %.1f fps (%.0f sec equivalent)\n",
        meta->frame_count, meta->fps, total_seconds);
    printf("Resolution: %zux%zux%zu (RGB) | BnConvNet v47 (4 CNN + GAP + 1 MLP) -> CIFAR-10\n\n",
        meta->width, meta->height, meta->channels);

    printf("--- Motion Detection ---\n");
    printf("Frames with motion:   %5zu (%5.1f%%)  <- ran CNN inference\n",
        stats->motion_detected, motion_pct);
    printf("Frames skipped:       %5zu (%5.1f%%)  <- saved compute + bandwidth\n",
        stats->frames_skipped, skip_pct);
    printf("Total frames:         %5zu\n\n", stats->frames_processed);

    printf("--- Inference Performance ---\n");
    printf("Avg latency per inference: %.1f us\n", avg_latency_us);
    printf("Total inference time:      %.3f ms\n\n", stats->elapsed_us / 1000.0);

    printf("--- Classification Distribution (on motion frames) ---\n");
    for (class_index = 0U; class_index < CIFAR10_CLASS_COUNT; ++class_index) {
        double pct = 0.0;
        if (stats->motion_detected > 0U) {
            pct = (double)stats->class_counts[class_index]
                * 100.0 / (double)stats->motion_detected;
        }
        printf("  %-12s  %5zu (%5.1f%%)\n",
            cifar10_class_names[class_index],
            stats->class_counts[class_index],
            pct);
    }
    printf("\n");

    printf("--- Bandwidth Savings ---\n");
    printf("Raw upload (all frames):     %.2f MB\n",
        total_gb * 1024.0);
    printf("Edge filtered (motion only): %.2f MB\n",
        filtered_gb * 1024.0);
    printf("Bandwidth saved:             %.2f MB (%5.1f%%)\n",
        saved_gb * 1024.0, skip_pct);
    printf("\n============================================================\n");
}

int main(void) {
    FILE* frames_file;
    VideoMeta meta;
    VideoStats stats;
    MotionAdaptiveThreshold adap_thresh;
    MotionTemporalSmoother smoother;
    void* infer_ctx;
    char error_buffer[256];
    float frame_buf_a[VIDEO_FRAME_SIZE];
    float frame_buf_b[VIDEO_FRAME_SIZE];
    float* prev_frame;  /* pointer to whichever buffer holds the previous frame */
    float* curr_frame;  /* pointer to whichever buffer holds the current frame */
    float* tmp_frame;   /* temporary for pointer swap */
    size_t frame_index;
    float output[CIFAR10_CLASS_COUNT];
    double t_start_us;
    double t_end_us;
    int has_prev;
    size_t cooldown_counter;

    (void)memset(&meta, 0, sizeof(meta));
    (void)memset(&stats, 0, sizeof(stats));
    error_buffer[0] = '\0';

    /* Initialize motion detection subsystems */
    motion_threshold_init(&adap_thresh, EVP_ADAPT_BASE, EVP_ADAPT_FACTOR, EVP_EMA_ALPHA);
    motion_smoother_init(&smoother, EVP_MOTION_CONFIRM_FRAMES, EVP_STATIC_CONFIRM_FRAMES);

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Edge Video Preprocessing v47 — Inference Pipeline ===\n\n");

    /* ── Load video metadata ── */
    if (video_meta_load(kMetaPath, &meta, error_buffer, sizeof(error_buffer)) != 0) {
        fprintf(stderr, "Failed to load video meta: %s\n", error_buffer);
        fprintf(stderr, "Path: %s\n", kMetaPath);
        fprintf(stderr, "Run data_prep.py first to generate video frames.\n");
        return 1;
    }

    printf("Loaded video metadata: %zu frames @ %.1f fps, %zux%zux%zu\n\n",
        meta.frame_count, meta.fps, meta.width, meta.height, meta.channels);

    /* ── Open video frames file ── */
#ifdef _WIN32
    if (fopen_s(&frames_file, kFramesPath, "rb") != 0) frames_file = NULL;
#else
    frames_file = fopen(kFramesPath, "rb");
#endif
    if (frames_file == NULL) {
        fprintf(stderr, "Failed to open video frames: %s\n", kFramesPath);
        return 1;
    }

    /* ── Create inference context and load weights ── */
    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        fclose(frames_file);
        return 1;
    }

    if (weights_load_from_file(infer_ctx, kWeightsPath) != 0) {
        fprintf(stderr, "Failed to load weights from %s\n", kWeightsPath);
        fprintf(stderr, "Run the training phase first.\n");
        infer_destroy(infer_ctx);
        fclose(frames_file);
        return 1;
    }

    printf("Weights loaded. Running motion-detection + inference pipeline ...\n\n");

    /* ── Frame-by-frame processing loop ── */
    has_prev = 0;
    prev_frame = frame_buf_a;
    curr_frame = frame_buf_b;
    cooldown_counter = 0U;

    for (frame_index = 0U; frame_index < meta.frame_count; ++frame_index) {
        float sad;
        int raw_motion;
        int debounced_motion;

        if (video_frame_read(frames_file, curr_frame) != 0) {
            break;
        }
        stats.frames_processed++;

        if (!has_prev) {
            /* First frame: always run inference */
            has_prev = 1;

            /* Feed SAD=0 to adaptive threshold (no frame comparison possible) */
            motion_threshold_update(&adap_thresh, 0.0f);

            t_start_us = get_time_us();
            if (infer_auto_run(infer_ctx, curr_frame, output) == 0
                && output_is_valid(output, CIFAR10_CLASS_COUNT)) {
                t_end_us = get_time_us();
                stats.elapsed_us += (t_end_us - t_start_us);
                stats.motion_detected++;

                {
                    int pred = cifar10_argmax(output, CIFAR10_CLASS_COUNT);
                    if (pred >= 0 && pred < (int)CIFAR10_CLASS_COUNT) {
                        stats.class_counts[pred]++;
#if EVP_VERBOSE_LOG
                        printf("  [FRAME %5zu] SAD=N/A (first)  class=%s  conf=%.3f\n",
                            frame_index + 1U,
                            cifar10_class_names[pred],
                            (double)output[pred]);
#endif
                    }
                }
            } else {
                t_end_us = get_time_us();
                stats.elapsed_us += (t_end_us - t_start_us);
            }
        } else {
            /* Compute SAD with previous frame */
            sad = video_frame_sad(prev_frame, curr_frame, VIDEO_FRAME_SIZE);

            /* Feed into adaptive threshold */
            motion_threshold_update(&adap_thresh, sad);

            /* Raw motion detection */
            {
                float threshold = motion_threshold_get(&adap_thresh);
                raw_motion = (sad > threshold) ? 1 : 0;
            }

            /* Apply temporal smoothing */
            debounced_motion = motion_smoother_update(&smoother, raw_motion);

            if (debounced_motion) {
                /* Debounced motion: check cooldown before running inference */
                if (EVP_MOTION_COOLDOWN_FRAMES == 0 || cooldown_counter == 0U) {
                    t_start_us = get_time_us();
                    if (infer_auto_run(infer_ctx, curr_frame, output) == 0
                        && output_is_valid(output, CIFAR10_CLASS_COUNT)) {
                        t_end_us = get_time_us();
                        stats.elapsed_us += (t_end_us - t_start_us);
                        stats.motion_detected++;

                        {
                            int pred = cifar10_argmax(output, CIFAR10_CLASS_COUNT);
                            if (pred >= 0 && pred < (int)CIFAR10_CLASS_COUNT) {
                                stats.class_counts[pred]++;
#if EVP_VERBOSE_LOG
                                printf("  [FRAME %5zu] SAD=%.1f  thresh=%.1f  class=%s  conf=%.3f\n",
                                    frame_index + 1U,
                                    (double)sad,
                                    (double)motion_threshold_get(&adap_thresh),
                                    cifar10_class_names[pred],
                                    (double)output[pred]);
#endif
                            }
                        }
                    } else {
                        t_end_us = get_time_us();
                        stats.elapsed_us += (t_end_us - t_start_us);
                    }

                    cooldown_counter = (size_t)EVP_MOTION_COOLDOWN_FRAMES;
                } else {
                    /* In cooldown: skip inference */
                    stats.frames_skipped++;
                    if (cooldown_counter > 0U) cooldown_counter--;
                }
            } else {
                /* Static frame: skip inference */
                stats.frames_skipped++;
            }
        }

        /* Swap frame pointers instead of memcpy */
        tmp_frame = prev_frame;
        prev_frame = curr_frame;
        curr_frame = tmp_frame;

        /* Progress indicator */
        if ((frame_index + 1U) % 500U == 0U) {
            printf("  Processed %zu / %zu frames (motion: %zu, skipped: %zu)\n",
                frame_index + 1U, meta.frame_count,
                stats.motion_detected, stats.frames_skipped);
        }
    }

    fclose(frames_file);
    infer_destroy(infer_ctx);

    /* ── Print final report ── */
    print_report(&meta, &stats);

    return 0;
}
