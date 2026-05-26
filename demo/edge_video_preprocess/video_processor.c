/**
 * @file video_processor.c
 * @brief Video frame reader and motion detector implementation.
 */

#include "video_processor.h"
#include "demo_edge_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int video_meta_load(
    const char* meta_path,
    VideoMeta* out_meta,
    char* error_buffer,
    size_t error_buffer_size)
{
    FILE* file;
    char line[256];

    if (meta_path == NULL || out_meta == NULL) {
        demo_set_error(error_buffer, error_buffer_size, "invalid arguments");
        return -1;
    }

    (void)memset(out_meta, 0, sizeof(*out_meta));

#ifdef _WIN32
    if (fopen_s(&file, meta_path, "r") != 0) file = NULL;
#else
    file = fopen(meta_path, "r");
#endif
    if (file == NULL) {
        demo_set_error(error_buffer, error_buffer_size, "failed to open meta file");
        return -1;
    }

    while (fgets(line, (int)sizeof(line), file) != NULL) {
        char key[64];
        int value;
        size_t kpos;
        size_t lpos;

        /* Skip leading whitespace */
        lpos = 0U;
        while (line[lpos] == ' ' || line[lpos] == '\t') lpos++;

        /* Read key until whitespace or end */
        kpos = 0U;
        while (kpos < sizeof(key) - 1U && line[lpos] != '\0' &&
               line[lpos] != ' ' && line[lpos] != '\t' && line[lpos] != '\n') {
            key[kpos] = line[lpos];
            kpos++;
            lpos++;
        }
        key[kpos] = '\0';

        /* Skip whitespace after key */
        while (line[lpos] == ' ' || line[lpos] == '\t') lpos++;

        {
            size_t val_pos = lpos; /* save for float reparse */

            /* Parse integer value (works for most fields) */
            value = 0;
            {
                int sign = 1;
                if (line[lpos] == '-') { sign = -1; lpos++; }
                while (line[lpos] >= '0' && line[lpos] <= '9') {
                    value = value * 10 + (int)(line[lpos] - '0');
                    lpos++;
                }
                value *= sign;
            }

            if (kpos > 0U) {
                if (strcmp(key, "fps") == 0) {
                    /* Parse as float from saved position to preserve fractional fps */
                    size_t pos = val_pos;
                    double fval = 0.0;
                    double factor = 1.0;
                    double sign_f = 1.0;

                    if (line[pos] == '-') { sign_f = -1.0; pos++; }
                    while (line[pos] >= '0' && line[pos] <= '9') {
                        fval = fval * 10.0 + (double)(line[pos] - '0');
                        pos++;
                    }
                    if (line[pos] == '.') {
                        pos++;
                        while (line[pos] >= '0' && line[pos] <= '9') {
                            factor *= 0.1;
                            fval += factor * (double)(line[pos] - '0');
                            pos++;
                        }
                    }
                    out_meta->fps = sign_f * fval;
                } else if (strcmp(key, "frame_count") == 0) {
                    out_meta->frame_count = (size_t)(value > 0 ? value : 0);
                } else if (strcmp(key, "width") == 0) {
                    out_meta->width = (size_t)(value > 0 ? value : 0);
                } else if (strcmp(key, "height") == 0) {
                    out_meta->height = (size_t)(value > 0 ? value : 0);
                } else if (strcmp(key, "channels") == 0) {
                    out_meta->channels = (size_t)(value > 0 ? value : 0);
                }
            }
        }
    }

    fclose(file);

    if (out_meta->frame_count == 0U) {
        demo_set_error(error_buffer, error_buffer_size, "no frame_count in meta file");
        return -1;
    }

    demo_set_error(error_buffer, error_buffer_size, "");
    return 0;
}

int video_frame_read(FILE* file, float* frame) {
    if (file == NULL || frame == NULL) return VIDEO_FRAME_ERR;

    if (fread(frame, sizeof(float), VIDEO_FRAME_SIZE, file) != VIDEO_FRAME_SIZE) {
        return feof(file) ? VIDEO_FRAME_EOF : VIDEO_FRAME_ERR;
    }

    return VIDEO_FRAME_OK;
}

float video_frame_sad(const float* a, const float* b, size_t count) {
    float sum = 0.0f;
    size_t i;

    if (a == NULL || b == NULL) return -1.0f;

    for (i = 0U; i < count; ++i) {
        float diff = a[i] - b[i];
        sum += (diff >= 0.0f) ? diff : -diff;
    }

    return sum;
}

/* ========================================================================
 *  Adaptive motion threshold (EMA-based)
 * ======================================================================== */

void motion_threshold_init(MotionAdaptiveThreshold* t,
    float base, float factor, float alpha)
{
    if (t == NULL) return;
    t->rolling_mean   = 0.0f;
    t->rolling_std    = 0.0f;
    t->base_threshold = base;
    t->adapt_factor   = factor;
    t->ema_alpha      = alpha;
    t->initialized    = 0;
}

void motion_threshold_update(MotionAdaptiveThreshold* t, float sad) {
    if (t == NULL) return;

    if (!t->initialized) {
        /* First sample: seed the mean, zero the variance. */
        t->rolling_mean = sad;
        t->rolling_std  = 0.0f;
        t->initialized  = 1;
    } else {
        float delta = sad - t->rolling_mean;
        /* EMA update for mean */
        t->rolling_mean += t->ema_alpha * delta;
        /* EMA update for variance (absolute delta as proxy for std) */
        {
            float abs_delta = (delta >= 0.0f) ? delta : -delta;
            t->rolling_std += t->ema_alpha * (abs_delta - t->rolling_std);
        }
    }
}

float motion_threshold_get(const MotionAdaptiveThreshold* t) {
    float adaptive;

    if (t == NULL) return 0.0f;
    if (!t->initialized) return t->base_threshold;

    /* Threshold = base + factor * rolling_std.
     * This creates a margin above the running mean that's proportional
     * to the SAD variance, not the absolute SAD level. Sustained motion
     * (high mean, low std) is detected; noisy static (low mean, high std)
     * raises the bar. */
    adaptive = t->base_threshold + t->adapt_factor * t->rolling_std;
    return adaptive;
}

/* ========================================================================
 *  Temporal smoother (debouncing)
 * ======================================================================== */

void motion_smoother_init(MotionTemporalSmoother* s,
    int motion_frames, int static_frames)
{
    if (s == NULL) return;
    s->consecutive_motion   = 0;
    s->consecutive_static   = 0;
    s->motion_confirm_frames = motion_frames;
    s->static_confirm_frames = static_frames;
    s->is_active            = 0;  /* start in static state */
}

int motion_smoother_update(MotionTemporalSmoother* s, int detected) {
    if (s == NULL) return 0;

    if (detected) {
        s->consecutive_motion++;
        s->consecutive_static = 0;

        if (!s->is_active && s->consecutive_motion >= s->motion_confirm_frames) {
            s->is_active = 1;
        }
    } else {
        s->consecutive_static++;
        s->consecutive_motion = 0;

        if (s->is_active && s->consecutive_static >= s->static_confirm_frames) {
            s->is_active = 0;
        }
    }

    return s->is_active;
}
