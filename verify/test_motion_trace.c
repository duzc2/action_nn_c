/**
 * @file test_motion_trace.c
 * @brief Visual frame-by-frame trace of the motion detection pipeline.
 *
 * Generates a 30-on/50-off burst pattern using sequential frame generation,
 * runs the adaptive threshold + temporal smoother, and prints a per-frame
 * trace showing SAD, threshold, raw detection, and debounced state.
 *
 * Build: cmake --build build/verify --target test_motion_trace
 * Run:   build/verify/test_motion_trace.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "video_processor.h"

/* ── RNG ──────────────────────────────────────────────────────────────── */
static uint32_t g_rng = 99U;
static uint32_t rng_next(void) {
    g_rng = g_rng * 1103515245U + 12345U;
    return g_rng;
}
static float rng_sym(float a) {
    uint32_t r = rng_next();
    float u = (float)(r & 0xFFFFFFU) / (float)0x1000000U;
    return -a + 2.0f * a * u;
}
static void frame_copy(float* d, const float* s, size_t n) {
    (void)memcpy(d, s, n * sizeof(float));
}
static void frame_base(float* f, size_t n) {
    size_t i;
    for (i = 0U; i < n; ++i)
        f[i] = (float)(rng_next() & 0xFFFFFFU) / (float)0x1000000U;
}

/* ── Generate frames sequentially ─────────────────────────────────────── */
static void gen_frames(float* frames, int* truth, size_t n,
    int motion_len, int static_len, float motion_amp, float static_amp)
{
    float base[VIDEO_FRAME_SIZE];
    size_t i, j;
    const size_t period = (size_t)(motion_len + static_len);

    frame_base(base, VIDEO_FRAME_SIZE);

    for (i = 0U; i < n; ++i) {
        float* cur = &frames[i * VIDEO_FRAME_SIZE];
        size_t p = i % period;
        int is_mot = (p < (size_t)motion_len) ? 1 : 0;

        if (i == 0U) {
            frame_copy(cur, base, VIDEO_FRAME_SIZE);
            for (j = 0U; j < VIDEO_FRAME_SIZE; ++j)
                cur[j] += rng_sym(is_mot ? motion_amp : static_amp);
        } else {
            const float* prev = &frames[(i - 1U) * VIDEO_FRAME_SIZE];
            float amp = is_mot ? motion_amp : static_amp;
            frame_copy(cur, prev, VIDEO_FRAME_SIZE);
            for (j = 0U; j < VIDEO_FRAME_SIZE; ++j)
                cur[j] += rng_sym(amp);
        }
        truth[i] = is_mot;
    }
}

/* ── Trace runner ─────────────────────────────────────────────────────── */
static void run_trace(const float* frames, const int* truth, size_t n,
    MotionAdaptiveThreshold* adap, MotionTemporalSmoother* s)
{
    float prev[VIDEO_FRAME_SIZE];
    int hp = 0;
    size_t i;

    /* Column headers */
    printf("%5s %7s %6s %8s %7s %3s %3s %4s| %5s %5s\n",
        "Frame", "Truth", "SAD", "Thresh", "EMA_std", "raw", "deb", "act",
        "TP", "FP");
    printf("%5s %7s %6s %8s %7s %3s %3s %4s| %5s %5s\n",
        "-----", "-----", "---", "------", "------", "---", "---", "----",
        "-----", "-----");

    for (i = 0U; i < n; ++i) {
        const float* cur = &frames[i * VIDEO_FRAME_SIZE];
        float sad = 0.0f;
        float thr;
        int raw = 0;
        int deb;

        if (!hp) {
            hp = 1;
            motion_threshold_update(adap, 0.0f);
            deb = s->is_active;
        } else {
            sad = video_frame_sad(prev, cur, VIDEO_FRAME_SIZE);
            motion_threshold_update(adap, sad);
            thr = motion_threshold_get(adap);
            raw = (sad > thr) ? 1 : 0;
            deb = motion_smoother_update(s, raw);
        }

        /* Per-frame output */
        if (i == 0U) {
            printf("%5zu %7s %6s %8s %7s %3s %3d %4s| %5s %5s\n",
                i,
                truth[i] ? "MOTION" : "static",
                "---", "---", "---", "---",
                deb,
                s->is_active ? "ON " : "off",
                "", "");
        } else {
            const char* stat = s->is_active ? "ON " : "off";
            const char* tm = "";
            const char* fp = "";

            if (truth[i] && deb)  tm = "  TP";
            if (!truth[i] && deb) fp = "  FP";

            printf("%5zu %7s %6.0f %8.1f %7.1f %3d %3d %4s|%s%s\n",
                i,
                truth[i] ? "MOTION" : "static",
                (double)sad,
                (double)thr,
                (double)adap->rolling_std,
                raw, deb, stat,
                tm, fp);
        }

        frame_copy(prev, cur, VIDEO_FRAME_SIZE);
    }
}

/* ── Main ─────────────────────────────────────────────────────────────── */
int main(void) {
    const size_t n = 160U;          /* 2 full cycles of 30 + 50 */
    const int    motion_len = 30;
    const int    static_len = 50;
    const float  motion_amp = 0.30f;
    const float  static_amp = 0.002f;

    MotionAdaptiveThreshold adap;
    MotionTemporalSmoother s;
    float* frames;
    int* truth;

    frames = (float*)calloc(n * VIDEO_FRAME_SIZE, sizeof(float));
    truth  = (int*)calloc(n, sizeof(int));

    /* Generate frames */
    gen_frames(frames, truth, n, motion_len, static_len,
        motion_amp, static_amp);

    printf("\n");
    printf("=== Motion Detection Pipeline — Frame-by-Frame Trace ===\n");
    printf("Scenario: %d-on / %d-off burst, 2 cycles (%zu frames)\n",
        motion_len, static_len, n);
    printf("Motion step: +/-%.2f per pixel -> SAD ~%.0f\n",
        (double)motion_amp,
        (double)(VIDEO_FRAME_SIZE * motion_amp * 0.5));
    printf("Static step: +/-%.3f per pixel -> SAD ~%.0f\n",
        (double)static_amp,
        (double)(VIDEO_FRAME_SIZE * static_amp * 0.5));
    printf("Threshold: base=50 + factor=2 * EMA_std\n");
    printf("Smoother: 3 confirm / 10 idle\n\n");
    printf("Legend: raw=SAD>thr?1:0  deb=smoothed output  act=smoother state\n");
    printf("        TP=true positive  FP=false positive\n\n");

    /* Init and run */
    motion_threshold_init(&adap, 50.0f, 2.0f, 0.1f);
    motion_smoother_init(&s, 3, 10);
    run_trace(frames, truth, n, &adap, &s);

    /* Summary */
    {
        size_t tp = 0, tn = 0, fp = 0, fn = 0;
        float prev2[VIDEO_FRAME_SIZE];
        int hp2 = 0;
        MotionAdaptiveThreshold adap2;
        MotionTemporalSmoother s2;
        size_t i;

        motion_threshold_init(&adap2, 50.0f, 2.0f, 0.1f);
        motion_smoother_init(&s2, 3, 10);

        for (i = 0U; i < n; ++i) {
            const float* cur = &frames[i * VIDEO_FRAME_SIZE];
            int deb;
            if (!hp2) {
                hp2 = 1;
                motion_threshold_update(&adap2, 0.0f);
                deb = s2.is_active;
            } else {
                float sad = video_frame_sad(prev2, cur, VIDEO_FRAME_SIZE);
                motion_threshold_update(&adap2, sad);
                float thr = motion_threshold_get(&adap2);
                deb = motion_smoother_update(&s2, (sad > thr) ? 1 : 0);
            }
            if (truth[i] && deb)       tp++;
            else if (!truth[i] && !deb) tn++;
            else if (!truth[i] && deb)  fp++;
            else                        fn++;
            frame_copy(prev2, cur, VIDEO_FRAME_SIZE);
        }

        double prec = tp + fp ? (double)tp / (tp + fp) : 0.0;
        double rec  = tp + fn ? (double)tp / (tp + fn) : 0.0;
        double f1   = prec + rec > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0;

        printf("\n=== Summary ===\n");
        printf("TP=%zu  TN=%zu  FP=%zu  FN=%zu\n", tp, tn, fp, fn);
        printf("Precision=%.1f%%  Recall=%.1f%%  F1=%.3f\n",
            prec * 100, rec * 100, f1);
    }

    free(frames);
    free(truth);
    return 0;
}
