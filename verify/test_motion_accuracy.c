/**
 * @file test_motion_accuracy.c
 * @brief Motion detection accuracy benchmark — precision, recall, F1.
 *
 * Tests the full pipeline (SAD + adaptive threshold + temporal smoother)
 * against synthetic frame sequences with ground-truth labels.
 *
 * Frames are generated SEQUENTIALLY: frame[i] = frame[i-1] + step[i].
 * Motion frames get large steps (creating high SAD); static frames get
 * tiny steps (creating low SAD).  The motion->static transition has a
 * tiny step, so there is no false SAD spike at motion boundaries.
 *
 * All motion bursts are >= 20 consecutive frames to satisfy the temporal
 * smoother's confirm/idle requirements (default: 3 motion / 10 static).
 *
 * Build: cmake --build build/verify --target test_motion_accuracy
 * Run:   build/verify/test_motion_accuracy.exe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "video_processor.h"

/* ========================================================================
 *  Helpers
 * ======================================================================== */

static int g_failures = 0;
static int g_test_num = 0;

#define TEST_ASSERT(cond, msg) do { \
    g_test_num++; \
    if (!(cond)) { printf("  FAIL test %d: %s\n", g_test_num, msg); g_failures++; } \
    else { printf("  OK   test %d: %s\n", g_test_num, msg); } \
} while (0)

#define TEST_SKIP(cond, msg) do { \
    g_test_num++; \
    printf("  SKIP test %d: %s\n", g_test_num, msg); \
} while (0)

static uint32_t g_rng = 42U;
static uint32_t rng_next(void) { g_rng = g_rng * 1103515245U + 12345U; return g_rng; }

/* Uniform random in [lo, hi) */
static float rng_uniform(float lo, float hi) {
    uint32_t r = rng_next();
    return lo + (float)(r & 0xFFFFFFU) / (float)0x1000000U * (hi - lo);
}

/* Symmetric uniform in [-a, +a] */
static float rng_sym(float a) { return rng_uniform(-a, a); }

static void frame_copy(float* d, const float* s, size_t n) { (void)memcpy(d, s, n * sizeof(float)); }

/**
 * @brief Generate a random base frame for the shared background.
 */
static void frame_base(float* f, size_t n) {
    size_t i;
    for (i = 0U; i < n; ++i) f[i] = rng_uniform(0.0f, 1.0f);
}

/**
 * @brief Generate frames SEQUENTIALLY.
 *
 * frame[0] = base + initial_noise
 * For i > 0: frame[i] = frame[i-1] + step[i]
 *   - motion frames: step = uniform[-motion_amp, +motion_amp]
 *   - static frames: step = uniform[-static_amp, +static_amp]
 *
 * This ensures:
 *   motion->motion SAD  = sum|large step|  -> consistently high
 *   static->static SAD  = sum|tiny step|   -> consistently low
 *   motion->static transition SAD = sum|tiny step| -> low (no false spike)
 *
 * @param frames      Output buffer of size n * VIDEO_FRAME_SIZE.
 * @param truth       Output ground truth labels (size n).
 * @param n           Number of frames.
 * @param is_motion   Callback: returns 1 if frame index i is a motion frame.
 * @param motion_amp  Per-pixel step amplitude during motion (e.g. 0.30f).
 * @param static_amp  Per-pixel step amplitude during static (e.g. 0.002f).
 */
static void generate_sequential(float* frames, int* truth, size_t n,
    int (*is_motion)(size_t i),
    float motion_amp, float static_amp)
{
    float base[VIDEO_FRAME_SIZE];
    size_t i, j;

    frame_base(base, VIDEO_FRAME_SIZE);

    for (i = 0U; i < n; ++i) {
        float* cur = &frames[i * VIDEO_FRAME_SIZE];
        if (i == 0U) {
            /* First frame: base + tiny noise (not motion content) */
            frame_copy(cur, base, VIDEO_FRAME_SIZE);
            for (j = 0U; j < VIDEO_FRAME_SIZE; ++j)
                cur[j] += rng_sym(static_amp);
            truth[i] = is_motion(i);
        } else {
            const float* prev = &frames[(i - 1U) * VIDEO_FRAME_SIZE];
            float amp = is_motion(i) ? motion_amp : static_amp;
            frame_copy(cur, prev, VIDEO_FRAME_SIZE);
            for (j = 0U; j < VIDEO_FRAME_SIZE; ++j)
                cur[j] += rng_sym(amp);
            truth[i] = is_motion(i);
        }
    }
}

/* ========================================================================
 *  Accuracy metrics
 * ======================================================================== */

typedef struct { size_t tp, tn, fp, fn, total; } Acc;

static void acc_init(Acc* a) { (void)memset(a, 0, sizeof(*a)); }
static void acc_add(Acc* a, int pred, int truth) {
    a->total++;
    if (pred && truth)       a->tp++;
    else if (!pred && !truth) a->tn++;
    else if (pred && !truth)  a->fp++;
    else                      a->fn++;
}
static double acc_prec(const Acc* a)  { size_t d = a->tp + a->fp; return d ? (double)a->tp / d : 0.0; }
static double acc_recall(const Acc* a){ size_t d = a->tp + a->fn; return d ? (double)a->tp / d : 0.0; }
static double acc_spec(const Acc* a)  { size_t d = a->tn + a->fp; return d ? (double)a->tn / d : 0.0; }
static double acc_f1(const Acc* a)    { double p = acc_prec(a), r = acc_recall(a); return (p + r > 0) ? 2.0 * p * r / (p + r) : 0.0; }

static void acc_print(const Acc* a, const char* label, double f1_min,
    double fpr_max, double fnr_max)
{
    double p = acc_prec(a), r = acc_recall(a), sp = acc_spec(a), f1 = acc_f1(a);
    double fpr = 1.0 - sp, fnr = 1.0 - r;
    size_t total_motion = a->tp + a->fn;

    printf("\n  --- %s ---\n", label);
    printf("  Total:%5zu  TP:%5zu  TN:%5zu  FP:%5zu  FN:%5zu\n",
        a->total, a->tp, a->tn, a->fp, a->fn);
    printf("  Prec=%5.1f%%  Rec=%5.1f%%  Spec=%5.1f%%  F1=%5.3f\n",
        p * 100, r * 100, sp * 100, f1);
    printf("  FPR=%6.3f  FNR=%6.3f\n", fpr, fnr);

    if (total_motion == 0U) {
        /* All-static scenario: only check FPR */
        TEST_SKIP(1, "No motion in truth — skipping F1/recall/FNR");
        TEST_ASSERT(fpr <= fpr_max, "FPR within limit (all-static scenario)");
        TEST_ASSERT(a->fp == 0U, "Zero false positives in all-static");
    } else {
        TEST_ASSERT(f1 >= f1_min, "F1 >= target");
        TEST_ASSERT(fpr <= fpr_max, "FPR <= limit");
        TEST_ASSERT(fnr <= fnr_max, "FNR <= limit");
        TEST_ASSERT(a->tp > 0U, "At least some motion detected");
    }
}

/* ========================================================================
 *  Pipeline runner
 * ======================================================================== */

static void run(const float* frames, const int* truth, size_t n,
    MotionAdaptiveThreshold* adap, MotionTemporalSmoother* s, Acc* acc)
{
    size_t i;
    float prev[VIDEO_FRAME_SIZE];
    int hp = 0;

    acc_init(acc);
    for (i = 0U; i < n; ++i) {
        const float* cur = &frames[i * VIDEO_FRAME_SIZE];
        if (!hp) {
            hp = 1;
            /* Seed adaptive threshold with SAD=0 (no previous frame). */
            motion_threshold_update(adap, 0.0f);
            acc_add(acc, s->is_active, truth[i]);
        } else {
            float sad = video_frame_sad(prev, cur, VIDEO_FRAME_SIZE);
            motion_threshold_update(adap, sad);
            float thr = motion_threshold_get(adap);
            int raw = (sad > thr) ? 1 : 0;
            int deb = motion_smoother_update(s, raw);
            acc_add(acc, deb, truth[i]);
        }
        frame_copy(prev, cur, VIDEO_FRAME_SIZE);
    }
}

/* ========================================================================
 *  Scenario A: All Static (100 frames).
 *
 *  All frames: base + tiny noise (±0.002).  SAD ≈ 3.
 *  base_threshold=50 >> 3.  Smoother requires 3 motion to trigger.
 *  Expect: 0 detections, FPR = 0.
 * ======================================================================== */

static int is_static(size_t i) { (void)i; return 0; }

static void test_a_all_static(void) {
    MotionAdaptiveThreshold adap;
    MotionTemporalSmoother s;
    Acc acc;
    float* frames;
    int* truth;
    const size_t n = 100U;

    frames = (float*)calloc(n * VIDEO_FRAME_SIZE, sizeof(float));
    truth  = (int*)calloc(n, sizeof(int));

    g_rng = 42U;
    generate_sequential(frames, truth, n, is_static, 0.0f, 0.002f);

    motion_threshold_init(&adap, 50.0f, 2.0f, 0.1f);
    motion_smoother_init(&s, 3, 10);
    run(frames, truth, n, &adap, &s, &acc);
    acc_print(&acc, "A: All Static (100 frames, SAD~3)", 0.0, 0.05, 0.0);
    free(frames); free(truth);
}

/* ========================================================================
 *  Scenario B: Long Bursts — 30 motion / 50 static, 160 frames (2 bursts).
 *
 *  Motion step: ±0.30 → SAD ≈ 460 per frame.
 *  Static step: ±0.002 → SAD ≈ 3.
 *  threshold base=50, factor=2 → max threshold ~428 < 460 → always detected.
 *  Smoother: 3 confirm / 10 idle.
 *  Per burst: TP ≈ 27 (30-3), FN ≈ 3, FP ≈ 10, F1 ≈ 0.80+.
 * ======================================================================== */

static int is_burst_b(size_t i) {
    size_t p = i % 80U;
    return (p < 30U) ? 1 : 0;
}

static void test_b_bursts(void) {
    MotionAdaptiveThreshold adap;
    MotionTemporalSmoother s;
    Acc acc;
    float* frames;
    int* truth;
    const size_t n = 160U;  /* 2 bursts of 30-on/50-off */

    frames = (float*)calloc(n * VIDEO_FRAME_SIZE, sizeof(float));
    truth  = (int*)calloc(n, sizeof(int));

    g_rng = 99U;
    generate_sequential(frames, truth, n, is_burst_b, 0.30f, 0.002f);

    motion_threshold_init(&adap, 50.0f, 2.0f, 0.1f);
    motion_smoother_init(&s, 3, 10);
    run(frames, truth, n, &adap, &s, &acc);
    acc_print(&acc, "B: Long Bursts (30-on/50-off, SAD~460, 2 bursts)", 0.78, 0.22, 0.25);
    free(frames); free(truth);
}

/* ========================================================================
 *  Scenario C: Frequent Long Bursts — 20-on/30-off, 200 frames (4 bursts).
 *
 *  Same step amplitudes as B.  3 confirm / 10 idle.
 *  Per 50-frame burst: 20 motion, 30 static.
 *  TP ≈ 17, FN ≈ 3, FP ≈ 10, TN ≈ 20 → F1 ≈ 0.72.
 * ======================================================================== */

static int is_burst_c(size_t i) {
    size_t p = i % 50U;
    return (p < 20U) ? 1 : 0;
}

static void test_c_frequent(void) {
    MotionAdaptiveThreshold adap;
    MotionTemporalSmoother s;
    Acc acc;
    float* frames;
    int* truth;
    const size_t n = 200U;  /* 4 bursts of 20-on/30-off */

    frames = (float*)calloc(n * VIDEO_FRAME_SIZE, sizeof(float));
    truth  = (int*)calloc(n, sizeof(int));

    g_rng = 123U;
    generate_sequential(frames, truth, n, is_burst_c, 0.30f, 0.002f);

    motion_threshold_init(&adap, 50.0f, 2.0f, 0.1f);
    motion_smoother_init(&s, 3, 10);
    run(frames, truth, n, &adap, &s, &acc);
    acc_print(&acc, "C: Frequent Bursts (20-on/30-off, SAD~460, 4 bursts)", 0.70, 0.35, 0.25);
    free(frames); free(truth);
}

/* ========================================================================
 *  Scenario D: Noisy Static — 100 frames, step ±0.005 → SAD ≈ 8-12.
 *
 *  base=50, factor=2, alpha=0.1.  Over time EM_std≈3, threshold≈56.
 *  SAD=12 < 56 → no false detections.  FPR = 0.
 * ======================================================================== */

static void test_d_noisy(void) {
    MotionAdaptiveThreshold adap;
    MotionTemporalSmoother s;
    Acc acc;
    float* frames;
    int* truth;
    const size_t n = 100U;

    frames = (float*)calloc(n * VIDEO_FRAME_SIZE, sizeof(float));
    truth  = (int*)calloc(n, sizeof(int));

    g_rng = 777U;
    generate_sequential(frames, truth, n, is_static, 0.0f, 0.005f);

    motion_threshold_init(&adap, 50.0f, 2.0f, 0.1f);
    motion_smoother_init(&s, 3, 10);
    run(frames, truth, n, &adap, &s, &acc);
    acc_print(&acc, "D: Noisy Static (step ±0.005, SAD~10, base=50)", 0.0, 0.05, 0.0);
    free(frames); free(truth);
}

/* ========================================================================
 *  Scenario E: Extended Motion — 30 static, 60 motion, 30 static (120 frames).
 *
 *  Motion step: ±0.25 → SAD ≈ 384.  3 confirm / 10 idle.
 *  The long motion period dilutes confirm/idle overhead.
 *  TP ≈ 57 (60-3), FN ≈ 3, FP ≈ 10 → F1 ≈ 0.90+.
 * ======================================================================== */

static int is_extended_motion(size_t i) {
    return (i >= 30U && i < 90U) ? 1 : 0;
}

static void test_e_extended(void) {
    MotionAdaptiveThreshold adap;
    MotionTemporalSmoother s;
    Acc acc;
    float* frames;
    int* truth;
    const size_t n = 120U;

    frames = (float*)calloc(n * VIDEO_FRAME_SIZE, sizeof(float));
    truth  = (int*)calloc(n, sizeof(int));

    g_rng = 444U;
    generate_sequential(frames, truth, n, is_extended_motion, 0.25f, 0.002f);

    motion_threshold_init(&adap, 50.0f, 2.0f, 0.1f);
    motion_smoother_init(&s, 3, 10);
    run(frames, truth, n, &adap, &s, &acc);
    acc_print(&acc, "E: Extended Motion (30s/60m/30s, SAD~384)", 0.88, 0.22, 0.12);
    free(frames); free(truth);
}

/* ========================================================================
 *  Scenario F: Gradual Ramp — step amplitude grows from 0.001 to 0.30
 *  over 300 frames.  Motion truth flips at amp > 0.04 (SAD ≈ 60 > base=50).
 *
 *  This tests the adaptive threshold's ability to handle slowly
 *  increasing noise levels.
 * ======================================================================== */

static int is_ramp_motion(size_t i) {
    float frac = (float)i / 300.0f;
    float amp = 0.001f + frac * 0.299f;
    return (amp > 0.04f) ? 1 : 0;
}

static void test_f_ramp(void) {
    MotionAdaptiveThreshold adap;
    MotionTemporalSmoother s;
    Acc acc;
    float* frames;
    int* truth;
    const size_t n = 300U;
    float base[VIDEO_FRAME_SIZE];
    size_t i, j;

    frames = (float*)calloc(n * VIDEO_FRAME_SIZE, sizeof(float));
    truth  = (int*)calloc(n, sizeof(int));

    g_rng = 555U;
    frame_base(base, VIDEO_FRAME_SIZE);

    /* Custom generation: each frame's step amplitude varies with the ramp. */
    for (i = 0U; i < n; ++i) {
        float frac = (float)i / (float)n;
        float amp = 0.001f + frac * 0.299f;
        float* cur = &frames[i * VIDEO_FRAME_SIZE];

        if (i == 0U) {
            frame_copy(cur, base, VIDEO_FRAME_SIZE);
            for (j = 0U; j < VIDEO_FRAME_SIZE; ++j) cur[j] += rng_sym(amp);
        } else {
            const float* prev = &frames[(i - 1U) * VIDEO_FRAME_SIZE];
            frame_copy(cur, prev, VIDEO_FRAME_SIZE);
            for (j = 0U; j < VIDEO_FRAME_SIZE; ++j) cur[j] += rng_sym(amp);
        }
        truth[i] = (amp > 0.04f) ? 1 : 0;
    }

    motion_threshold_init(&adap, 50.0f, 2.0f, 0.1f);
    motion_smoother_init(&s, 3, 10);
    run(frames, truth, n, &adap, &s, &acc);
    acc_print(&acc, "F: Gradual Ramp (amp 0.001->0.30, SAD 0->460)", 0.85, 0.20, 0.25);
    free(frames); free(truth);
}

/* ========================================================================
 *  Scenario G: Threshold Sweep — burst pattern, sweep base_threshold
 *  from 10 to 200, find best F1.
 *
 *  20-on/30-off bursts, 200 frames (4 bursts).
 * ======================================================================== */

static int is_sweep_burst(size_t i) {
    size_t p = i % 50U;
    return (p < 20U) ? 1 : 0;
}

static void test_g_sweep(void) {
    const size_t n = 200U;
    float* frames;
    int* truth;
    float best_f1 = 0.0f;
    float best_thresh = 0.0f;
    size_t i;

    frames = (float*)calloc(n * VIDEO_FRAME_SIZE, sizeof(float));
    truth  = (int*)calloc(n, sizeof(int));

    g_rng = 333U;
    generate_sequential(frames, truth, n, is_sweep_burst, 0.30f, 0.002f);

    printf("\n  --- G: Threshold Sweep (20-on/30-off, SAD~460) ---\n");
    printf("  %8s  %7s  %7s  %7s  %7s\n", "Base", "Prec%", "Rec%", "F1", "FPR%");

    {
        float bv;
        for (bv = 10.0f; bv <= 200.0f; bv += 10.0f) {
            MotionAdaptiveThreshold adap;
            MotionTemporalSmoother s;
            Acc acc;
            float prev[VIDEO_FRAME_SIZE];
            int hp = 0;

            motion_threshold_init(&adap, bv, 2.0f, 0.1f);
            motion_smoother_init(&s, 3, 10);
            acc_init(&acc);

            for (i = 0U; i < n; ++i) {
                const float* cur = &frames[i * VIDEO_FRAME_SIZE];
                if (!hp) {
                    hp = 1;
                    motion_threshold_update(&adap, 0.0f);
                    acc_add(&acc, s.is_active, truth[i]);
                } else {
                    float sad = video_frame_sad(prev, cur, VIDEO_FRAME_SIZE);
                    motion_threshold_update(&adap, sad);
                    float thr = motion_threshold_get(&adap);
                    acc_add(&acc, motion_smoother_update(&s, (sad > thr) ? 1 : 0), truth[i]);
                }
                frame_copy(prev, cur, VIDEO_FRAME_SIZE);
            }

            double p = acc_prec(&acc), r = acc_recall(&acc);
            double f1 = acc_f1(&acc);
            printf("  %8.0f  %7.1f  %7.1f  %7.3f  %7.2f\n",
                (double)bv, p * 100, r * 100, f1, (1.0 - acc_spec(&acc)) * 100);
            if (f1 > best_f1) { best_f1 = (float)f1; best_thresh = bv; }
        }
    }

    printf("  Best F1=%.3f at base_threshold=%.0f\n", (double)best_f1, (double)best_thresh);
    TEST_ASSERT(best_f1 >= 0.70, "Best F1 >= 0.70 in threshold sweep");
    free(frames); free(truth);
}

/* ========================================================================
 *  Scenario H: Short Burst Edge — 5-on/15-off, 200 frames (10 bursts).
 *
 *  Very short bursts; with 3 confirm / 10 idle the overhead is high.
 *  Motion step: ±0.30 → SAD ~460.  3 confirm / 10 idle.
 *  Per burst: PT ≈ 2 (frames 2-3), FN ≈ 3 (0,1,4), FP ≈ 10.
 *  F1 ≈ 0.20 expected.  Validates the pipeline handles extremes correctly.
 * ======================================================================== */

static int is_short_burst(size_t i) {
    size_t p = i % 20U;
    return (p < 5U) ? 1 : 0;
}

static void test_h_short_burst(void) {
    MotionAdaptiveThreshold adap;
    MotionTemporalSmoother s;
    Acc acc;
    float* frames;
    int* truth;
    const size_t n = 200U;  /* 10 bursts of 5-on/15-off */

    frames = (float*)calloc(n * VIDEO_FRAME_SIZE, sizeof(float));
    truth  = (int*)calloc(n, sizeof(int));

    g_rng = 222U;
    generate_sequential(frames, truth, n, is_short_burst, 0.30f, 0.002f);

    motion_threshold_init(&adap, 50.0f, 2.0f, 0.1f);
    motion_smoother_init(&s, 3, 10);
    run(frames, truth, n, &adap, &s, &acc);
    acc_print(&acc, "H: Short Burst (5-on/15-off, 10 bursts, stress test)", 0.10, 0.99, 0.99);
    free(frames); free(truth);
}

/* ========================================================================
 *  Main
 * ======================================================================== */

int main(void) {
    printf("=== Motion Detection Accuracy Benchmark ===\n");
    printf("Formula: threshold = base + factor * EMA_std\n");
    printf("Smoother: 3 confirm / 10 idle frames\n");
    printf("Frame generation: sequential (frame[i] = frame[i-1] + step)\n\n");

    test_a_all_static();
    test_b_bursts();
    test_c_frequent();
    test_d_noisy();
    test_e_extended();
    test_f_ramp();
    test_g_sweep();
    test_h_short_burst();

    printf("\n=== Results: %d failures ===\n", g_failures);
    printf("Targets per-scenario (adjusted for smoother overhead):\n");
    printf("  A/D: FPR=0 for all-static/noisy-static\n");
    printf("  B:   30-on/50-off, F1>=0.78, FPR<=22%%, FNR<=25%%\n");
    printf("  C:   20-on/30-off, F1>=0.70, FPR<=35%%, FNR<=25%% (10 idle / 30 static -> min FPR 33%%)\n");
    printf("  E:   60-on/30s,    F1>=0.88, FPR<=22%%, FNR<=12%%\n");
    printf("  F:   Ramp,         F1>=0.85, FPR<=20%%, FNR<=25%%\n");
    printf("  G:   Sweep,        Best F1>=0.70\n");
    printf("  H:   5-on/15-off,   F1>=0.10 (stress test, 3/10 overhead limit)\n");
    return (g_failures > 0) ? 1 : 0;
}
