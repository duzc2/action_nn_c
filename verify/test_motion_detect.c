/**
 * @file test_motion_detect.c
 * @brief Unit tests for adaptive threshold and temporal smoother.
 */

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "video_processor.h"

static int g_failures = 0;
static int g_test_num = 0;

#define TEST_ASSERT(cond, msg) do { \
    g_test_num++; \
    if (!(cond)) { \
        printf("  FAIL test %d: %s\n", g_test_num, msg); \
        g_failures++; \
    } else { \
        printf("  OK   test %d: %s\n", g_test_num, msg); \
    } \
} while (0)

/* ── Test 1: Adaptive threshold initialization ── */
static void test_adaptive_init(void) {
    MotionAdaptiveThreshold t;
    motion_threshold_init(&t, 50.0f, 2.0f, 0.1f);

    TEST_ASSERT(t.base_threshold == 50.0f, "base_threshold == 50.0f");
    TEST_ASSERT(t.adapt_factor == 2.0f,   "adapt_factor == 2.0f");
    TEST_ASSERT(t.ema_alpha == 0.1f,      "ema_alpha == 0.1f");
    TEST_ASSERT(t.rolling_mean == 0.0f,   "rolling_mean == 0.0f");
    TEST_ASSERT(t.rolling_std == 0.0f,    "rolling_std == 0.0f");
    TEST_ASSERT(t.initialized == 0,       "not yet initialized");

    /* Before any SAD data, threshold should be base */
    {
        float thresh = motion_threshold_get(&t);
        TEST_ASSERT(thresh == 50.0f, "uninitialized -> base threshold returned");
    }
}

/* ── Test 2: Adaptive threshold update & get ── */
static void test_adaptive_update(void) {
    MotionAdaptiveThreshold t;

    motion_threshold_init(&t, 50.0f, 2.0f, 0.1f);

    /* First SAD: seed the mean */
    motion_threshold_update(&t, 100.0f);
    TEST_ASSERT(t.initialized == 1,       "initialized after first sample");
    TEST_ASSERT(fabsf(t.rolling_mean - 100.0f) < 0.01f, "mean seeded to first SAD");

    /* Second and third identical SADs: mean should be stable */
    motion_threshold_update(&t, 100.0f);
    motion_threshold_update(&t, 100.0f);
    TEST_ASSERT(fabsf(t.rolling_mean - 100.0f) < 0.01f, "mean stable with identical SAD");

    /* Adaptive threshold: base=50 + factor*std = 50 + 2*0 = 50 */
    {
        float thresh = motion_threshold_get(&t);
        TEST_ASSERT(fabsf(thresh - 50.0f) < 1.0f, "threshold = base + 2*std (50+0=50)");
    }

    /* Feed a large outlier: 500. Mean=140, std=40.
     * Threshold = 50 + 2*40 = 130 */
    motion_threshold_update(&t, 500.0f);
    /* After EMA: mean = 100 + 0.1*(500-100) = 140 */
    TEST_ASSERT(fabsf(t.rolling_mean - 140.0f) < 1.0f, "EMA mean updated after outlier");
    /* After EMA: std = 0 + 0.1*(|500-100| - 0) = 40 */
    TEST_ASSERT(fabsf(t.rolling_std - 40.0f) < 1.0f, "EMA std updated after outlier");
    {
        float thresh = motion_threshold_get(&t);
        TEST_ASSERT(fabsf(thresh - 130.0f) < 1.0f, "threshold = base + 2*std (50+80=130)");
    }
}

/* ── Test 3: Temporal smoother state machine ── */
static void test_temporal_smoother(void) {
    MotionTemporalSmoother s;
    int result;
    int i;

    /* Init: need 3 motion to trigger, 10 static to go idle */
    motion_smoother_init(&s, 3, 10);

    /* Starts as static */
    TEST_ASSERT(s.is_active == 0, "initial state is static");
    TEST_ASSERT(s.consecutive_motion == 0, "consecutive_motion is 0");
    TEST_ASSERT(s.consecutive_static == 0, "consecutive_static is 0");

    /* 1st motion frame: still static (need 3) */
    result = motion_smoother_update(&s, 1);
    TEST_ASSERT(result == 0, "1st motion -> still static");
    TEST_ASSERT(s.consecutive_motion == 1, "consecutive_motion == 1");

    /* 2nd motion: still static */
    result = motion_smoother_update(&s, 1);
    TEST_ASSERT(result == 0, "2nd motion -> still static");
    TEST_ASSERT(s.consecutive_motion == 2, "consecutive_motion == 2");

    /* 3rd motion: should trigger */
    result = motion_smoother_update(&s, 1);
    TEST_ASSERT(result == 1, "3rd motion -> triggers active");
    TEST_ASSERT(s.is_active == 1, "is_active == 1");
    TEST_ASSERT(s.consecutive_motion == 3, "consecutive_motion == 3");

    /* One static frame: not enough to go idle */
    result = motion_smoother_update(&s, 0);
    TEST_ASSERT(result == 1, "1 static after motion -> still active");
    TEST_ASSERT(s.consecutive_static == 1, "consecutive_static == 1");
    TEST_ASSERT(s.consecutive_motion == 0, "consecutive_motion reset to 0");

    /* 9 more static frames = 10 total: should go idle */
    for (i = 0; i < 8; i++) {
        result = motion_smoother_update(&s, 0);
    }
    TEST_ASSERT(result == 1, "9 static -> still active (need 10)");
    /* 10th static frame */
    result = motion_smoother_update(&s, 0);
    TEST_ASSERT(result == 0, "10th static -> returns to idle");
    TEST_ASSERT(s.is_active == 0, "is_active back to 0");
}

/* ── Test 4: Edge case — all zero SAD ── */
static void test_edge_zero(void) {
    MotionAdaptiveThreshold t;

    motion_threshold_init(&t, 50.0f, 2.0f, 0.1f);

    /* Feed all zeros */
    {
        int i;
        for (i = 0; i < 100; i++) {
            motion_threshold_update(&t, 0.0f);
        }
    }

    /* Zero mean -> adaptive = 0, so base (50) wins */
    {
        float thresh = motion_threshold_get(&t);
        TEST_ASSERT(thresh == 50.0f, "all-zero SAD -> falls back to base threshold");
    }

    /* SAD < unknown threshold always true, but smoother should handle it */
    {
        MotionTemporalSmoother s;
        int result;
        motion_smoother_init(&s, 3, 10);
        result = motion_smoother_update(&s, 0);
        TEST_ASSERT(result == 0, "zero SAD -> smoother stays static");
        TEST_ASSERT(s.consecutive_static == 1, "consecutive_static incremented");
    }
}

int main(void) {
    printf("=== Motion Detection Unit Tests ===\n\n");

    /* Prevent any static-analysis warnings about unused variables */
    (void)g_failures;
    (void)g_test_num;

    printf("--- Adaptive Threshold ---\n");
    test_adaptive_init();
    test_adaptive_update();

    printf("\n--- Temporal Smoother ---\n");
    test_temporal_smoother();

    printf("\n--- Edge Cases ---\n");
    test_edge_zero();

    printf("\n=== Results: %d failures ===\n", g_failures);

    return (g_failures > 0) ? 1 : 0;
}
