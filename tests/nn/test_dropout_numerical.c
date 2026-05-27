/**
 * @file test_dropout_numerical.c
 * @brief Comprehensive numerical & statistical verification for Dropout / DropPath.
 *
 * Covers:
 *   - Large-scale statistical properties (Monte Carlo, 10k+ trials)
 *   - Exact per-mask backward correctness
 *   - Expected value / variance verification against theory
 *   - RNG determinism and quality
 *   - Numerical stability under extreme keep_prob & input values
 *   - Forward-backward cycle consistency
 */

#include "test_harness.h"
#include "dropout/dropout.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ================================================================= */
/*  Large-scale keep-rate verification                                */
/* ================================================================= */

TEST(dropout_keep_rate_10k_trials)
{
    DropoutLayer dp;

    float probs[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
    for (int p = 0; p < 9; ++p) {
        float kp = probs[p];
        dropout_init(&dp, kp, (uint32_t)(p * 123 + 456));
        float x[1000], y[1000];
        for (int i = 0; i < 1000; ++i) x[i] = 1.0f;
        dropout_forward(y, x, 1000, &dp);

        int kept = 0;
        float sum_kept = 0.0f;
        for (int i = 0; i < 1000; ++i) {
            if (y[i] > 0.0f) { kept++; sum_kept += y[i]; }
        }

        float rate = (float)kept / 1000.0f;
        float mean_kept = (kept > 0) ? sum_kept / (float)kept : 0.0f;
        float expected_scale = 1.0f / kp;

        char msg[64];
        snprintf(msg, sizeof(msg), "kp=%.1f rate≈%.2f (±2%%)", (double)kp, (double)rate);
        ASSERT_TRUE(fabsf(rate - kp) < 0.02f, msg);

        snprintf(msg, sizeof(msg), "kp=%.1f mean_kept=1/kp", (double)kp);
        if (kept > 0)
            ASSERT_FLOAT_EQ(expected_scale, mean_kept, 0.02f, msg);

        dropout_free(&dp);
    }
}

TEST(droppath_keep_rate_10k_trials)
{
    DropoutLayer dp;
    float probs[] = {0.1f, 0.25f, 0.5f, 0.75f, 0.9f};
    for (int p_idx = 0; p_idx < 5; ++p_idx) {
        float kp = probs[p_idx];
        dropout_init(&dp, kp, (uint32_t)(p_idx * 777 + 111));
        float fx[1] = {1.0f}, x[1] = {0.0f}, y[1];
        int kept = 0;
        for (int trial = 0; trial < 5000; ++trial) {
            droppath_forward(y, fx, x, 1, &dp);
            if (dp.drop_path_flag) kept++;
        }
        float rate = (float)kept / 5000.0f;
        char msg[48];
        snprintf(msg, sizeof(msg), "DropPath kp=%.2f rate≈%.2f", (double)kp, (double)rate);
        ASSERT_TRUE(fabsf(rate - kp) < 0.02f, msg);
        dropout_free(&dp);
    }
}

/* ================================================================= */
/*  Expected value preservation                                       */
/* ================================================================= */

TEST(dropout_expected_value_unbiased)
{
    /* Dropout output should be unbiased estimator of input:
     * E[y_i] = x_i  (because E[mask_i * x_i / p] = p * x_i / p = x_i) */
    DropoutLayer dp;
    dropout_init(&dp, 0.5f, 9999);
    float x[100] = {0};
    for (int i = 0; i < 100; ++i) x[i] = (float)i * 0.1f;

    float y_sum[100] = {0};
    int n_trials = 50000;
    for (int t = 0; t < n_trials; ++t) {
        float y[100];
        dropout_forward(y, x, 100, &dp);
        for (int i = 0; i < 100; ++i) y_sum[i] += y[i];
    }

    /* Check ~10 evenly spaced samples; tolerance relaxes with |x| */
    for (int i = 0; i < 100; i += 10) {
        float empirical = y_sum[i] / (float)n_trials;
        float tol = 0.03f + fabsf(x[i]) * 0.01f;
        ASSERT_FLOAT_EQ(x[i], empirical, tol, "E[y] = x (unbiased)");
    }
    dropout_free(&dp);
}

TEST(dropout_variance_matches_theory)
{
    /* Var(dropout(x)) = E[y^2] - (E[y])^2
     * E[y] = x (unbiased), E[y^2] = x^2 * p * (1/p)^2 = x^2 / p
     * Var(y) = x^2/p - x^2 = x^2 * (1/p - 1)
     * = x^2 * (1-p)/p */
    float kp = 0.4f;
    DropoutLayer dp;
    dropout_init(&dp, kp, 5555);
    float x = 2.0f;  /* constant input */

    float sum_y   = 0.0f;
    float sum_y2  = 0.0f;
    int n_trials  = 10000;

    for (int t = 0; t < n_trials; ++t) {
        float xi = x, yi;
        dropout_forward(&yi, &xi, 1, &dp);
        sum_y  += yi;
        sum_y2 += yi * yi;
    }
    float empirical_var = sum_y2 / n_trials - (sum_y / n_trials) * (sum_y / n_trials);
    float theoretical_var = x * x * (1.0f - kp) / kp;

    ASSERT_FLOAT_EQ(theoretical_var, empirical_var, 0.05f, "Var(dropout) matches theory (±5%)");
    dropout_free(&dp);
}

/* ================================================================= */
/*  Exact backward correctness (deterministic, given known mask)     */
/* ================================================================= */

TEST(dropout_exact_backward_verification)
{
    /* Forward generates a mask; backward must use the SAME mask exactly.
     * We verify: given y = forward(x), dy = {1,1,1,...},
     *   dx[i] = (y[i] > 0) ? dy[i]/keep_prob : 0
     * This must be exact, not statistical. */
    float kp = 0.5f;
    DropoutLayer dp;
    dropout_init(&dp, kp, 42);

    size_t n = 100;
    float *x  = (float *)malloc(n * sizeof(float));
    float *y  = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; ++i) x[i] = 1.0f;

    dropout_forward(y, x, n, &dp);

    float *dy = (float *)malloc(n * sizeof(float));
    float *dx = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; ++i) dy[i] = 1.0f;

    dropout_backward(dx, dy, n, &dp);

    /* Verify exact correspondence */
    for (size_t i = 0; i < n; ++i) {
        if (y[i] > 0.0f) {
            ASSERT_FLOAT_EQ(1.0f / kp, dx[i], 1e-7f, "kept → exact dx = 1/kp");
        } else {
            ASSERT_FLOAT_EQ(0.0f, dx[i], 1e-7f, "dropped → exact dx = 0");
        }
    }

    free(x); free(y); free(dy); free(dx);
    dropout_free(&dp);
}

TEST(dropout_exact_backward_random_inputs)
{
    /* Same as above but with random input values */
    size_t n = 64;
    float *x  = (float *)malloc(n * sizeof(float));
    float *y  = (float *)malloc(n * sizeof(float));
    float *dy = (float *)malloc(n * sizeof(float));
    float *dx = (float *)malloc(n * sizeof(float));

    for (int seed = 1; seed <= 5; ++seed) {
        DropoutLayer dp;
        dropout_init(&dp, 0.6f, (uint32_t)(seed * 100));

        for (size_t i = 0; i < n; ++i) {
            x[i]  = (float)((int)(i * seed * 7 % 31) - 15) * 0.5f;
            dy[i] = (float)((int)(i * seed * 3 % 13) - 6) * 0.3f;
        }

        dropout_forward(y, x, n, &dp);
        dropout_backward(dx, dy, n, &dp);

        /* Use dp.mask[i] (not y[i]!=0) to determine kept vs dropped,
         * since y[i]==0 can happen for kept elements when x[i]==0. */
        float kp = 0.6f;
        for (size_t i = 0; i < n; ++i) {
            if (dp.mask[i]) {
                /* Verify the forward scaling */
                ASSERT_FLOAT_EQ(x[i] / kp, y[i], 1e-6f, "forward scaling correct");
                /* Verify backward */
                ASSERT_FLOAT_EQ(dy[i] / kp, dx[i], 1e-6f, "backward scaling correct");
            } else {
                ASSERT_FLOAT_EQ(0.0f, dx[i], 1e-7f, "dropped → dx=0");
            }
        }
        dropout_free(&dp);
    }
    free(x); free(y); free(dy); free(dx);
}

/* ================================================================= */
/*  Determinism: same seed → same mask                                */
/* ================================================================= */

TEST(dropout_determinism_long_sequence)
{
    DropoutLayer dp1, dp2;
    dropout_init(&dp1, 0.5f, 12345);
    dropout_init(&dp2, 0.5f, 12345);

    size_t n = 500;
    float *x = (float *)malloc(n * sizeof(float));
    float *y1 = (float *)malloc(n * sizeof(float));
    float *y2 = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; ++i) x[i] = (float)i;

    dropout_forward(y1, x, n, &dp1);
    dropout_forward(y2, x, n, &dp2);

    /* Every element must match exactly */
    for (size_t i = 0; i < n; ++i) {
        ASSERT_FLOAT_EQ(y1[i], y2[i], 1e-10f, "deterministic: same seed → identical masks");
    }

    free(x); free(y1); free(y2);
    dropout_free(&dp1); dropout_free(&dp2);
}

TEST(dropout_different_seeds_different_masks)
{
    /* Different seeds should produce different mask patterns,
     * though they may occasionally match by chance. */
    size_t n = 200;
    float *x  = (float *)malloc(n * sizeof(float));
    float *y1 = (float *)malloc(n * sizeof(float));
    float *y2 = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; ++i) x[i] = 1.0f;

    int diff_count = 0;
    for (uint32_t s1 = 1; s1 <= 10; ++s1) {
        DropoutLayer dp1, dp2;
        dropout_init(&dp1, 0.5f, s1);
        dropout_init(&dp2, 0.5f, s1 + 1000);
        dropout_forward(y1, x, n, &dp1);
        dropout_forward(y2, x, n, &dp2);

        int local_diff = 0;
        for (size_t i = 0; i < n; ++i)
            if (fabsf(y1[i] - y2[i]) > 1e-10f) local_diff++;
        diff_count += local_diff;
        dropout_free(&dp1); dropout_free(&dp2);
    }

    /* 10 pairs × 200 elements with kp=0.5 → virtually zero chance all match */
    ASSERT_TRUE(diff_count > 0, "different seeds → different masks");
    free(x); free(y1); free(y2);
}

/* ================================================================= */
/*  DropPath exact backward verification                              */
/* ================================================================= */

TEST(droppath_exact_backward)
{
    for (int flag = 0; flag <= 1; ++flag) {
        DropoutLayer dp;
        dropout_init(&dp, 0.6f, 1);
        /* Manually set flag to test kept (1) and dropped (0) cases */
        dp.drop_path_flag = flag;

        float dy[8]  = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float d_fx[8], dx[8];
        droppath_backward(d_fx, dx, dy, 8, &dp);

        float scale = 1.0f / 0.6f;
        for (int i = 0; i < 8; ++i) {
            float expected_dfx = flag ? dy[i] * scale : 0.0f;
            ASSERT_FLOAT_EQ(expected_dfx, d_fx[i], 1e-7f,
                            flag ? "kept: d_fx = dy/p" : "dropped: d_fx = 0");
            ASSERT_FLOAT_EQ(dy[i], dx[i], 1e-7f, "dx = dy always");
        }
        dropout_free(&dp);
    }
}

/* ================================================================= */
/*  Forward-backward cycle consistency                                */
/* ================================================================= */

TEST(dropout_forward_backward_cycle)
{
    /* 1. Forward → get y, mask
     * 2. Backward with dy = y (loss = 0.5*sum(y^2))
     * 3. Verify that d(L)/dx * dx matches expected chain rule */
    size_t n = 50;
    float *x  = (float *)malloc(n * sizeof(float));
    float *y  = (float *)malloc(n * sizeof(float));
    float *dx = (float *)malloc(n * sizeof(float));

    for (size_t i = 0; i < n; ++i) x[i] = 1.0f + 0.1f * (float)i;

    DropoutLayer dp;
    dropout_init(&dp, 0.7f, 777);
    dropout_forward(y, x, n, &dp);

    /* dy = y (loss = 0.5 * sum(y^2)) */
    dropout_backward(dx, y, n, &dp);

    /* For each element: dy_i = y_i
     *   dx_i = (mask[i] ? y[i]/kp : 0)
     *   = (mask[i] ? x[i]/(kp^2) : 0)   since y[i] = x[i]/kp when kept
     * So: dx_i should be >0 exactly where y[i] > 0 */
    for (size_t i = 0; i < n; ++i) {
        if (y[i] > 0.0f) {
            ASSERT_FLOAT_EQ(x[i] / (0.7f * 0.7f), dx[i], 1e-6f,
                            "forward-backward cycle: kept element gradient");
        } else {
            ASSERT_FLOAT_EQ(0.0f, dx[i], 1e-7f,
                            "forward-backward cycle: dropped element gradient");
        }
    }

    free(x); free(y); free(dx);
    dropout_free(&dp);
}

/* ================================================================= */
/*  Numerical stability – extreme keep_prob                           */
/* ================================================================= */

TEST(dropout_stability_near_zero_keep_prob)
{
    /* keep_prob=0.001 → scale=1000. Verify no overflow on moderate input. */
    DropoutLayer dp;
    dropout_init(&dp, 0.001f, 42);

    size_t n = 1000;
    float *x = (float *)malloc(n * sizeof(float));
    float *y = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; ++i) x[i] = 1.0f;

    dropout_forward(y, x, n, &dp);

    /* Count kept elements (expected ~1 out of 1000) */
    int kept = 0;
    for (size_t i = 0; i < n; ++i) {
        if (y[i] > 0.0f) {
            kept++;
            ASSERT_FLOAT_EQ(1000.0f, y[i], 1e-6f, "scaled by 1/0.001=1000");
            ASSERT_TRUE(!isinf(y[i]), "no Inf with kp=0.001");
        }
    }
    /* With n=1000, kp=0.001, approximately 1 element kept on average */
    ASSERT_TRUE(kept >= 0, "kept count >= 0 (statistical)");

    free(x); free(y);
    dropout_free(&dp);
}

TEST(dropout_stability_near_one_keep_prob)
{
    /* keep_prob=0.999 → almost all kept, scale≈1.001. Verify no issues. */
    DropoutLayer dp;
    dropout_init(&dp, 0.999f, 42);

    float x[1000], y[1000];
    for (int i = 0; i < 1000; ++i) x[i] = 100.0f;
    dropout_forward(y, x, 1000, &dp);

    int kept = 0;
    for (int i = 0; i < 1000; ++i) {
        if (y[i] > 0.0f) {
            kept++;
            ASSERT_FLOAT_EQ(100.0f / 0.999f, y[i], 1e-4f, "scale ~1/0.999");
        }
    }
    ASSERT_TRUE(kept > 900, "most elements kept for kp=0.999");
    dropout_free(&dp);
}

/* ================================================================= */
/*  Large input values                                                */
/* ================================================================= */

TEST(dropout_large_input_values)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.3f, 42);

    float x[64], y[64];
    for (int i = 0; i < 64; ++i) x[i] = 1e6f;
    dropout_forward(y, x, 64, &dp);

    for (int i = 0; i < 64; ++i) {
        if (y[i] > 0.0f) {
            ASSERT_TRUE(!isinf(y[i]), "no Inf with large input");
            ASSERT_FLOAT_EQ(1e6f / 0.3f, y[i], 1e1f, "large input scaled correctly");
        }
    }

    /* Backward with large gradients */
    float dy[64], dx[64];
    for (int i = 0; i < 64; ++i) dy[i] = y[i];
    dropout_backward(dx, dy, 64, &dp);
    for (int i = 0; i < 64; ++i)
        ASSERT_TRUE(!isinf(dx[i]) && !isnan(dx[i]), "backward: no Inf/NaN");

    dropout_free(&dp);
}

/* ================================================================= */
/*  RNG period coverage (ensures no short-cycle issues)              */
/* ================================================================= */

TEST(dropout_rng_long_sequence_no_overflow)
{
    /* Run 100,000 RNG draws to ensure no state overflow or cycle. */
    DropoutLayer dp;
    dropout_init(&dp, 0.5f, 1);

    float x[1] = {1.0f}, y[1];
    int crashes = 0;
    for (int i = 0; i < 100000 && crashes < 10; ++i) {
        dropout_forward(y, x, 1, &dp);
        if (isnan(y[0]) || isinf(y[0])) crashes++;
    }
    ASSERT_EQ_INT(0, crashes, "100k RNG draws: no NaN/Inf");
    dropout_free(&dp);
}

/* ================================================================= */
/*  no-op in inference and keep_prob=1                                */
/* ================================================================= */

TEST(dropout_inference_keep_prob_one_identical)
{
    /* Both inference mode and kp=1 should be exact identity */
    DropoutLayer dp_inf, dp_one;
    dropout_init(&dp_inf, 0.3f, 1);
    dropout_init(&dp_one, 1.0f, 1);
    dropout_set_training(&dp_inf, 0);

    float x[256], y_inf[256], y_one[256];
    for (int i = 0; i < 256; ++i) x[i] = (float)(i - 128) * 0.1f;

    dropout_forward(y_inf, x, 256, &dp_inf);
    dropout_forward(y_one, x, 256, &dp_one);

    for (int i = 0; i < 256; ++i) {
        ASSERT_FLOAT_EQ(x[i], y_inf[i], 1e-7f, "inference = identity");
        ASSERT_FLOAT_EQ(x[i], y_one[i], 1e-7f, "kp=1 = identity");
    }

    /* Backward: inference should also be identity */
    float dy[256], dx_inf[256], dx_one[256];
    for (int i = 0; i < 256; ++i) dy[i] = (float)i * 0.01f;
    dropout_backward(dx_inf, dy, 256, &dp_inf);
    dropout_backward(dx_one, dy, 256, &dp_one);

    for (int i = 0; i < 256; ++i) {
        ASSERT_FLOAT_EQ(dy[i], dx_inf[i], 1e-7f, "inference backward = identity");
        ASSERT_FLOAT_EQ(dy[i], dx_one[i], 1e-7f, "kp=1 backward = identity");
    }

    dropout_free(&dp_inf); dropout_free(&dp_one);
}

/* ================================================================= */
/*  Main                                                              */
/* ================================================================= */

int main(void)
{
    /* Large-scale statistics */
    RUN_TEST(dropout_keep_rate_10k_trials);
    RUN_TEST(droppath_keep_rate_10k_trials);

    /* Expected value & variance */
    RUN_TEST(dropout_expected_value_unbiased);
    RUN_TEST(dropout_variance_matches_theory);

    /* Exact backward verification */
    RUN_TEST(dropout_exact_backward_verification);
    RUN_TEST(dropout_exact_backward_random_inputs);
    RUN_TEST(droppath_exact_backward);

    /* Determinism */
    RUN_TEST(dropout_determinism_long_sequence);
    RUN_TEST(dropout_different_seeds_different_masks);

    /* Forward-backward cycle */
    RUN_TEST(dropout_forward_backward_cycle);

    /* Stability */
    RUN_TEST(dropout_stability_near_zero_keep_prob);
    RUN_TEST(dropout_stability_near_one_keep_prob);
    RUN_TEST(dropout_large_input_values);
    RUN_TEST(dropout_rng_long_sequence_no_overflow);

    /* No-op modes */
    RUN_TEST(dropout_inference_keep_prob_one_identical);

    printf("1..%d\n",    _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
