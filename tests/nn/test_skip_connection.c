/**
 * @file test_skip_connection.c
 * @brief Unit tests for residual/skip connection forward and backward passes.
 */

#include "test_harness.h"
#include "residual/skip_connection.h"
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Initialisation & basic properties                                  */
/* ------------------------------------------------------------------ */

TEST(skip_init_defaults)
{
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    ASSERT_EQ_INT(SKIP_LAUREL_RW, (int)skip.mode, "mode set");
    ASSERT_FLOAT_EQ(0.0f, skip.alpha_raw, 1e-7f, "alpha_raw init 0");
    ASSERT_FLOAT_EQ(0.5f, skip.alpha_cache, 1e-7f, "alpha_cache init 0.5");
    ASSERT_FLOAT_EQ(0.5f, skip.beta_cache, 1e-7f, "beta_cache init 0.5");
    ASSERT_FLOAT_EQ(0.0f, skip.grad_alpha_raw, 1e-7f, "grad zero at init");
}

/* ------------------------------------------------------------------ */
/*  SKIP_NONE                                                          */
/* ------------------------------------------------------------------ */

TEST(skip_none_forward)
{
    SkipConnection skip;
    skip_init(&skip, SKIP_NONE);
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float f[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float out[4];
    skip_forward(out, f, x, 4, &skip);
    for (int i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(f[i], out[i], 1e-7f, "SKIP_NONE = F(x)");
    }
}

TEST(skip_none_backward_passthrough)
{
    SkipConnection skip;
    skip_init(&skip, SKIP_NONE);
    float f[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dout[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float d_f[4], d_x[4];
    skip_backward(d_f, d_x, f, x, dout, 4, &skip);
    for (int i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(dout[i], d_f[i], 1e-7f, "dF = dout");
        ASSERT_FLOAT_EQ(0.0f, d_x[i], 1e-7f, "dx = 0");
    }
}

/* ------------------------------------------------------------------ */
/*  SKIP_IDENTITY                                                      */
/* ------------------------------------------------------------------ */

TEST(skip_identity_forward)
{
    SkipConnection skip;
    skip_init(&skip, SKIP_IDENTITY);
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float f[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float out[4];
    skip_forward(out, f, x, 4, &skip);
    for (int i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(f[i] + x[i], out[i], 1e-7f, "y = F(x) + x");
    }
}

TEST(skip_identity_backward)
{
    SkipConnection skip;
    skip_init(&skip, SKIP_IDENTITY);
    float f[4]    = {0.1f, 0.2f, 0.3f, 0.4f};
    float x[4]    = {1.0f, 2.0f, 3.0f, 4.0f};
    float dout[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float d_f[4], d_x[4];
    skip_backward(d_f, d_x, f, x, dout, 4, &skip);
    for (int i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(dout[i], d_f[i], 1e-7f, "dF = dout");
        ASSERT_FLOAT_EQ(dout[i], d_x[i], 1e-7f, "dx = dout");
    }
}

/* ------------------------------------------------------------------ */
/*  SKIP_LAUREL_RW – forward                                           */
/* ------------------------------------------------------------------ */

TEST(laurel_rw_forward_alpha_beta_sum_to_one)
{
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float f[4] = {0.5f, 0.6f, 0.7f, 0.8f};
    float out[4];
    skip_forward(out, f, x, 4, &skip);
    ASSERT_FLOAT_EQ(skip.alpha_cache + skip.beta_cache, 1.0f, 1e-7f, "α+β = 1");
}

TEST(laurel_rw_forward_initial_alpha_is_half)
{
    /* alpha_raw=0 → sigmoid(0)=0.5 */
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    ASSERT_FLOAT_EQ(0.5f, skip.alpha_cache, 1e-7f, "initial α = 0.5");
    ASSERT_FLOAT_EQ(0.5f, skip.beta_cache, 1e-7f, "initial β = 0.5");
}

TEST(laurel_rw_forward_known_values)
{
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    /* manually set alpha_raw for deterministic test */
    skip.alpha_raw = 1.0f;  /* sigmoid(1) ≈ 0.73106 */
    float expected_alpha = 1.0f / (1.0f + expf(-1.0f));
    float expected_beta  = 1.0f - expected_alpha;

    float x[3] = {2.0f, -1.0f, 3.0f};
    float f[3] = {4.0f, 0.0f, -2.0f};
    float out[3];
    skip_forward(out, f, x, 3, &skip);
    for (int i = 0; i < 3; ++i) {
        float expected = expected_alpha * f[i] + expected_beta * x[i];
        ASSERT_FLOAT_EQ(expected, out[i], 1e-6f, "y = α·F + β·x");
    }
}

TEST(laurel_rw_forward_caches_updated)
{
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    skip.alpha_raw = -0.5f;
    float x[2] = {1.0f, 0.0f};
    float f[2] = {0.0f, 1.0f};
    float out[2];
    skip_forward(out, f, x, 2, &skip);
    float expected_alpha = 1.0f / (1.0f + expf(0.5f));
    ASSERT_FLOAT_EQ(expected_alpha, skip.alpha_cache, 1e-6f, "alpha cache correct");
    ASSERT_FLOAT_EQ(1.0f - expected_alpha, skip.beta_cache, 1e-6f, "beta cache correct");
}

TEST(laurel_rw_forward_identity_when_alpha_raw_large_pos)
{
    /* α_raw → +∞ means α≈1, β≈0 → y ≈ F(x) */
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    skip.alpha_raw = 10.0f;  /* sigmoid(10) ≈ 0.99995 */
    float x[3] = {1.0f, 2.0f, 3.0f};
    float f[3] = {4.0f, 5.0f, 6.0f};
    float out[3];
    skip_forward(out, f, x, 3, &skip);
    for (int i = 0; i < 3; ++i) {
        ASSERT_FLOAT_EQ(f[i], out[i], 1e-3f, "α≈1 → y≈F(x)");
    }
}

TEST(laurel_rw_forward_identity_when_alpha_raw_large_neg)
{
    /* α_raw → -∞ means α≈0, β≈1 → y ≈ x (skip F entirely) */
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    skip.alpha_raw = -10.0f;  /* sigmoid(-10) ≈ 0.000045 */
    float x[3] = {1.0f, 2.0f, 3.0f};
    float f[3] = {4.0f, 5.0f, 6.0f};
    float out[3];
    skip_forward(out, f, x, 3, &skip);
    for (int i = 0; i < 3; ++i) {
        ASSERT_FLOAT_EQ(x[i], out[i], 1e-3f, "α≈0 → y≈x");
    }
}

/* ------------------------------------------------------------------ */
/*  SKIP_LAUREL_RW – backward / gradient check                         */
/* ------------------------------------------------------------------ */

TEST(laurel_rw_backward_alpha_raw_gradient_sign)
{
    /* For F > x globally, α should increase (grad negative after loss),
     * so grad_alpha_raw accumulates the raw gradient. Let's just verify
     * it's non-zero and the right sign for a simple case. */
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    /* F >> x  →  using F is better → want α↑ → gradient should push α_raw↑ */
    float f[3]    = {10.0f, 10.0f, 10.0f};
    float x[3]    = {0.0f, 0.0f, 0.0f};
    float dout[3] = {1.0f, 1.0f, 1.0f};
    float out[3];  /* separate output buffer */
    float d_f[3], d_x[3];
    skip_forward(out, f, x, 3, &skip);  /* set caches */
    skip_backward(d_f, d_x, f, x, dout, 3, &skip);
    /* dα = Σ dout_i * (F_i - x_i) = 3 * 10 = 30
     * dα_raw = dα * α * β = 30 * 0.5 * 0.5 = 7.5 */
    float expected = 30.0f * 0.5f * 0.5f;
    ASSERT_FLOAT_EQ(expected, skip.grad_alpha_raw, 1e-4f, "grad_alpha_raw = Σ(F-x)·α·β");
}

TEST(laurel_rw_backward_finite_diff)
{
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    skip.alpha_raw = 0.8f;  /* sigmoid(0.8) ≈ 0.68997 */

    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float f[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    float out[4];
    skip_forward(out, f, x, 4, &skip);

    /* Use fixed dout (linear loss L = Σ dout_i * y_i) */
    float dout[4] = {0.8f, -0.4f, 1.5f, 0.3f};

    /* Analytical backward */
    float d_f_analytical[4], d_x_analytical[4];
    skip.grad_alpha_raw = 0.0f;  /* reset */
    skip_backward(d_f_analytical, d_x_analytical, f, x, dout, 4, &skip);
    float analytical_grad = skip.grad_alpha_raw;

    /* Finite difference for alpha_raw */
    float h = 1e-3f;
    SkipConnection skip_plus, skip_minus;
    skip_init(&skip_plus,  SKIP_LAUREL_RW);
    skip_init(&skip_minus, SKIP_LAUREL_RW);
    skip_plus.alpha_raw  = 0.8f + h;
    skip_minus.alpha_raw = 0.8f - h;

    float out_plus[4], out_minus[4];
    skip_forward(out_plus,  f, x, 4, &skip_plus);
    skip_forward(out_minus, f, x, 4, &skip_minus);

    float L_plus = 0.0f, L_minus = 0.0f;
    for (int i = 0; i < 4; ++i) {
        L_plus  += dout[i] * out_plus[i];
        L_minus += dout[i] * out_minus[i];
    }
    float fd_grad = (L_plus - L_minus) / (2.0f * h);
    ASSERT_FLOAT_EQ(fd_grad, analytical_grad, 1e-3f, "alpha_raw gradient FD check");
}

TEST(laurel_rw_backward_dF_and_dX)
{
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    skip.alpha_raw = 1.0f;
    float x[2]    = {1.0f, 2.0f};
    float f[2]    = {3.0f, 4.0f};
    float out[2];
    skip_forward(out, f, x, 2, &skip);

    float dout[2] = {0.1f, 0.2f};
    float d_f[2], d_x[2];
    skip.grad_alpha_raw = 0.0f;
    skip_backward(d_f, d_x, f, x, dout, 2, &skip);

    float alpha = skip.alpha_cache;
    float beta  = skip.beta_cache;
    ASSERT_FLOAT_EQ(dout[0] * alpha, d_f[0], 1e-7f, "dF = dout·α");
    ASSERT_FLOAT_EQ(dout[1] * alpha, d_f[1], 1e-7f, "dF = dout·α");
    ASSERT_FLOAT_EQ(dout[0] * beta, d_x[0], 1e-7f, "dX = dout·β");
    ASSERT_FLOAT_EQ(dout[1] * beta, d_x[1], 1e-7f, "dX = dout·β");
}

TEST(laurel_rw_gradient_accumulates)
{
    /* Two backward calls should ADD to grad_alpha_raw */
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    float x[2] = {1.0f, 2.0f};
    float f[2] = {3.0f, 4.0f};
    float out[2], d_f[2], d_x[2];
    skip_forward(out, f, x, 2, &skip);

    /* First backward */
    float dout1[2] = {1.0f, 0.0f};
    skip.grad_alpha_raw = 0.0f;
    skip_backward(d_f, d_x, f, x, dout1, 2, &skip);
    float g1 = skip.grad_alpha_raw;

    /* Second backward should ADD */
    float dout2[2] = {0.0f, 1.0f};
    skip_backward(d_f, d_x, f, x, dout2, 2, &skip);
    float g_total = skip.grad_alpha_raw;

    /* Both calls together should equal single call with combined dout */
    float dout_combined[2] = {1.0f, 1.0f};
    skip.grad_alpha_raw = 0.0f;
    skip_backward(d_f, d_x, f, x, dout_combined, 2, &skip);
    float g_combined = skip.grad_alpha_raw;

    ASSERT_FLOAT_EQ(g_combined, g_total, 1e-7f, "gradient is additive");
    ASSERT_TRUE(g1 != g_total, "second call changed the gradient");
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    RUN_TEST(skip_init_defaults);

    /* SKIP_NONE */
    RUN_TEST(skip_none_forward);
    RUN_TEST(skip_none_backward_passthrough);

    /* SKIP_IDENTITY */
    RUN_TEST(skip_identity_forward);
    RUN_TEST(skip_identity_backward);

    /* SKIP_LAUREL_RW forward */
    RUN_TEST(laurel_rw_forward_alpha_beta_sum_to_one);
    RUN_TEST(laurel_rw_forward_initial_alpha_is_half);
    RUN_TEST(laurel_rw_forward_known_values);
    RUN_TEST(laurel_rw_forward_caches_updated);
    RUN_TEST(laurel_rw_forward_identity_when_alpha_raw_large_pos);
    RUN_TEST(laurel_rw_forward_identity_when_alpha_raw_large_neg);

    /* SKIP_LAUREL_RW backward */
    RUN_TEST(laurel_rw_backward_alpha_raw_gradient_sign);
    RUN_TEST(laurel_rw_backward_finite_diff);
    RUN_TEST(laurel_rw_backward_dF_and_dX);
    RUN_TEST(laurel_rw_gradient_accumulates);

    printf("1..%d\n",    _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
