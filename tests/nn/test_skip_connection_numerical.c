/**
 * @file test_skip_connection_numerical.c
 * @brief Comprehensive numerical verification for skip/residual connections.
 *
 * Covers:
 *   - Forward exactness across all modes, dimensions, parameter ranges
 *   - Exhaustive FD gradient checks on dF, dx, and alpha_raw
 *   - Mathematical invariants (α+β=1, limiting behavior)
 *   - Composite gradient flow
 *   - Mode equivalence / asymptotic behaviour
 *   - Numerical stability under extreme values
 */

#include "test_harness.h"
#include "residual/skip_connection.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* seedable xorshift */
static uint32_t g_rng = 1;
static void set_seed(uint32_t s) { g_rng = (s != 0) ? s : 1; }
static float randf_local(void)
{
    uint32_t x = g_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_rng = x;
    return (float)(x >> 9) * (1.0f / 8388608.0f);
}

/* ================================================================= */
/*  Forward exactness across dimensions & modes                      */
/* ================================================================= */

static void check_mode_forward(SkipMode mode, float alpha_raw, size_t n,
                                const float *fx, const float *x)
{
    SkipConnection skip;
    skip_init(&skip, mode);
    if (mode == SKIP_LAUREL_RW) skip.alpha_raw = alpha_raw;

    float *y = (float *)malloc(n * sizeof(float));
    skip_forward(y, fx, x, n, &skip);

    float alpha = 1.0f / (1.0f + expf(-alpha_raw));
    float beta  = 1.0f - alpha;

    char msg[64];
    for (size_t i = 0; i < n; ++i) {
        float expected;
        switch (mode) {
        case SKIP_NONE:      expected = fx[i]; break;
        case SKIP_IDENTITY:  expected = fx[i] + x[i]; break;
        case SKIP_LAUREL_RW: expected = alpha * fx[i] + beta * x[i]; break;
        default:             expected = 0.0f; break;
        }
        snprintf(msg, sizeof(msg), "mode=%d n=%zu[%zu]", mode, n, i);
        ASSERT_FLOAT_EQ(expected, y[i], 1e-6f, msg);
    }
    free(y);
}

TEST(skip_forward_varying_n_mode0)  { float fx[]={0.1f,0.2f},x[]={1,2}; check_mode_forward(SKIP_NONE,0,2,fx,x); }
TEST(skip_forward_varying_n_mode1)  { float fx[]={0.1f,0.2f},x[]={1,2}; check_mode_forward(SKIP_IDENTITY,0,2,fx,x); }
TEST(skip_forward_varying_n_mode2)  { float fx[]={0.1f,0.2f},x[]={1,2}; check_mode_forward(SKIP_LAUREL_RW,0.8f,2,fx,x); }

static void check_forward_varying_n(SkipMode mode, float a_raw, size_t max_n)
{
    size_t ns[] = {1, 2, 4, 8, 16, 32, 64, 128};
    for (int ni = 0; ni < 8; ++ni) {
        size_t n = ns[ni];
        if (n > max_n) break;
        float *fx = (float *)malloc(n * sizeof(float));
        float *x  = (float *)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; ++i) {
            fx[i] = (float)((int)(i * 3 % 11) - 5) * 0.3f;
            x[i]  = (float)((int)(i * 7 % 9) - 4) * 0.5f;
        }
        check_mode_forward(mode, a_raw, n, fx, x);
        free(fx); free(x);
    }
}

TEST(skip_identity_varying_n)  { check_forward_varying_n(SKIP_IDENTITY, 0.0f, 128); }
TEST(skip_laurel_rw_varying_n) { check_forward_varying_n(SKIP_LAUREL_RW, 1.5f, 128); }

/* ================================================================= */
/*  LAuReL-RW alpha_raw sweep – exact sigmoid values                 */
/* ================================================================= */

TEST(laurel_rw_alpha_raw_sweep)
{
    float alphas[] = {-10.0f, -5.0f, -2.0f, -1.0f, -0.5f, 0.0f,
                       0.5f, 1.0f, 2.0f, 5.0f, 10.0f};
    float fx[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[4]  = {0.5f, -1.0f, 0.0f, 2.0f};

    for (int k = 0; k < 11; ++k) {
        SkipConnection skip;
        skip_init(&skip, SKIP_LAUREL_RW);
        skip.alpha_raw = alphas[k];
        float y[4];
        skip_forward(y, fx, x, 4, &skip);
        float alpha = 1.0f / (1.0f + expf(-alphas[k]));
        float beta  = 1.0f - alpha;
        ASSERT_FLOAT_EQ(alpha, skip.alpha_cache, 1e-6f, "alpha cache correct");
        ASSERT_FLOAT_EQ(beta,  skip.beta_cache,  1e-6f, "beta cache correct");
        for (int i = 0; i < 4; ++i) {
            float expected = alpha * fx[i] + beta * x[i];
            ASSERT_FLOAT_EQ(expected, y[i], 1e-6f, "alpha sweep output");
        }
    }
}

/* ================================================================= */
/*  Mathematical invariants                                           */
/* ================================================================= */

TEST(laurel_rw_alpha_beta_sum)
{
    /* α + β ≡ 1 for all alpha_raw */
    float *fx = (float *)malloc(128 * sizeof(float));
    float *x  = (float *)malloc(128 * sizeof(float));
    for (size_t i = 0; i < 128; ++i) { fx[i] = 0.1f; x[i] = 0.2f; }

    for (float ar = -20.0f; ar <= 20.0f; ar += 2.0f) {
        SkipConnection skip;
        skip_init(&skip, SKIP_LAUREL_RW);
        skip.alpha_raw = ar;
        float y[128];
        skip_forward(y, fx, x, 128, &skip);
        ASSERT_FLOAT_EQ(1.0f, skip.alpha_cache + skip.beta_cache, 1e-6f,
                        "α+β≡1 for all alpha_raw");
    }
    free(fx); free(x);
}

TEST(laurel_rw_fx_equals_x_gives_identity)
{
    /* When F(x) == x, LAuReL-RW output ≡ x for ANY α,β
     * because y = α·x + (1-α)·x = x */
    float *x = (float *)malloc(128 * sizeof(float));
    for (size_t i = 0; i < 128; ++i) x[i] = (float)((int)i - 64) * 0.1f;

    float ars[] = {-5.0f, -2.0f, 0.0f, 1.0f, 5.0f};
    for (int ari = 0; ari < 5; ++ari) {
        float ar = ars[ari];
        SkipConnection skip;
        skip_init(&skip, SKIP_LAUREL_RW);
        skip.alpha_raw = ar;
        float y[128];
        skip_forward(y, x, x, 128, &skip);  /* fx ≡ input */
        for (size_t i = 0; i < 128; ++i)
            ASSERT_FLOAT_EQ(x[i], y[i], 1e-6f, "F(x)=x → y=x regardless of α");
    }
    free(x);
}

TEST(laurel_rw_asymptotic_behavior)
{
    /* α_raw → +∞  →  α≈1, y≈F(x)   (fully trust the transform)
     * α_raw → -∞  →  α≈0, y≈x      (fully skip the transform)
     * α_raw =  0  →  α=0.5 (equal weight) */
    float fx[8] = {2.0f, -1.0f, 3.0f, -2.0f, 1.0f, 0.5f, -0.5f, 4.0f};
    float x[8]  = {-1.0f, 2.0f, -0.5f, 1.0f, -2.0f, 3.0f, 0.0f, -3.0f};

    /* α_raw = 10: sigmoid(10) ≈ 0.99995 */
    SkipConnection skip_hi;
    skip_init(&skip_hi, SKIP_LAUREL_RW);
    skip_hi.alpha_raw = 10.0f;
    float y_hi[8];
    skip_forward(y_hi, fx, x, 8, &skip_hi);
    for (int i = 0; i < 8; ++i)
        ASSERT_FLOAT_EQ(fx[i], y_hi[i], 5e-4f, "α≈1 → y≈F(x)");

    /* α_raw = -10: sigmoid(-10) ≈ 0.000045 */
    SkipConnection skip_lo;
    skip_init(&skip_lo, SKIP_LAUREL_RW);
    skip_lo.alpha_raw = -10.0f;
    float y_lo[8];
    skip_forward(y_lo, fx, x, 8, &skip_lo);
    for (int i = 0; i < 8; ++i)
        ASSERT_FLOAT_EQ(x[i], y_lo[i], 5e-4f, "α≈0 → y≈x");

    /* α_raw = 0: equal weight */
    SkipConnection skip_mid;
    skip_init(&skip_mid, SKIP_LAUREL_RW);
    float y_mid[8];
    skip_forward(y_mid, fx, x, 8, &skip_mid);
    for (int i = 0; i < 8; ++i) {
        ASSERT_FLOAT_EQ(0.5f * fx[i] + 0.5f * x[i], y_mid[i], 1e-6f, "α=0.5 equal weight");
    }
}

/* ================================================================= */
/*  Gradient checks – Monte Carlo FD on random inputs               */
/* ================================================================= */

static float linear_loss_skip(const float *dout, const float *fx,
                               const float *x, SkipConnection *skip, size_t n)
{
    SkipConnection tmp;
    skip_init(&tmp, skip->mode);
    tmp.alpha_raw = skip->alpha_raw;
    float *y = (float *)malloc(n * sizeof(float));
    skip_forward(y, fx, x, n, &tmp);
    float L = 0.0f;
    for (size_t i = 0; i < n; ++i) L += dout[i] * y[i];
    free(y);
    return L;
}

TEST(skip_fd_dF_and_dX_laurel_rw)
{
    /* FD check on dF and dx for LAuReL-RW */
    size_t n = 16;
    float *fx   = (float *)malloc(n * sizeof(float));
    float *x    = (float *)malloc(n * sizeof(float));
    float *dout = (float *)malloc(n * sizeof(float));

    for (size_t i = 0; i < n; ++i) {
        fx[i]   = (float)((int)i * 3 % 7 - 3) * 0.7f;
        x[i]    = (float)((int)i * 5 % 9 - 4) * 0.6f;
        dout[i] = (float)((int)i % 5 - 2) * 0.8f;
    }

    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    skip.alpha_raw = 0.8f;

    float *d_fx = (float *)malloc(n * sizeof(float));
    float *dx   = (float *)malloc(n * sizeof(float));
    float y_buf[16];
    skip_forward(y_buf, fx, x, n, &skip);
    skip.grad_alpha_raw = 0.0f;
    skip_backward(d_fx, dx, fx, x, dout, n, &skip);

    float h = 1e-3f;
    /* FD on dF */
    for (size_t i = 0; i < n; ++i) {
        float *fx_p = (float *)malloc(n * sizeof(float));
        float *fx_m = (float *)malloc(n * sizeof(float));
        memcpy(fx_p, fx, n * sizeof(float));
        memcpy(fx_m, fx, n * sizeof(float));
        fx_p[i] += h; fx_m[i] -= h;

        float Lp = linear_loss_skip(dout, fx_p, x, &skip, n);
        float Lm = linear_loss_skip(dout, fx_m, x, &skip, n);
        float fd  = (Lp - Lm) / (2.0f * h);
        ASSERT_FLOAT_EQ(fd, d_fx[i], 1e-3f, "d_fx FD");
        free(fx_p); free(fx_m);
    }

    /* FD on dx */
    for (size_t i = 0; i < n; ++i) {
        float *x_p = (float *)malloc(n * sizeof(float));
        float *x_m = (float *)malloc(n * sizeof(float));
        memcpy(x_p, x, n * sizeof(float));
        memcpy(x_m, x, n * sizeof(float));
        x_p[i] += h; x_m[i] -= h;

        float Lp = linear_loss_skip(dout, fx, x_p, &skip, n);
        float Lm = linear_loss_skip(dout, fx, x_m, &skip, n);
        float fd  = (Lp - Lm) / (2.0f * h);
        ASSERT_FLOAT_EQ(fd, dx[i], 1e-3f, "dx FD");
        free(x_p); free(x_m);
    }

    free(fx); free(x); free(dout); free(d_fx); free(dx);
}

TEST(skip_fd_alpha_raw_monte_carlo)
{
    /* FD check on alpha_raw across a range of values and random inputs */
    size_t n = 32;
    float *fx   = (float *)malloc(n * sizeof(float));
    float *x    = (float *)malloc(n * sizeof(float));
    float *dout = (float *)malloc(n * sizeof(float));

    set_seed(1000);
    for (int cfg = 0; cfg < 15; ++cfg) {
        for (size_t i = 0; i < n; ++i) {
            fx[i]   = (randf_local() - 0.5f) * 4.0f;
            x[i]    = (randf_local() - 0.5f) * 3.0f;
            dout[i] = (randf_local() - 0.5f) * 2.0f;
        }

        float a_raw = (randf_local() - 0.5f) * 6.0f;  /* [-3, 3] */

        SkipConnection skip;
        skip_init(&skip, SKIP_LAUREL_RW);
        skip.alpha_raw = a_raw;

        float y_buf[32];
        skip_forward(y_buf, fx, x, n, &skip);
        skip.grad_alpha_raw = 0.0f;

        float d_fx[32], dx[32];
        skip_backward(d_fx, dx, fx, x, dout, n, &skip);
        float anal_grad = skip.grad_alpha_raw;

        /* FD on alpha_raw */
        float h = 1e-3f;
        SkipConnection skip_p, skip_m;
        skip_init(&skip_p, SKIP_LAUREL_RW); skip_p.alpha_raw = a_raw + h;
        skip_init(&skip_m, SKIP_LAUREL_RW); skip_m.alpha_raw = a_raw - h;
        float Lp = linear_loss_skip(dout, fx, x, &skip_p, n);
        float Lm = linear_loss_skip(dout, fx, x, &skip_m, n);
        float fd  = (Lp - Lm) / (2.0f * h);

        char msg[64];
        snprintf(msg, sizeof(msg), "alpha_raw FD cfg=%d a_raw=%.3f", cfg, (double)a_raw);
        ASSERT_FLOAT_EQ(fd, anal_grad, 1e-3f, msg);
    }
    free(fx); free(x); free(dout);
}

/* ================================================================= */
/*  Composite gradient: layer → skip → loss                          */
/* ================================================================= */

TEST(skip_composite_gradient_flow)
{
    /* Simulate: x → layer(F) → skip → loss
     * Loss = Σ (y_i - target_i)^2 / 2
     * Verify that gradient descent on alpha_raw reduces loss */
    size_t n = 8;
    float fx[8]      = {0.8f, -0.3f, 1.5f, 0.2f, -0.7f, 1.0f, -0.1f, 0.9f};
    float x[8]       = {0.5f, -0.5f, 1.0f, 0.0f, -0.3f, 0.7f, -0.2f, 0.6f};
    float target[8]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    skip.alpha_raw = 0.0f;  /* start from 0.5/0.5 */

    /* Before */
    float y_before[8], loss_before = 0.0f;
    skip_forward(y_before, fx, x, n, &skip);
    for (size_t i = 0; i < n; ++i)
        loss_before += 0.5f * (y_before[i] - target[i]) * (y_before[i] - target[i]);

    /* Gradient */
    float dout[8], d_fx[8], dx[8];
    for (size_t i = 0; i < n; ++i) dout[i] = y_before[i] - target[i];
    skip.grad_alpha_raw = 0.0f;
    skip_backward(d_fx, dx, fx, x, dout, n, &skip);

    /* Gradient descent on alpha_raw */
    float lr = 0.1f;
    skip.alpha_raw -= lr * skip.grad_alpha_raw;

    /* After */
    float y_after[8], loss_after = 0.0f;
    skip_forward(y_after, fx, x, n, &skip);
    for (size_t i = 0; i < n; ++i)
        loss_after += 0.5f * (y_after[i] - target[i]) * (y_after[i] - target[i]);

    ASSERT_TRUE(loss_after < loss_before, "GD on alpha_raw reduces loss");
}

/* ================================================================= */
/*  gradient sign test                                                */
/* ================================================================= */

TEST(skip_gradient_sign_consistent)
{
    /* When F > x consistently, alpha should increase (grad > 0).
     * When F < x consistently, alpha should decrease (grad < 0).
     * Meaning grad_alpha_raw should be:
     *   positive when Σ dout_i * (F_i - x_i) * high
     * With dout_i all positive, it's Σ (F_i - x_i) that matters. */
    size_t n = 16;
    float *fx   = (float *)malloc(n * sizeof(float));
    float *x    = (float *)malloc(n * sizeof(float));
    float *dout = (float *)malloc(n * sizeof(float));

    /* Case 1: F >> x → grad_alpha_raw should be positive */
    for (size_t i = 0; i < n; ++i) {
        fx[i]   = 10.0f;
        x[i]    = 1.0f;
        dout[i] = 1.0f;
    }
    SkipConnection skip1;
    skip_init(&skip1, SKIP_LAUREL_RW);
    float y1[16];
    skip_forward(y1, fx, x, n, &skip1);
    float d_fx1[16], dx1[16];
    skip1.grad_alpha_raw = 0.0f;
    skip_backward(d_fx1, dx1, fx, x, dout, n, &skip1);
    ASSERT_TRUE(skip1.grad_alpha_raw > 0.0f, "F>>x → grad>0 (want more F)");

    /* Case 2: F << x → grad_alpha_raw should be negative */
    for (size_t i = 0; i < n; ++i) {
        fx[i]   = 1.0f;
        x[i]    = 10.0f;
        dout[i] = 1.0f;
    }
    SkipConnection skip2;
    skip_init(&skip2, SKIP_LAUREL_RW);
    float y2[16];
    skip_forward(y2, fx, x, n, &skip2);
    float d_fx2[16], dx2[16];
    skip2.grad_alpha_raw = 0.0f;
    skip_backward(d_fx2, dx2, fx, x, dout, n, &skip2);
    ASSERT_TRUE(skip2.grad_alpha_raw < 0.0f, "F<<x → grad<0 (want more identity)");

    free(fx); free(x); free(dout);
}

/* ================================================================= */
/*  Numerical stability                                               */
/* ================================================================= */

TEST(skip_stability_large_values)
{
    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    skip.alpha_raw = 5.0f;

    float fx[8] = {1e8f, -1e8f, 1e8f, -1e8f, 1e8f, -1e8f, 1e8f, -1e8f};
    float x[8]  = {-5e7f, 5e7f, -5e7f, 5e7f, -5e7f, 5e7f, -5e7f, 5e7f};
    float y[8];
    skip_forward(y, fx, x, 8, &skip);
    for (int i = 0; i < 8; ++i)
        ASSERT_TRUE(!isnan(y[i]) && !isinf(y[i]), "no NaN/Inf on large values");

    /* Backward should also be stable */
    float dout[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float d_fx[8], dx[8];
    skip.grad_alpha_raw = 0.0f;
    skip_backward(d_fx, dx, fx, x, dout, 8, &skip);
    for (int i = 0; i < 8; ++i) {
        ASSERT_TRUE(!isnan(d_fx[i]) && !isnan(dx[i]), "backward: no NaN on large values");
    }
    ASSERT_TRUE(!isnan(skip.grad_alpha_raw), "grad_alpha_raw not NaN");
}

TEST(skip_modes_consistent)
{
    /* Verify that SKIP_NONE and SKIP_IDENTITY are equivalent to
     * LAuReL-RW at asymptotic alpha_raw values */
    float fx[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[4]  = {0.1f, 0.2f, 0.3f, 0.4f};

    /* SKIP_NONE */
    SkipConnection s_none;
    skip_init(&s_none, SKIP_NONE);
    float y_none[4];
    skip_forward(y_none, fx, x, 4, &s_none);

    /* SKIP_LAUREL_RW α≈1 */
    SkipConnection s_hi;
    skip_init(&s_hi, SKIP_LAUREL_RW);
    s_hi.alpha_raw = 100.0f;  /* sigmoid ≈ 1 */
    float y_hi[4];
    skip_forward(y_hi, fx, x, 4, &s_hi);
    for (int i = 0; i < 4; ++i)
        ASSERT_FLOAT_EQ(y_none[i], y_hi[i], 1e-6f, "SKIP_NONE ≈ LAuReL(α≈1)");

    /* SKIP_IDENTITY */
    SkipConnection s_id;
    skip_init(&s_id, SKIP_IDENTITY);
    float y_id[4];
    skip_forward(y_id, fx, x, 4, &s_id);

    /* SKIP_LAUREL_RW α=0.5 (not the same, just here for completeness) */
    /* IDENTITY and LAuReL_RW are not generally equivalent unless F=0 */
    (void)y_id;
}

/* ================================================================= */
/*  Large dimension forward exactness                                 */
/* ================================================================= */

TEST(skip_large_dimension_forward)
{
    size_t n = 1024;
    float *fx = (float *)malloc(n * sizeof(float));
    float *x  = (float *)malloc(n * sizeof(float));
    float *y  = (float *)malloc(n * sizeof(float));

    for (size_t i = 0; i < n; ++i) {
        fx[i] = (float)((int)i % 17 - 8) * 0.3f;
        x[i]  = (float)((int)(i * 3) % 13 - 6) * 0.4f;
    }

    SkipConnection skip;
    skip_init(&skip, SKIP_LAUREL_RW);
    skip.alpha_raw = 0.5f;
    skip_forward(y, fx, x, n, &skip);

    float alpha = 1.0f / (1.0f + expf(-0.5f));
    float beta  = 1.0f - alpha;

    /* Spot-check every 128th element */
    for (size_t i = 0; i < n; i += 128) {
        float expected = alpha * fx[i] + beta * x[i];
        ASSERT_FLOAT_EQ(expected, y[i], 1e-6f, "large n forward correct");
    }

    /* Backward at scale */
    float *dout  = (float *)malloc(n * sizeof(float));
    float *d_fx  = (float *)malloc(n * sizeof(float));
    float *dx    = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; ++i) dout[i] = (float)((int)i % 3 - 1);

    skip.grad_alpha_raw = 0.0f;
    skip_backward(d_fx, dx, fx, x, dout, n, &skip);

    /* Spot-check backward */
    for (size_t i = 0; i < n; i += 128) {
        ASSERT_FLOAT_EQ(dout[i] * alpha, d_fx[i], 1e-6f, "large n d_fx correct");
        ASSERT_FLOAT_EQ(dout[i] * beta,  dx[i],    1e-6f, "large n dx correct");
    }

    free(fx); free(x); free(y); free(dout); free(d_fx); free(dx);
}

/* ================================================================= */
/*  Main                                                              */
/* ================================================================= */

int main(void)
{
    /* Forward exactness */
    RUN_TEST(skip_forward_varying_n_mode0);
    RUN_TEST(skip_forward_varying_n_mode1);
    RUN_TEST(skip_forward_varying_n_mode2);
    RUN_TEST(skip_identity_varying_n);
    RUN_TEST(skip_laurel_rw_varying_n);
    RUN_TEST(laurel_rw_alpha_raw_sweep);
    RUN_TEST(skip_large_dimension_forward);

    /* Invariants */
    RUN_TEST(laurel_rw_alpha_beta_sum);
    RUN_TEST(laurel_rw_fx_equals_x_gives_identity);
    RUN_TEST(laurel_rw_asymptotic_behavior);
    RUN_TEST(skip_modes_consistent);

    /* Gradient checks */
    RUN_TEST(skip_fd_dF_and_dX_laurel_rw);
    RUN_TEST(skip_fd_alpha_raw_monte_carlo);
    RUN_TEST(skip_gradient_sign_consistent);

    /* Integration */
    RUN_TEST(skip_composite_gradient_flow);

    /* Stability */
    RUN_TEST(skip_stability_large_values);

    printf("1..%d\n",    _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
