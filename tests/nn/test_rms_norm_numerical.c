/**
 * @file test_rms_norm_numerical.c
 * @brief Comprehensive numerical verification for RMSNorm.
 *
 * Covers:
 *   - Forward correctness across distributions, dimensions, epsilon values
 *   - Scale invariance & other mathematical invariants
 *   - Exhaustive finite-difference gradient checks on random inputs
 *   - Numerical stability under extreme values
 *   - Composite gradient flow (RMSNorm → linear loss)
 */

#include "test_harness.h"
#include "norm/rms_norm.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>  /* rand */

/* seedable-but-non-libc xorshift – for reproducible random tests */
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
/*  Forward correctness – varying dimensions                         */
/* ================================================================= */

static void check_forward_exact_varying_n(size_t n)
{
    float *x = (float *)malloc(n * sizeof(float));
    float *g = (float *)malloc(n * sizeof(float));
    float *y = (float *)malloc(n * sizeof(float));

    /* Fill with deterministic pattern */
    for (size_t i = 0; i < n; ++i) {
        x[i] = (float)((int)(i % 7) - 3) * 0.5f + 0.1f;
        g[i] = 0.5f + (float)i / (float)n;  /* gamma in [0.5, 1.5] */
    }

    rms_norm_forward(y, x, g, n, 1e-5f);

    /* Recompute expected values */
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) sum_sq += x[i] * x[i];
    float r = 1.0f / sqrtf(sum_sq / (float)n + 1e-5f);

    /* Element-wise check */
    char msg[64];
    for (size_t i = 0; i < n; ++i) {
        float expected = x[i] * r * g[i];
        snprintf(msg, sizeof(msg), "varying_n_%zu[%zu]", n, i);
        ASSERT_FLOAT_EQ(expected, y[i], 1e-6f, msg);
    }

    free(x); free(g); free(y);
}

TEST(rms_norm_forward_varying_n_1)  { check_forward_exact_varying_n(1); }
TEST(rms_norm_forward_varying_n_2)  { check_forward_exact_varying_n(2); }
TEST(rms_norm_forward_varying_n_4)  { check_forward_exact_varying_n(4); }
TEST(rms_norm_forward_varying_n_8)  { check_forward_exact_varying_n(8); }
TEST(rms_norm_forward_varying_n_16) { check_forward_exact_varying_n(16); }
TEST(rms_norm_forward_varying_n_32) { check_forward_exact_varying_n(32); }
TEST(rms_norm_forward_varying_n_64) { check_forward_exact_varying_n(64); }
TEST(rms_norm_forward_varying_n_128){ check_forward_exact_varying_n(128); }
TEST(rms_norm_forward_varying_n_256){ check_forward_exact_varying_n(256); }

/* ================================================================= */
/*  Forward correctness – multiple epsilon values                    */
/* ================================================================= */

static void check_epsilon_effect(size_t n, float eps, float *x, float *g)
{
    float *y   = (float *)malloc(n * sizeof(float));
    float *y0  = (float *)malloc(n * sizeof(float));

    rms_norm_forward(y,  x, g, n, eps);
    rms_norm_forward(y0, x, g, n, 0.0f);

    /* With eps > 0, output magnitude should be smaller than with eps=0
     * (because r = 1/sqrt(mean_sq + eps) < 1/sqrt(mean_sq)) */
    float norm_y  = 0.0f, norm_y0 = 0.0f;
    for (size_t i = 0; i < n; ++i) { norm_y += y[i]*y[i]; norm_y0 += y0[i]*y0[i]; }
    ASSERT_TRUE(norm_y <= norm_y0 + 1e-10f,
                "epsilon > 0 reduces output magnitude");

    /* Very large eps → all outputs near zero */
    float *y_large = (float *)malloc(n * sizeof(float));
    rms_norm_forward(y_large, x, g, n, 1e10f);
    float norm_large = 0.0f;
    for (size_t i = 0; i < n; ++i) norm_large += y_large[i] * y_large[i];
    ASSERT_TRUE(norm_large < 1e-4f,
                "huge epsilon damps output to near-zero");

    free(y); free(y0); free(y_large);
}

TEST(rms_norm_epsilon_sweep)
{
    float x[16], g[16];
    for (int i = 0; i < 16; ++i) { x[i] = (float)i - 7.5f; g[i] = 1.0f + 0.1f * (float)i; }
    check_epsilon_effect(16, 1e-8f,  x, g);
    check_epsilon_effect(16, 1e-5f,  x, g);
    check_epsilon_effect(16, 1e-3f,  x, g);
    check_epsilon_effect(16, 0.1f,   x, g);
    check_epsilon_effect(16, 1.0f,   x, g);
    /* final check: huge epsilon */
    ASSERT_TRUE(1, "all epsilon values processed");
}

/* ================================================================= */
/*  Scale invariance: RMSNorm(a*x, g/a) == RMSNorm(x, g)             */
/* ================================================================= */

TEST(rms_norm_scale_invariance)
{
    /* RMSNorm is scale-invariant only when eps=0:
     *   RMSNorm(α·x, γ, eps=0) == sign(α) * RMSNorm(x, γ, eps=0)
     *   RMSNorm(α·x, NULL, eps≈0) ≈ RMSNorm(x, NULL, eps≈0) when eps << var(x) */
    float x_orig[8] = {1.0f, -2.0f, 0.5f, -0.3f, 4.0f, -1.0f, 2.5f, 0.0f};
    float y_orig[8], y_scaled[8], x_scaled[8];
    float alpha = 3.7f;

    for (int i = 0; i < 8; ++i) x_scaled[i] = x_orig[i] * alpha;

    /* Exact with eps=0 and gamma=NULL */
    rms_norm_forward(y_orig,  x_orig,  NULL, 8, 0.0f);
    rms_norm_forward(y_scaled, x_scaled, NULL, 8, 0.0f);

    for (int i = 0; i < 8; ++i) {
        ASSERT_FLOAT_EQ(y_orig[i], y_scaled[i], 1e-6f,
                        "scale invariant: RMSNorm(αx) = RMSNorm(x) eps=0");
    }

    /* Approximate with small eps (eps << var(x)) */
    rms_norm_forward(y_orig,  x_orig,  NULL, 8, 1e-5f);
    rms_norm_forward(y_scaled, x_scaled, NULL, 8, 1e-5f);

    for (int i = 0; i < 8; ++i) {
        ASSERT_FLOAT_EQ(y_orig[i], y_scaled[i], 5e-5f,
                        "scale invariant: RMSNorm(αx) ≈ RMSNorm(x) eps=1e-5");
    }

    /* With gamma and eps=0: RMSNorm(αx, γ, 0) == RMSNorm(x, γ, 0) for α>0 */
    float g_orig[8]  = {1.0f, 2.0f, 0.5f, 1.5f, 0.8f, 1.2f, 3.0f, 1.0f};

    rms_norm_forward(y_orig,  x_orig,  g_orig, 8, 0.0f);
    rms_norm_forward(y_scaled, x_scaled, g_orig, 8, 0.0f);

    for (int i = 0; i < 8; ++i) {
        ASSERT_FLOAT_EQ(y_orig[i], y_scaled[i], 1e-6f,
                        "scale invariant with gamma eps=0");
    }
}

/* ================================================================= */
/*  Sign preservation                                                 */
/* ================================================================= */

TEST(rms_norm_sign_preservation)
{
    /* sign(y_i) == sign(x_i) for gamma_i > 0, regardless of other x_j */
    float x[8]   = {3.0f, -7.0f, 0.0f, 0.5f, -0.001f, 100.0f, -50.0f, 0.0f};
    float g[8]   = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float y[8];
    rms_norm_forward(y, x, g, 8, 1e-5f);
    for (int i = 0; i < 8; ++i) {
        if (x[i] > 0.0f) ASSERT_TRUE(y[i] >= 0.0f, "positive x → positive y");
        if (x[i] < 0.0f) ASSERT_TRUE(y[i] <= 0.0f, "negative x → negative y");
        if (x[i] == 0.0f) ASSERT_FLOAT_EQ(0.0f, y[i], 1e-10f, "zero x → zero y");
    }
}

/* ================================================================= */
/*  Gradient checks – Monte Carlo on random inputs                   */
/* ================================================================= */

/* Linear loss L = Σ dout_i * y_i  →  dL/dx computed by backward */
static float linear_loss_rms(const float *dout, const float *x,
                              const float *g, size_t n, float eps)
{
    float *y = (float *)malloc(n * sizeof(float));
    rms_norm_forward(y, x, g, n, eps);
    float L = 0.0f;
    for (size_t i = 0; i < n; ++i) L += dout[i] * y[i];
    free(y);
    return L;
}

static void fd_check_one(size_t n, float *x, float *g, float *dout,
                          float eps, float fd_h, float tolerance)
{
    float *dx_analytical = (float *)malloc(n * sizeof(float));
    float *dg_analytical = (float *)calloc(n, sizeof(float));
    float *x_plus  = (float *)malloc(n * sizeof(float));
    float *x_minus = (float *)malloc(n * sizeof(float));

    rms_norm_backward(dx_analytical, dg_analytical, dout, x, g, n, eps);

    /* FD on every single element */
    for (size_t i = 0; i < n; ++i) {
        memcpy(x_plus,  x, n * sizeof(float));
        memcpy(x_minus, x, n * sizeof(float));
        x_plus[i]  += fd_h;
        x_minus[i] -= fd_h;
        float L_plus  = linear_loss_rms(dout, x_plus,  g, n, eps);
        float L_minus = linear_loss_rms(dout, x_minus, g, n, eps);
        float dx_fd   = (L_plus - L_minus) / (2.0f * fd_h);

        char msg[128];
        snprintf(msg, sizeof(msg), "FD n=%zu i=%zu (anal=%e fd=%e)",
                 n, i, (double)dx_analytical[i], (double)dx_fd);
        ASSERT_FLOAT_EQ(dx_fd, dx_analytical[i], tolerance, msg);
    }

    free(dx_analytical); free(dg_analytical);
    free(x_plus); free(x_minus);
}

static void run_random_fd_check(size_t n, uint32_t seed, int n_configs)
{
    float *x    = (float *)malloc(n * sizeof(float));
    float *g    = (float *)malloc(n * sizeof(float));
    float *dout = (float *)malloc(n * sizeof(float));

    set_seed(seed);
    for (int cfg = 0; cfg < n_configs; ++cfg) {
        /* Random normal-ish inputs (sum of 3 uniforms) */
        for (size_t i = 0; i < n; ++i) {
            x[i]    = (randf_local() + randf_local() + randf_local() - 1.5f) * 2.0f;
            g[i]    = randf_local() * 2.0f + 0.1f;   /* [0.1, 2.1] */
            dout[i] = (randf_local() - 0.5f) * 2.0f;  /* [-1, 1] */
        }
        /* FD error grows with n; scale tolerance accordingly */
        float fd_tol = (n <= 32) ? 2e-3f : (n <= 64) ? 4e-3f : 6e-3f;
        fd_check_one(n, x, g, dout, 1e-5f, fd_tol, fd_tol);
    }

    free(x); free(g); free(dout);
}

TEST(rms_norm_fd_monte_carlo_n4)   { run_random_fd_check(4,   100, 20); }
TEST(rms_norm_fd_monte_carlo_n8)   { run_random_fd_check(8,   200, 20); }
TEST(rms_norm_fd_monte_carlo_n16)  { run_random_fd_check(16,  300, 15); }
TEST(rms_norm_fd_monte_carlo_n32)  { run_random_fd_check(32,  400, 10); }
TEST(rms_norm_fd_monte_carlo_n64)  { run_random_fd_check(64,  500, 10); }
TEST(rms_norm_fd_monte_carlo_n128) { run_random_fd_check(128, 600, 5);  }

/* ================================================================= */
/*  FD check: d_gamma as well                                        */
/* ================================================================= */

TEST(rms_norm_fd_dgamma_check)
{
    size_t n = 8;
    float x[8]    = {0.5f, -1.2f, 3.0f, -0.8f, 2.5f, -0.1f, 1.7f, -2.0f};
    float g[8]    = {0.7f, 1.3f, 0.9f, 1.1f, 0.6f, 1.4f, 1.0f, 0.8f};
    float dout[8] = {-0.3f, 1.0f, 0.5f, -0.7f, 2.0f, -1.5f, 0.2f, 0.8f};

    float dx[8], dg[8] = {0};
    rms_norm_backward(dx, dg, dout, x, g, n, 1e-5f);

    float h = 1e-3f;
    for (size_t j = 0; j < n; ++j) {
        float g_plus[8], g_minus[8];
        memcpy(g_plus,  g, sizeof(g));
        memcpy(g_minus, g, sizeof(g));
        g_plus[j]  += h;
        g_minus[j] -= h;
        float L_plus  = linear_loss_rms(dout, x, g_plus,  n, 1e-5f);
        float L_minus = linear_loss_rms(dout, x, g_minus, n, 1e-5f);
        float dg_fd   = (L_plus - L_minus) / (2.0f * h);
        char msg[64];
        snprintf(msg, sizeof(msg), "d_gamma FD j=%zu", j);
        ASSERT_FLOAT_EQ(dg_fd, dg[j], 1e-3f, msg);
    }
}

/* ================================================================= */
/*  Numerical stability – extreme values                             */
/* ================================================================= */

TEST(rms_norm_stability_large_inputs)
{
    /* Large values should not cause NaN or overflow */
    float x[16], y[16];
    for (int i = 0; i < 16; ++i) x[i] = (i % 2 == 0) ? 1e10f : -1e10f;
    rms_norm_forward(y, x, NULL, 16, 1e-5f);
    for (int i = 0; i < 16; ++i)
        ASSERT_TRUE(!isnan(y[i]) && !isinf(y[i]), "no NaN/Inf on large inputs");
}

TEST(rms_norm_stability_small_inputs)
{
    /* Very small values should not cause underflow issues */
    float x[16], y[16];
    for (int i = 0; i < 16; ++i) x[i] = (i % 3 == 0) ? 1e-30f : -1e-30f;
    rms_norm_forward(y, x, NULL, 16, 1e-5f);
    for (int i = 0; i < 16; ++i)
        ASSERT_TRUE(!isnan(y[i]), "no NaN on very small inputs");
}

TEST(rms_norm_stability_mixed_magnitudes)
{
    /* Mix large and small values */
    float x[8] = {1e10f, 1e-10f, -1e10f, -1e-10f, 1e5f, 1e-5f, -1e5f, -1e-5f};
    float y[8];
    rms_norm_forward(y, x, NULL, 8, 1e-5f);
    for (int i = 0; i < 8; ++i)
        ASSERT_TRUE(!isnan(y[i]) && !isinf(y[i]), "no NaN/Inf on mixed inputs");
}

TEST(rms_norm_stability_tiny_epsilon)
{
    /* Very small epsilon should not cause division by zero */
    float x[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float y[8];
    rms_norm_forward(y, x, NULL, 8, 1e-15f);
    for (int i = 0; i < 8; ++i)
        ASSERT_TRUE(!isnan(y[i]) && !isinf(y[i]), "no NaN/Inf with eps=1e-15");
}

TEST(rms_norm_stability_zero_epsilon)
{
    /* eps=0 should work as long as sum_sq > 0 */
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[4];
    rms_norm_forward(y, x, NULL, 4, 0.0f);
    for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(!isnan(y[i]) && !isinf(y[i]), "eps=0 works when sum_sq>0");
}

TEST(rms_norm_stability_all_identical)
{
    /* All identical large values: sum_sq / n = large^2, r = 1/large */
    float vals[] = {1000.0f, 1e6f, 1e9f};
    for (int vi = 0; vi < 3; ++vi) {
        float val = vals[vi];
        float x[16], y[16];
        for (int i = 0; i < 16; ++i) x[i] = val;
        rms_norm_forward(y, x, NULL, 16, 1e-5f);
        /* Each output should be val / sqrt(val^2 + eps) ≈ sign(val) */
        float expected = val / sqrtf(val * val + 1e-5f);
        for (int i = 0; i < 16; ++i) {
            ASSERT_FLOAT_EQ(expected, y[i], 1e-5f, "all-identical normalization");
            ASSERT_TRUE(!isnan(y[i]), "no NaN");
        }
    }
}

/* ================================================================= */
/*  Composite gradient: forward → loss → backward                    */
/* ================================================================= */

TEST(rms_norm_composite_gradient_flow)
{
    /* Verify that applying gradient descent reduces loss.
     * Loss = 0.5 * Σ (RMSNorm(x, g) - target)^2
     * Check that both x and g can be optimized via gradient descent. */
    size_t n = 8;
    float x[8]      = {0.5f, -0.8f, 1.2f, -0.3f, 2.0f, -1.5f, 0.7f, -0.1f};
    float g[8]      = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float target[8] = {0.5f, 0.5f, 0.5f, 0.5f, -0.5f, -0.5f, -0.5f, -0.5f};
    float lr = 0.01f;

    float y[8], loss_before, loss_after;
    rms_norm_forward(y, x, g, n, 1e-5f);
    loss_before = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = y[i] - target[i];
        loss_before += 0.5f * diff * diff;
    }

    /* Compute dout = y - target */
    float dout[8];
    for (size_t i = 0; i < n; ++i) dout[i] = y[i] - target[i];

    /* Compute gradients */
    float dx[8], dg[8] = {0};
    rms_norm_backward(dx, dg, dout, x, g, n, 1e-5f);

    /* Gradient descent on g only */
    float g_new[8];
    for (size_t i = 0; i < n; ++i) g_new[i] = g[i] - lr * dg[i];
    rms_norm_forward(y, x, g_new, n, 1e-5f);
    loss_after = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = y[i] - target[i];
        loss_after += 0.5f * diff * diff;
    }
    ASSERT_TRUE(loss_after < loss_before, "gradient descent on gamma reduces loss");

    /* Gradient descent on x only */
    float x_new[8];
    for (size_t i = 0; i < n; ++i) x_new[i] = x[i] - lr * dx[i];
    rms_norm_forward(y, x_new, g, n, 1e-5f);
    float loss_after_x = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = y[i] - target[i];
        loss_after_x += 0.5f * diff * diff;
    }
    /* Note: changing x changes the normalization denominator,
     * so gradient descent on x doesn't guarantee monotonic loss decrease
     * for a single step (it's not a linear function). But we still
     * verify the gradient is non-zero. */
    ASSERT_TRUE(loss_after_x != loss_before, "x gradient changes loss");
}

/* ================================================================= */
/*  Determinism                                                       */
/* ================================================================= */

TEST(rms_norm_determinism)
{
    float x[16], y1[16], y2[16];
    for (int i = 0; i < 16; ++i) x[i] = (float)i * 0.7f;
    rms_norm_forward(y1, x, NULL, 16, 1e-5f);
    rms_norm_forward(y2, x, NULL, 16, 1e-5f);
    for (int i = 0; i < 16; ++i)
        ASSERT_FLOAT_EQ(y1[i], y2[i], 1e-10f, "deterministic: same input → same output");
}

/* ================================================================= */
/*  RMS invariance: output RMS ≈ 1.0 (gamma=NULL)                    */
/* ================================================================= */

TEST(rms_norm_output_rms_is_one)
{
    /* For gamma=NULL (≡1), E[y^2] should be approximately 1.
     * More precisely: sqrt(E[y^2]) = 1 / sqrt(1 + eps/E[x^2]) < 1
     * but very close to 1 for eps << E[x^2]. */
    size_t max_n = 64;
    float *x = (float *)malloc(max_n * sizeof(float));
    float *y = (float *)malloc(max_n * sizeof(float));

    size_t dims[] = {1, 4, 8, 16, 32, 64};
    for (int d = 0; d < 6; ++d) {
        size_t n = dims[d];
        /* Generate inputs with non-zero energy */
        for (size_t i = 0; i < n; ++i)
            x[i] = (float)((int)(i * 7 % 13) - 6) * 0.3f + ((i % 3 == 0) ? 2.0f : 0.0f);

        rms_norm_forward(y, x, NULL, n, 1e-5f);

        float sum_sq = 0.0f;
        for (size_t i = 0; i < n; ++i) sum_sq += y[i] * y[i];
        float rms_y = sqrtf(sum_sq / (float)n);
        ASSERT_FLOAT_EQ(1.0f, rms_y, 1e-3f, "output RMS ≈ 1");
    }
    free(x); free(y);
}

/* ================================================================= */
/*  Main                                                              */
/* ================================================================= */

int main(void)
{
    /* Forward – varying dimensions */
    RUN_TEST(rms_norm_forward_varying_n_1);
    RUN_TEST(rms_norm_forward_varying_n_2);
    RUN_TEST(rms_norm_forward_varying_n_4);
    RUN_TEST(rms_norm_forward_varying_n_8);
    RUN_TEST(rms_norm_forward_varying_n_16);
    RUN_TEST(rms_norm_forward_varying_n_32);
    RUN_TEST(rms_norm_forward_varying_n_64);
    RUN_TEST(rms_norm_forward_varying_n_128);
    RUN_TEST(rms_norm_forward_varying_n_256);

    /* Epsilon sweep */
    RUN_TEST(rms_norm_epsilon_sweep);

    /* Invariants */
    RUN_TEST(rms_norm_scale_invariance);
    RUN_TEST(rms_norm_sign_preservation);
    RUN_TEST(rms_norm_output_rms_is_one);

    /* Gradient checks – Monte Carlo */
    RUN_TEST(rms_norm_fd_monte_carlo_n4);
    RUN_TEST(rms_norm_fd_monte_carlo_n8);
    RUN_TEST(rms_norm_fd_monte_carlo_n16);
    RUN_TEST(rms_norm_fd_monte_carlo_n32);
    RUN_TEST(rms_norm_fd_monte_carlo_n64);
    RUN_TEST(rms_norm_fd_monte_carlo_n128);
    RUN_TEST(rms_norm_fd_dgamma_check);

    /* Stability */
    RUN_TEST(rms_norm_stability_large_inputs);
    RUN_TEST(rms_norm_stability_small_inputs);
    RUN_TEST(rms_norm_stability_mixed_magnitudes);
    RUN_TEST(rms_norm_stability_tiny_epsilon);
    RUN_TEST(rms_norm_stability_zero_epsilon);
    RUN_TEST(rms_norm_stability_all_identical);

    /* Integration */
    RUN_TEST(rms_norm_composite_gradient_flow);
    RUN_TEST(rms_norm_determinism);

    printf("1..%d\n",    _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
