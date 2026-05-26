/**
 * @brief Unit test for Bug #5: BN forward/backward running stats consistency.
 *
 * Verifies that the backward pass reads cached x_hat and inv_std from
 * bn_pre_cache / bn_spatial_var — NOT from post-EMA running stats.
 *
 * Strategy:
 *   Run forward+backward on one sample.  Forward normalizes with running
 *   stats at (mean=0, var=1), then EMA-updates running stats.  If backward
 *   reads running stats, bn_gamma_grad uses x_hat_reconstructed_with_new_stats.
 *   If backward reads cache, bn_gamma_grad uses x_hat_cached.
 *   Since post-EMA stats differ from initial values, these two x_hat values
 *   differ — and bn_gamma_grad tells us which one backward actually used.
 */
#include "../src/nn/types/cnn/cnn_config.h"
#include "../src/nn/types/cnn/cnn_infer_ops.h"
#include "../src/nn/types/cnn/cnn_train_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMG_H   6
#define IMG_W   6
#define IMG_C   1
#define FILTERS 2
#define KERNEL  3

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
} while(0)

#define CHECK_FEQ(a, b, tol, msg) do { \
    if (fabsf((float)(a) - (float)(b)) > (tol)) { \
        fprintf(stderr, "FAIL: %s (%.6f vs %.6f)\n", msg, (double)(a), (double)(b)); \
        failures++; \
    } \
} while(0)

int main(void) {
    CnnConfig config;
    CnnTrainConfig train_cfg;
    CnnInferContext* infer_ctx;
    CnnTrainContext* train_ctx;
    float* img;
    float* target;

    memset(&config, 0, sizeof(config));
    memset(&train_cfg, 0, sizeof(train_cfg));

    config.filter_count       = FILTERS;
    config.kernel_size        = KERNEL;
    config.channel_count      = IMG_C;
    config.frame_width        = IMG_W;
    config.frame_height       = IMG_H;
    config.sequence_length    = 1U;
    config.feature_size       = FILTERS;
    config.pooling_mode       = CNN_POOL_AVG;
    config.conv_mode          = CNN_CONV_STANDARD;
    config.stride             = 1U;
    config.use_batch_norm     = 1;
    config.bn_momentum        = 0.9f;
    config.bn_epsilon         = 1e-5f;
    config.output_activation  = CNN_ACT_NONE;
    config.pooling_activation = CNN_ACT_NONE;
    config.total_input_size   = IMG_H * IMG_W * IMG_C;

    infer_ctx = nn_cnn_infer_create_with_config(&config, 12345U);
    CHECK(infer_ctx != NULL, "infer_create returned NULL");

    /* Set BN params to known values for clean math */
    infer_ctx->bn_gamma[0] = 1.0f;
    infer_ctx->bn_gamma[1] = 1.0f;
    infer_ctx->bn_beta[0]  = 0.0f;
    infer_ctx->bn_beta[1]  = 0.0f;
    infer_ctx->bn_running_mean[0] = 0.0f;
    infer_ctx->bn_running_mean[1] = 0.0f;
    infer_ctx->bn_running_var[0]  = 1.0f;
    infer_ctx->bn_running_var[1]  = 1.0f;

    train_cfg.learning_rate   = 0.1f;
    train_cfg.momentum        = 0.0f;
    train_cfg.weight_decay    = 0.0f;
    train_cfg.batch_size      = 1U;
    train_cfg.seed            = 12345U;
    train_cfg.debug_level     = 0;        /* quiet for test output */
    train_cfg.debug_layer_index = 0;

    train_ctx = nn_cnn_train_create(infer_ctx, &train_cfg);
    CHECK(train_ctx != NULL, "train_create returned NULL");
    CHECK(train_ctx->bn_pre_cache != NULL, "bn_pre_cache not allocated");
    CHECK(train_ctx->bn_spatial_var != NULL, "bn_spatial_var not allocated");

    img    = malloc(IMG_C * IMG_H * IMG_W * sizeof(float));
    target = malloc(FILTERS * sizeof(float));
    for (size_t i = 0; i < IMG_C * IMG_H * IMG_W; i++)
        img[i] = sinf((float)i * 0.7f) * 0.8f + 0.3f;
    target[0] = 0.2f;
    target[1] = 0.8f;

    /* Run one forward+backward step */
    nn_cnn_train_step_with_data(train_ctx, img, target);

    /* ────────────────────────────────────────────────────────
     * TEST 1:  Cached x_hat matches forward's pre-EMA computation
     * ──────────────────────────────────────────────────────── */
    printf("=== Test 1: Cache stores pre-EMA x_hat and inv_std ===\n");
    for (int f = 0; f < FILTERS; f++) {
        float raw = train_ctx->pooled_linear_cache[f];
        float inv_std_expected = 1.0f / sqrtf(1.0f + 1e-5f);
        float x_hat_expected = (raw - 0.0f) * inv_std_expected;
        float x_hat_cached = train_ctx->bn_pre_cache[f];
        float inv_std_cached = train_ctx->bn_spatial_var[f];

        printf("  filter %d: raw=%.6f  x_hat(expected)=%.6f  x_hat(cached)=%.6f  inv_std(cached)=%.6f\n",
               f, raw, x_hat_expected, x_hat_cached, inv_std_cached);

        CHECK_FEQ(x_hat_cached, x_hat_expected, 1e-4f, "cached x_hat != pre-EMA computation");
        CHECK_FEQ(inv_std_cached, inv_std_expected, 1e-4f, "cached inv_std != pre-EMA computation");
    }

    /* ────────────────────────────────────────────────────────
     * TEST 2:  Running stats CHANGED (EMA ran AFTER caching)
     * ──────────────────────────────────────────────────────── */
    printf("\n=== Test 2: Running stats updated by EMA ===\n");
    for (int f = 0; f < FILTERS; f++) {
        float mean = infer_ctx->bn_running_mean[f];
        float var  = infer_ctx->bn_running_var[f];
        printf("  filter %d: mean=%.6f (was 0.0)  var=%.6f (was 1.0)\n", f, mean, var);
        CHECK(fabsf(mean) > 1e-6f, "running_mean not updated by EMA");
        CHECK(var < 0.999f, "running_var not updated by EMA");
    }

    /* ────────────────────────────────────────────────────────
     * TEST 3:  Backward gradients used CACHED x_hat, not RUNNING-based x_hat
     *
     * If backward read running stats:
     *   x_hat_running = (raw - post_mean) * (1/sqrt(post_var+eps))
     *   bn_gamma_grad_running = dpool_act * x_hat_running
     *
     * If backward read cache:
     *   bn_gamma_grad_cached = dpool_act * x_hat_cached
     *
     * We verify: bn_gamma_grad / dpool_act ≈ cached x_hat (not running-based)
     *
     * We recover dpool_act from: bn_beta_grad = dpool_act
     * (since beta_grad accumulates dpool_act directly)
     * ──────────────────────────────────────────────────────── */
    printf("\n=== Test 3: Backward uses cached x_hat, not running stats ===\n");
    for (int f = 0; f < FILTERS; f++) {
        float raw = train_ctx->pooled_linear_cache[f];
        float post_mean = infer_ctx->bn_running_mean[f];
        float post_var  = infer_ctx->bn_running_var[f];
        float post_inv_std = 1.0f / sqrtf(post_var + 1e-5f);

        float x_hat_cached   = train_ctx->bn_pre_cache[f];
        float x_hat_running  = (raw - post_mean) * post_inv_std;
        float inv_std_cached = train_ctx->bn_spatial_var[f];

        float dpool_act      = train_ctx->bn_beta_grad[f];  /* beta_grad = sum(dpool_act) */
        float gamma_grad     = train_ctx->bn_gamma_grad[f];  /* gamma_grad = sum(dpool_act * x_hat) */

        float x_hat_implied_by_grad = (fabsf(dpool_act) > 1e-6f) ? (gamma_grad / dpool_act) : 0.0f;

        printf("  filter %d:\n", f);
        printf("    raw=%.6f  pre_mean=0.0  post_mean=%.6f  post_var=%.6f\n", raw, post_mean, post_var);
        printf("    x_hat(cached)=%.6f  x_hat(running)=%.6f\n", x_hat_cached, x_hat_running);
        printf("    dpool_act=%.6f  gamma_grad=%.6f  beta_grad=%.6f\n", dpool_act, gamma_grad, train_ctx->bn_beta_grad[f]);
        printf("    implied x_hat = gamma_grad / beta_grad = %.6f\n", x_hat_implied_by_grad);
        printf("    cached inv_std=%.6f  inv_std(running)=%.6f\n", inv_std_cached, post_inv_std);

        /* Verify x_hat_cached ≠ x_hat_running (otherwise test can't distinguish) */
        CHECK(fabsf(x_hat_cached - x_hat_running) > 1e-4f,
              "cached and running x_hat are too close to distinguish");

        /* Verify gamma_grad implies cached x_hat, not running x_hat */
        CHECK_FEQ(x_hat_implied_by_grad, x_hat_cached, 1e-3f,
                  "implied x_hat (gamma_grad/beta_grad) != cached x_hat");

        /* Verify gradients are non-zero (actual learning happened) */
        CHECK(fabsf(gamma_grad) > 1e-6f, "bn_gamma_grad is zero");
        CHECK(fabsf(dpool_act) > 1e-6f, "bn_beta_grad is zero");
    }

    nn_cnn_train_destroy(train_ctx);
    nn_cnn_infer_destroy(infer_ctx);
    free(img);
    free(target);

    if (failures == 0) {
        printf("\n=== ALL TESTS PASSED ===\n");
        printf("Bug #5 fix verified: backward reads cached x_hat/inv_std,\n");
        printf("  NOT post-EMA running stats.\n");
        return 0;
    }
    printf("\n=== %d TEST(S) FAILED ===\n", failures);
    return 1;
}
