/**
 * @file test_dropout.c
 * @brief Unit tests for Dropout and Stochastic Depth (DropPath).
 */

#include "test_harness.h"
#include "dropout/dropout.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

TEST(dropout_init_sets_fields)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.7f, 42);
    ASSERT_FLOAT_EQ(0.7f, dp.keep_prob, 1e-7f, "keep_prob set");
    ASSERT_EQ_INT(1, dp.training, "defaults to training mode");
    ASSERT_NULL(dp.mask, "mask initially NULL");
    ASSERT_EQ_SIZE(0, dp.mask_capacity, "capacity 0");
    dropout_free(&dp);
}

TEST(dropout_init_clamps_keep_prob)
{
    DropoutLayer dp1, dp2;
    dropout_init(&dp1, 2.0f, 1);
    dropout_init(&dp2, -0.5f, 1);
    ASSERT_FLOAT_EQ(1.0f, dp1.keep_prob, 1e-7f, "clamped to 1.0");
    ASSERT_FLOAT_EQ(0.0f, dp2.keep_prob, 1e-7f, "clamped to 0.0");
    dropout_free(&dp1);
    dropout_free(&dp2);
}

TEST(dropout_set_training)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.5f, 1);
    dropout_set_training(&dp, 0);
    ASSERT_EQ_INT(0, dp.training, "training=0");
    dropout_set_training(&dp, 1);
    ASSERT_EQ_INT(1, dp.training, "training=1");
    dropout_free(&dp);
}

/* ------------------------------------------------------------------ */
/*  Per-element Dropout – inference mode                               */
/* ------------------------------------------------------------------ */

TEST(dropout_inference_passthrough)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.5f, 42);
    dropout_set_training(&dp, 0);

    float x[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[5];
    dropout_forward(y, x, 5, &dp);
    for (int i = 0; i < 5; ++i) {
        ASSERT_FLOAT_EQ(x[i], y[i], 1e-7f, "inference = identity");
    }
    dropout_free(&dp);
}

TEST(dropout_inference_backward_passthrough)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.5f, 42);
    dropout_set_training(&dp, 0);

    float dy[5] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    float dx[5];
    dropout_backward(dx, dy, 5, &dp);
    for (int i = 0; i < 5; ++i) {
        ASSERT_FLOAT_EQ(dy[i], dx[i], 1e-7f, "inference backward = identity");
    }
    dropout_free(&dp);
}

/* ------------------------------------------------------------------ */
/*  Per-element Dropout – keep_prob == 1                               */
/* ------------------------------------------------------------------ */

TEST(dropout_keep_prob_one_passthrough)
{
    DropoutLayer dp;
    dropout_init(&dp, 1.0f, 42);
    float x[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[5];
    dropout_forward(y, x, 5, &dp);
    for (int i = 0; i < 5; ++i) {
        ASSERT_FLOAT_EQ(x[i], y[i], 1e-7f, "keep_prob=1 = identity");
    }
    dropout_free(&dp);
}

/* ------------------------------------------------------------------ */
/*  Per-element Dropout – keep_prob == 0                               */
/* ------------------------------------------------------------------ */

TEST(dropout_keep_prob_zero_all_zeros)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.0f, 42);
    float x[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[5];
    dropout_forward(y, x, 5, &dp);
    for (int i = 0; i < 5; ++i) {
        ASSERT_FLOAT_EQ(0.0f, y[i], 1e-7f, "keep_prob=0 → all zeros");
    }
    dropout_free(&dp);
}

/* ------------------------------------------------------------------ */
/*  Per-element Dropout – statistical properties                       */
/* ------------------------------------------------------------------ */

TEST(dropout_keep_rate_statistical)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.6f, 12345);
    float x[1000];
    float y[1000];
    for (int i = 0; i < 1000; ++i) x[i] = 1.0f;

    dropout_forward(y, x, 1000, &dp);

    /* Count zeros */
    int zeros = 0;
    int kept = 0;
    float sum_kept = 0.0f;
    for (int i = 0; i < 1000; ++i) {
        if (y[i] == 0.0f) {
            zeros++;
        } else {
            kept++;
            sum_kept += y[i];
        }
    }

    float keep_rate = (float)kept / 1000.0f;
    ASSERT_TRUE(fabsf(keep_rate - 0.6f) < 0.05f, "keep rate ≈ 0.6 (±5%)");

    /* Kept values should be 1/keep_prob = 1/0.6 ≈ 1.667 */
    float mean_kept = sum_kept / (float)kept;
    ASSERT_FLOAT_EQ(1.0f / 0.6f, mean_kept, 0.01f, "kept elements scaled by 1/0.6");

    dropout_free(&dp);
}

TEST(dropout_deterministic_with_same_seed)
{
    DropoutLayer dp1, dp2;
    dropout_init(&dp1, 0.5f, 42);
    dropout_init(&dp2, 0.5f, 42);

    float x[20], y1[20], y2[20];
    for (int i = 0; i < 20; ++i) x[i] = (float)i;

    dropout_forward(y1, x, 20, &dp1);
    dropout_forward(y2, x, 20, &dp2);
    for (int i = 0; i < 20; ++i) {
        ASSERT_FLOAT_EQ(y1[i], y2[i], 1e-7f, "same seed → same mask");
    }
    dropout_free(&dp1);
    dropout_free(&dp2);
}

/* ------------------------------------------------------------------ */
/*  Per-element Dropout – backward                                     */
/* ------------------------------------------------------------------ */

TEST(dropout_backward_reproduces_mask)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.5f, 42);
    float x[100], y[100];
    for (int i = 0; i < 100; ++i) x[i] = 1.0f;

    dropout_forward(y, x, 100, &dp);

    /* Backward with dy = 1: dx[i] should be 1/keep_prob where mask[i]=1, else 0 */
    float dy[100], dx[100];
    for (int i = 0; i < 100; ++i) dy[i] = 1.0f;
    dropout_backward(dx, dy, 100, &dp);

    for (int i = 0; i < 100; ++i) {
        if (y[i] > 0.0f) {
            ASSERT_FLOAT_EQ(1.0f / 0.5f, dx[i], 1e-7f, "kept → dy/0.5");
        } else {
            ASSERT_FLOAT_EQ(0.0f, dx[i], 1e-7f, "dropped → 0");
        }
    }
    dropout_free(&dp);
}

/* ------------------------------------------------------------------ */
/*  DropPath – inference mode                                          */
/* ------------------------------------------------------------------ */

TEST(droppath_inference_passthrough)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.7f, 42);
    dropout_set_training(&dp, 0);

    float fx[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[4]  = {0.1f, 0.2f, 0.3f, 0.4f};
    float y[4];
    droppath_forward(y, fx, x, 4, &dp);
    ASSERT_EQ_INT(1, dp.drop_path_flag, "inference sets flag=1");
    for (int i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(fx[i] + x[i], y[i], 1e-7f, "inference: y = fx + x");
    }
    dropout_free(&dp);
}

TEST(droppath_inference_backward_full)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.7f, 42);
    dropout_set_training(&dp, 0);

    float dy[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float d_fx[4], dx[4];
    droppath_backward(d_fx, dx, dy, 4, &dp);
    for (int i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(dy[i], d_fx[i], 1e-7f, "d_fx = dy");
        ASSERT_FLOAT_EQ(dy[i], dx[i], 1e-7f, "dx = dy");
    }
    dropout_free(&dp);
}

/* ------------------------------------------------------------------ */
/*  DropPath – always keep (keep_prob == 1)                            */
/* ------------------------------------------------------------------ */

TEST(droppath_keep_prob_one)
{
    DropoutLayer dp;
    dropout_init(&dp, 1.0f, 42);

    float fx[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[4]  = {0.1f, 0.2f, 0.3f, 0.4f};
    float y[4];
    droppath_forward(y, fx, x, 4, &dp);
    ASSERT_EQ_INT(1, dp.drop_path_flag, "flag=1");
    float scale = 1.0f / 1.0f;  /* = 1 */
    for (int i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(fx[i] * scale + x[i], y[i], 1e-7f, "y = fx + x");
    }
    dropout_free(&dp);
}

/* ------------------------------------------------------------------ */
/*  DropPath – always drop (keep_prob == 0)                            */
/* ------------------------------------------------------------------ */

TEST(droppath_keep_prob_zero)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.0f, 42);

    float fx[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[4]  = {0.1f, 0.2f, 0.3f, 0.4f};
    float y[4];
    droppath_forward(y, fx, x, 4, &dp);
    ASSERT_EQ_INT(0, dp.drop_path_flag, "flag=0");
    for (int i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(x[i], y[i], 1e-7f, "path dropped → y = x");
    }
    dropout_free(&dp);
}

TEST(droppath_keep_prob_zero_backward)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.0f, 42);
    dp.drop_path_flag = 0;

    float dy[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float d_fx[4], dx[4];
    droppath_backward(d_fx, dx, dy, 4, &dp);
    for (int i = 0; i < 4; ++i) {
        ASSERT_FLOAT_EQ(0.0f, d_fx[i], 1e-7f, "dropped → d_fx = 0");
        ASSERT_FLOAT_EQ(dy[i], dx[i], 1e-7f, "dx = dy");
    }
    dropout_free(&dp);
}

/* ------------------------------------------------------------------ */
/*  DropPath – training stochastic behaviour                           */
/* ------------------------------------------------------------------ */

TEST(droppath_path_rate_statistical)
{
    /* With keep_prob=0.5, over many trials ~50% should be kept */
    DropoutLayer dp;
    dropout_init(&dp, 0.5f, 12345);

    float fx[1] = {1.0f}, x[1] = {0.0f}, y[1];
    int kept = 0, dropped = 0;
    for (int trial = 0; trial < 1000; ++trial) {
        droppath_forward(y, fx, x, 1, &dp);
        if (dp.drop_path_flag) kept++; else dropped++;
    }
    float keep_rate = (float)kept / 1000.0f;
    ASSERT_TRUE(fabsf(keep_rate - 0.5f) < 0.05f, "DropPath keep rate ≈ 0.5 (±5%)");
    dropout_free(&dp);
}

TEST(droppath_backward_when_kept)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.5f, 42);
    /* Simulate a forward where the path was kept */
    dp.drop_path_flag = 1;

    float dy[3] = {1.0f, 2.0f, 3.0f};
    float d_fx[3], dx[3];
    droppath_backward(d_fx, dx, dy, 3, &dp);

    float scale = 1.0f / 0.5f;
    for (int i = 0; i < 3; ++i) {
        ASSERT_FLOAT_EQ(dy[i] * scale, d_fx[i], 1e-7f, "d_fx = dy * 1/p");
        ASSERT_FLOAT_EQ(dy[i], dx[i], 1e-7f, "dx = dy");
    }
    dropout_free(&dp);
}

/* ------------------------------------------------------------------ */
/*  Edge cases                                                         */
/* ------------------------------------------------------------------ */

TEST(dropout_n_zero_does_not_crash)
{
    DropoutLayer dp;
    dropout_init(&dp, 0.5f, 42);
    float x[1] = {1.0f}, y[1];
    dropout_forward(y, x, 0, &dp);  /* n=0 */
    dropout_backward(y, x, 0, &dp);
    droppath_forward(y, x, x, 0, &dp);
    droppath_backward(y, y, x, 0, &dp);
    ASSERT_TRUE(1, "did not crash on n=0");
    dropout_free(&dp);
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(void)
{
    RUN_TEST(dropout_init_sets_fields);
    RUN_TEST(dropout_init_clamps_keep_prob);
    RUN_TEST(dropout_set_training);

    RUN_TEST(dropout_inference_passthrough);
    RUN_TEST(dropout_inference_backward_passthrough);
    RUN_TEST(dropout_keep_prob_one_passthrough);
    RUN_TEST(dropout_keep_prob_zero_all_zeros);
    RUN_TEST(dropout_keep_rate_statistical);
    RUN_TEST(dropout_deterministic_with_same_seed);
    RUN_TEST(dropout_backward_reproduces_mask);

    RUN_TEST(droppath_inference_passthrough);
    RUN_TEST(droppath_inference_backward_full);
    RUN_TEST(droppath_keep_prob_one);
    RUN_TEST(droppath_keep_prob_zero);
    RUN_TEST(droppath_keep_prob_zero_backward);
    RUN_TEST(droppath_path_rate_statistical);
    RUN_TEST(droppath_backward_when_kept);

    RUN_TEST(dropout_n_zero_does_not_crash);

    printf("1..%d\n",    _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
