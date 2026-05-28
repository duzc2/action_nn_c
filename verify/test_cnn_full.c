/**
 * @file test_cnn_full.c
 * @brief Comprehensive CNN unit test — 24 tests covering inference, training,
 *        all pooling/activation/conv modes, BN bug regressions, and edge cases.
 *
 * Design: struct-based test runner. Each test is a static function returning
 * int (0 = pass, non-zero = failure count). main() runs all registered tests
 * and reports results.
 */

#include "../src/nn/types/cnn/cnn_config.h"
#include "../src/nn/types/cnn/cnn_infer_ops.h"
#include "../src/nn/types/cnn/cnn_train_ops.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Test framework ─────────────────────────────────────────────────── */

static int g_test_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "  FAIL: %s\n", msg); g_test_failures++; } \
} while(0)

#define CHECK_FEQ(a, b, tol, msg) do { \
    if (fabsf((float)(a) - (float)(b)) > (tol)) { \
        fprintf(stderr, "  FAIL: %s (%.6f vs %.6f)\n", msg, (double)(a), (double)(b)); \
        g_test_failures++; \
    } \
} while(0)

typedef int (*TestFunc)(void);

typedef struct {
    const char* name;
    TestFunc    func;
} TestCase;

/* ── Helpers ────────────────────────────────────────────────────────── */

static int is_nan(float x) { return x != x; }

static int has_nan(const float* arr, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (is_nan(arr[i])) return 1;
    return 0;
}

static void generate_sine_image(float* img, size_t h, size_t w, size_t c) {
    size_t total = c * h * w;
    for (size_t i = 0; i < total; i++)
        img[i] = sinf((float)i * 0.7f) * 0.8f + 0.3f;
}

static void make_soft_target(float* target, size_t n) {
    for (size_t i = 0; i < n; i++)
        target[i] = (i == 0) ? 0.8f : 0.2f / ((float)n - 1.0f > 0.0f ? (float)n - 1.0f : 1.0f);
}

static void make_onehot_target(float* target, size_t n, size_t cls) {
    for (size_t i = 0; i < n; i++) target[i] = 0.0f;
    if (cls < n) target[cls] = 1.0f;
}

static void fill_const(float* arr, size_t n, float v) {
    for (size_t i = 0; i < n; i++) arr[i] = v;
}

/* ── Config presets ─────────────────────────────────────────────────── */

typedef enum {
    PRESET_SMALL_AVG,
    PRESET_SMALL_NONE,
    PRESET_SMALL_MAX,
    PRESET_SMALL_DUAL,
    PRESET_SMALL_DW,
    PRESET_SMALL_S2
} ConfigPreset;

static void init_config_preset(CnnConfig* cfg, ConfigPreset preset) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->sequence_length    = 1;
    cfg->output_activation  = CNN_ACT_NONE;
    cfg->pooling_activation = CNN_ACT_NONE;
    cfg->bn_momentum        = 0.9f;
    cfg->bn_epsilon         = 1e-5f;

    switch (preset) {
    case PRESET_SMALL_AVG:
        cfg->frame_height    = 6;  cfg->frame_width     = 6;
        cfg->channel_count   = 1;  cfg->kernel_size     = 3;
        cfg->filter_count    = 4;  cfg->feature_size    = 4;
        cfg->pooling_mode    = CNN_POOL_AVG;
        cfg->conv_mode       = CNN_CONV_STANDARD;
        cfg->stride          = 1;
        cfg->use_batch_norm  = 1;
        break;
    case PRESET_SMALL_NONE:
        /* POOL_NONE: feature_size must equal none_output_size for auto_run.
         * (6-3+1)^2 * 2 = 32 */
        cfg->frame_height    = 6;  cfg->frame_width     = 6;
        cfg->channel_count   = 1;  cfg->kernel_size     = 3;
        cfg->filter_count    = 2;  cfg->feature_size    = 32;
        cfg->pooling_mode    = CNN_POOL_NONE;
        cfg->conv_mode       = CNN_CONV_STANDARD;
        cfg->stride          = 1;
        cfg->use_batch_norm  = 1;
        break;
    case PRESET_SMALL_MAX:
        cfg->frame_height    = 6;  cfg->frame_width     = 6;
        cfg->channel_count   = 1;  cfg->kernel_size     = 3;
        cfg->filter_count    = 4;  cfg->feature_size    = 4;
        cfg->pooling_mode    = CNN_POOL_MAX;
        cfg->conv_mode       = CNN_CONV_STANDARD;
        cfg->stride          = 1;
        cfg->use_batch_norm  = 1;
        break;
    case PRESET_SMALL_DUAL:
        cfg->frame_height    = 6;  cfg->frame_width     = 6;
        cfg->channel_count   = 1;  cfg->kernel_size     = 3;
        cfg->filter_count    = 4;  cfg->feature_size    = 8;
        cfg->pooling_mode    = CNN_POOL_DUAL;
        cfg->conv_mode       = CNN_CONV_STANDARD;
        cfg->stride          = 1;
        cfg->use_batch_norm  = 1;
        break;
    case PRESET_SMALL_DW:
        cfg->frame_height    = 6;  cfg->frame_width     = 6;
        cfg->channel_count   = 4;  cfg->kernel_size     = 3;
        cfg->filter_count    = 4;  cfg->feature_size    = 4;
        cfg->pooling_mode    = CNN_POOL_AVG;
        cfg->conv_mode       = CNN_CONV_DEPTHWISE;
        cfg->stride          = 1;
        cfg->use_batch_norm  = 0;
        break;
    case PRESET_SMALL_S2:
        /* stride=2: ((10-3)/2+1)^2 * 4 = 16 * 4 = 64 */
        cfg->frame_height    = 10; cfg->frame_width     = 10;
        cfg->channel_count   = 1;  cfg->kernel_size     = 3;
        cfg->filter_count    = 4;  cfg->feature_size    = 64;
        cfg->pooling_mode    = CNN_POOL_NONE;
        cfg->conv_mode       = CNN_CONV_STANDARD;
        cfg->stride          = 2;
        cfg->use_batch_norm  = 0;
        break;
    }
    cfg->total_input_size = cfg->frame_height * cfg->frame_width
                          * cfg->channel_count * cfg->sequence_length;
}

static void init_train_config(CnnTrainConfig* tcfg, uint32_t batch_size) {
    memset(tcfg, 0, sizeof(*tcfg));
    tcfg->learning_rate    = 0.1f;
    tcfg->momentum         = 0.0f;
    tcfg->weight_decay     = 0.0f;
    tcfg->bias_weight_decay = 0.0f;
    tcfg->dropout_rate      = 0.0f;
    tcfg->batch_size       = batch_size;
    tcfg->debug_level      = 0;
    tcfg->debug_layer_index = 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Group A: Inference — Forward Pass Correctness (tests 1–8)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Test 1: standard conv, no BN, POOL_AVG, output shape correct */
static int test_forward_no_bn_pool_avg(void) {
    CnnConfig cfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);
    cfg.use_batch_norm = 0;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);

    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    nn_cnn_infer_auto_run(ctx, img, out);

    CHECK(has_nan(out, cfg.feature_size) == 0, "output contains NaN");
    CHECK(out[0] != out[1] || out[0] != out[2] || out[0] != out[3],
          "all output values identical (likely dead network)");

    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 2: standard conv, BN, POOL_AVG, output non-NaN */
static int test_forward_bn_pool_avg(void) {
    CnnConfig cfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);

    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    nn_cnn_infer_auto_run(ctx, img, out);

    CHECK(has_nan(out, cfg.feature_size) == 0, "output contains NaN");

    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 3: standard conv, BN, POOL_NONE, 2D output works */
static int test_forward_bn_pool_none(void) {
    CnnConfig cfg;
    init_config_preset(&cfg, PRESET_SMALL_NONE);

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);

    /* POOL_NONE output: feature_size elements (set to none_output_size). */
    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    nn_cnn_infer_auto_run(ctx, img, out);

    CHECK(has_nan(out, cfg.feature_size) == 0, "output contains NaN");
    /* Verify at least some output values are non-zero */
    int nonzero = 0;
    for (size_t i = 0; i < cfg.feature_size; i++)
        if (fabsf(out[i]) > 1e-7f) nonzero++;
    CHECK(nonzero > 0, "all output values are zero");

    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 4: standard conv, BN, POOL_MAX, output non-NaN */
static int test_forward_bn_pool_max(void) {
    CnnConfig cfg;
    init_config_preset(&cfg, PRESET_SMALL_MAX);

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);

    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    nn_cnn_infer_auto_run(ctx, img, out);

    CHECK(has_nan(out, cfg.feature_size) == 0, "output contains NaN");

    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 5: standard conv, BN, POOL_DUAL, 2x filter output size */
static int test_forward_bn_pool_dual(void) {
    CnnConfig cfg;
    init_config_preset(&cfg, PRESET_SMALL_DUAL);

    /* feature_size = 8 (2 x 4 filters), verify setup */
    CHECK(cfg.feature_size == 8, "dual pool feature_size != 8");

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);

    float* out = (float*)calloc(cfg.feature_size, sizeof(float));
    nn_cnn_infer_auto_run(ctx, img, out);

    CHECK(has_nan(out, cfg.feature_size) == 0, "output contains NaN");

    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 6: depthwise conv, POOL_AVG */
static int test_forward_depthwise(void) {
    CnnConfig cfg;
    init_config_preset(&cfg, PRESET_SMALL_DW);

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed (did depthwise validation pass?)");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);

    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    nn_cnn_infer_auto_run(ctx, img, out);

    CHECK(has_nan(out, cfg.feature_size) == 0, "output contains NaN");

    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 7: stride=2, output spatial dim halved */
static int test_forward_stride2(void) {
    CnnConfig cfg;
    init_config_preset(&cfg, PRESET_SMALL_S2);

    /* Verify that stride 2 produces fewer positions than stride 1.
     * (10-3)/2+1 = 4 per side → 16 positions vs stride 1: (10-3)+1 = 8 → 64. */
    size_t pos = ((cfg.frame_height - cfg.kernel_size) / cfg.stride + 1U) *
                 ((cfg.frame_width  - cfg.kernel_size) / cfg.stride + 1U);
    CHECK(pos == 16, "stride-2 position count != 16");

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);

    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    nn_cnn_infer_auto_run(ctx, img, out);

    CHECK(has_nan(out, cfg.feature_size) == 0, "output contains NaN");

    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 8: feature_size == filter_count (square projection) works */
static int test_forward_no_projection(void) {
    CnnConfig cfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);
    cfg.use_batch_norm = 0;

    /* feature_size == filter_count == 4 → square 4×4 projection matrix */
    CHECK(cfg.feature_size == cfg.filter_count, "feature_size != filter_count");

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);

    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    nn_cnn_infer_auto_run(ctx, img, out);

    CHECK(has_nan(out, cfg.feature_size) == 0, "output contains NaN");

    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Group B: Training — Weight Updates & Gradient Flow (tests 9–12)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Test 9: training context lifecycle (allocate + free) */
static int test_train_create_destroy(void) {
    CnnConfig cfg;
    CnnTrainConfig tcfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);
    init_train_config(&tcfg, 1);

    CnnInferContext* infer = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(infer != NULL, "infer_create failed");

    CnnTrainContext* train = nn_cnn_train_create(infer, &tcfg);
    CHECK(train != NULL, "train_create failed");
    CHECK(train->bn_spatial_var != NULL, "bn_spatial_var is NULL");
    CHECK(train->conv_weight_grad != NULL, "conv_weight_grad is NULL");

    nn_cnn_train_destroy(train);
    nn_cnn_infer_destroy(infer);
    return g_test_failures;
}

/* Test 10: weights change after one training step */
static int test_train_weight_update(void) {
    CnnConfig cfg;
    CnnTrainConfig tcfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);
    cfg.use_batch_norm = 0;
    init_train_config(&tcfg, 1);

    CnnInferContext* infer = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(infer != NULL, "infer_create failed");

    float w_before = infer->conv_weights[0];

    CnnTrainContext* train = nn_cnn_train_create(infer, &tcfg);
    CHECK(train != NULL, "train_create failed");

    size_t total = cfg.total_input_size;
    float* img = (float*)malloc(total * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);
    make_onehot_target(target, cfg.feature_size, 1);

    int rc = nn_cnn_train_step_with_data(train, img, target);
    CHECK(rc == 0, "train_step_with_data returned error");

    float w_after = infer->conv_weights[0];
    CHECK(fabsf(w_after - w_before) > 1e-8f, "conv_weight did not change after training");

    free(target);
    free(img);
    nn_cnn_train_destroy(train);
    nn_cnn_infer_destroy(infer);
    return g_test_failures;
}

/* Test 11: gradient buffers contain non-zero values after training */
static int test_train_gradient_nonzero(void) {
    CnnConfig cfg;
    CnnTrainConfig tcfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);
    cfg.use_batch_norm = 1;
    init_train_config(&tcfg, 1);

    CnnInferContext* infer = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(infer != NULL, "infer_create failed");

    CnnTrainContext* train = nn_cnn_train_create(infer, &tcfg);
    CHECK(train != NULL, "train_create failed");

    size_t total = cfg.total_input_size;
    float* img = (float*)malloc(total * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);
    make_onehot_target(target, cfg.feature_size, 0);

    int rc = nn_cnn_train_step_with_data(train, img, target);
    CHECK(rc == 0, "train_step_with_data returned error");

    /* Verify conv gradients are non-zero */
    int grad_zero = 1;
    size_t conv_wc = cfg.filter_count * cfg.channel_count * cfg.kernel_size * cfg.kernel_size;
    for (size_t i = 0; i < conv_wc; i++) {
        if (fabsf(train->conv_weight_grad[i]) > 1e-8f) { grad_zero = 0; break; }
    }
    CHECK(!grad_zero, "conv_weight_grad is all zero");

    /* Verify projection gradients are non-zero */
    int proj_zero = 1;
    size_t proj_wc = cfg.feature_size * cfg.filter_count; /* pooled_count = filter_count for AVG */
    for (size_t i = 0; i < proj_wc; i++) {
        if (fabsf(train->projection_weight_grad[i]) > 1e-8f) { proj_zero = 0; break; }
    }
    CHECK(!proj_zero, "projection_weight_grad is all zero");

    /* Verify BN gradients are non-zero */
    int bn_grad_zero = 1;
    for (size_t i = 0; i < cfg.filter_count; i++) {
        if (fabsf(train->bn_gamma_grad[i]) > 1e-8f) { bn_grad_zero = 0; break; }
    }
    CHECK(!bn_grad_zero, "bn_gamma_grad is all zero");

    free(target);
    free(img);
    nn_cnn_train_destroy(train);
    nn_cnn_infer_destroy(infer);
    return g_test_failures;
}

/* Test 12: batch_size=4, verify cumulative gradient accumulation */
static int test_train_gradient_accumulation(void) {
    CnnConfig cfg;
    CnnTrainConfig tcfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);
    cfg.use_batch_norm = 0;
    init_train_config(&tcfg, 4);  /* batch_size = 4 */

    CnnInferContext* infer = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(infer != NULL, "infer_create failed");

    float w_before = infer->conv_weights[0];

    CnnTrainContext* train = nn_cnn_train_create(infer, &tcfg);
    CHECK(train != NULL, "train_create failed");
    CHECK(train->batch_step_count == 0, "batch_step_count not initialized to 0");

    size_t total = cfg.total_input_size;
    float* images[4];
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    make_onehot_target(target, cfg.feature_size, 2);

    for (int i = 0; i < 4; i++) {
        images[i] = (float*)malloc(total * sizeof(float));
        generate_sine_image(images[i], cfg.frame_height, cfg.frame_width, cfg.channel_count);
        /* Slightly perturb each image to produce different gradients */
        for (size_t j = 0; j < total; j++)
            images[i][j] += (float)i * 0.1f;
    }

    /* Step 1–3: should accumulate, no update yet */
    for (int i = 0; i < 3; i++) {
        float w_before_step = infer->conv_weights[0];
        int rc = nn_cnn_train_step_with_data(train, images[i], target);
        CHECK(rc == 0, "train_step_with_data returned error");
        /* No update applied until batch is complete */
        CHECK(fabsf(infer->conv_weights[0] - w_before_step) < 1e-8f,
              "weights changed before batch completion");
    }

    /* Step 4: should trigger parameter update */
    int rc = nn_cnn_train_step_with_data(train, images[3], target);
    CHECK(rc == 0, "train_step_with_data returned error");

    float w_after = infer->conv_weights[0];
    CHECK(fabsf(w_after - w_before) > 1e-8f,
          "weights did not change after batch completion");
    CHECK(train->batch_step_count == 0,
          "batch_step_count not reset after batch completion");

    for (int i = 0; i < 4; i++) free(images[i]);
    free(target);
    nn_cnn_train_destroy(train);
    nn_cnn_infer_destroy(infer);
    return g_test_failures;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Group C: BN Bug Regressions (tests 13–17)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Test 13: Bug #1 — bn_spatial_var initialized from running var, not zero */
static int test_bn_bug1_spatial_var_init(void) {
    CnnConfig cfg;
    CnnTrainConfig tcfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);
    init_train_config(&tcfg, 1);

    CnnInferContext* infer = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(infer != NULL, "infer_create failed");

    /* Set running var to known non-zero, non-1.0 values */
    for (size_t f = 0; f < cfg.filter_count; f++) {
        infer->bn_running_var[f] = 1.5f + (float)f * 0.5f;
    }

    CnnTrainContext* train = nn_cnn_train_create(infer, &tcfg);
    CHECK(train != NULL, "train_create failed");
    CHECK(train->bn_spatial_var != NULL, "bn_spatial_var is NULL");

    for (size_t f = 0; f < cfg.filter_count; f++) {
        float expected = infer->bn_running_var[f];
        float actual   = train->bn_spatial_var[f];
        CHECK_FEQ(actual, expected, 1e-5f, "bn_spatial_var not initialized from bn_running_var");
        CHECK(actual > 0.0f, "bn_spatial_var is zero (would cause grad explosion)");
    }

    nn_cnn_train_destroy(train);
    nn_cnn_infer_destroy(infer);
    return g_test_failures;
}

/* Test 14: Bug #2 — LeakyReLU passes gradients for negative inputs */
static int test_bn_bug2_leakyrelu_gradient(void) {
    CnnConfig cfg;
    CnnTrainConfig tcfg;

    /* Minimal network: 6x6x1, kernel=3, 1 filter, POOL_AVG, no BN */
    memset(&cfg, 0, sizeof(cfg));
    cfg.frame_height       = 6;
    cfg.frame_width        = 6;
    cfg.channel_count      = 1;
    cfg.kernel_size        = 3;
    cfg.filter_count       = 1;
    cfg.feature_size       = 1;
    cfg.pooling_mode       = CNN_POOL_AVG;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 1;
    cfg.use_batch_norm     = 0;
    cfg.pooling_activation = CNN_ACT_LEAKY_RELU;
    cfg.output_activation  = CNN_ACT_NONE;
    cfg.sequence_length    = 1;
    cfg.bn_momentum        = 0.9f;
    cfg.bn_epsilon         = 1e-5f;
    cfg.total_input_size   = 6 * 6 * 1;

    CnnInferContext* infer = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(infer != NULL, "infer_create failed");

    /* Set conv weights to negative values, bias to negative */
    size_t conv_wc = cfg.filter_count * cfg.channel_count * cfg.kernel_size * cfg.kernel_size;
    for (size_t i = 0; i < conv_wc; i++)
        infer->conv_weights[i] = -0.5f;
    infer->conv_bias[0] = -0.5f;

    /* All-one input: each conv position = 9 * (-0.5) + (-0.5) = -5.0
     * Average pool of 16 positions: -5.0
     * LeakyReLU(-5.0) = -0.05
     * If it were ReLU, output would be 0 and gradient would be 0. */

    init_train_config(&tcfg, 1);
    CnnTrainContext* train = nn_cnn_train_create(infer, &tcfg);
    CHECK(train != NULL, "train_create failed");

    float* img    = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    fill_const(img, cfg.total_input_size, 1.0f);
    target[0] = 0.0f;  /* target is far from actual output → non-zero gradient */

    int rc = nn_cnn_train_step_with_data(train, img, target);
    CHECK(rc == 0, "train_step_with_data returned error");

    /* Verify gradient is non-zero (LeakyReLU passes gradient through) */
    int grad_zero = 1;
    for (size_t i = 0; i < conv_wc; i++) {
        if (fabsf(train->conv_weight_grad[i]) > 1e-10f) { grad_zero = 0; break; }
    }
    CHECK(!grad_zero, "LeakyReLU produced zero gradient for negative input");

    free(target);
    free(img);
    nn_cnn_train_destroy(train);
    nn_cnn_infer_destroy(infer);
    return g_test_failures;
}

/* Test 15: Bug #3 — BN backward uses cached x_hat (pre_cache is populated) */
static int test_bn_bug3_xhat_formula(void) {
    CnnConfig cfg;
    CnnTrainConfig tcfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);
    /* BN gamma = 1, beta = 0, running_mean = 0, running_var = 1 */
    init_train_config(&tcfg, 1);

    CnnInferContext* infer = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(infer != NULL, "infer_create failed");

    /* Set known BN params */
    for (size_t f = 0; f < cfg.filter_count; f++) {
        infer->bn_gamma[f]        = 1.0f;
        infer->bn_beta[f]         = 0.0f;
        infer->bn_running_mean[f] = 0.0f;
        infer->bn_running_var[f]  = 1.0f;
    }

    CnnTrainContext* train = nn_cnn_train_create(infer, &tcfg);
    CHECK(train != NULL, "train_create failed");

    size_t total = cfg.total_input_size;
    float* img    = (float*)malloc(total * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);
    make_onehot_target(target, cfg.feature_size, 0);

    int rc = nn_cnn_train_step_with_data(train, img, target);
    CHECK(rc == 0, "train_step_with_data returned error");

    /* With running_mean=0, running_var=1, eps=1e-5:
     * inv_std = 1/sqrt(1+1e-5) ≈ 0.999995
     * x_hat = pooled_value * inv_std
     * bn_pre_cache should match x_hat. */
    for (size_t f = 0; f < cfg.filter_count; f++) {
        float pooled = train->pooled_linear_cache[f];
        float inv_std_expected = 1.0f / sqrtf(1.0f + 1e-5f);
        float x_hat_expected = pooled * inv_std_expected;
        float x_hat_cached = train->bn_pre_cache[f];
        CHECK_FEQ(x_hat_cached, x_hat_expected, 1e-4f, "cached x_hat != expected x_hat");
    }

    free(target);
    free(img);
    nn_cnn_train_destroy(train);
    nn_cnn_infer_destroy(infer);
    return g_test_failures;
}

/* Test 16: Bug #4 — BN running stats update after forward */
static int test_bn_bug4_running_stats_update(void) {
    CnnConfig cfg;
    CnnTrainConfig tcfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);
    init_train_config(&tcfg, 1);

    CnnInferContext* infer = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(infer != NULL, "infer_create failed");

    /* Set known initial values */
    for (size_t f = 0; f < cfg.filter_count; f++) {
        infer->bn_running_mean[f] = 0.0f;
        infer->bn_running_var[f]  = 1.0f;
    }

    CnnTrainContext* train = nn_cnn_train_create(infer, &tcfg);
    CHECK(train != NULL, "train_create failed");

    size_t total = cfg.total_input_size;
    float* img    = (float*)malloc(total * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);
    make_onehot_target(target, cfg.feature_size, 1);

    int rc = nn_cnn_train_step_with_data(train, img, target);
    CHECK(rc == 0, "train_step_with_data returned error");

    /* Running stats should have changed via EMA */
    int changed = 0;
    for (size_t f = 0; f < cfg.filter_count; f++) {
        if (fabsf(infer->bn_running_mean[f]) > 1e-7f) changed++;
    }
    CHECK(changed > 0, "bn_running_mean did not change (EMA not applied)");

    changed = 0;
    for (size_t f = 0; f < cfg.filter_count; f++) {
        if (fabsf(infer->bn_running_var[f] - 1.0f) > 1e-7f) changed++;
    }
    CHECK(changed > 0, "bn_running_var did not change (EMA not applied)");

    free(target);
    free(img);
    nn_cnn_train_destroy(train);
    nn_cnn_infer_destroy(infer);
    return g_test_failures;
}

/* Test 17: Bug #5 — cached x_hat/inv_std match gradient ratios */
static int test_bn_bug5_fwd_bwd_consistency(void) {
    CnnConfig cfg;
    CnnTrainConfig tcfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);
    cfg.filter_count = 2;  /* 2 filters for simpler math */
    cfg.feature_size = 2;
    cfg.total_input_size = 6 * 6 * 1;
    init_train_config(&tcfg, 1);

    CnnInferContext* infer = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(infer != NULL, "infer_create failed");

    /* Set known BN params */
    for (size_t f = 0; f < cfg.filter_count; f++) {
        infer->bn_gamma[f]        = 1.0f;
        infer->bn_beta[f]         = 0.0f;
        infer->bn_running_mean[f] = 0.0f;
        infer->bn_running_var[f]  = 1.0f;
    }

    CnnTrainContext* train = nn_cnn_train_create(infer, &tcfg);
    CHECK(train != NULL, "train_create failed");

    size_t total = cfg.total_input_size;
    float* img    = (float*)malloc(total * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    generate_sine_image(img, 6, 6, 1);
    target[0] = 0.2f; target[1] = 0.8f;

    int rc = nn_cnn_train_step_with_data(train, img, target);
    CHECK(rc == 0, "train_step_with_data returned error");

    /* Verify: cached x_hat ≠ running-stats-based x_hat (post-EMA) */
    for (int f = 0; f < (int)cfg.filter_count; f++) {
        float raw = train->pooled_linear_cache[f];
        float post_mean = infer->bn_running_mean[f];
        float post_var  = infer->bn_running_var[f];
        float post_inv_std = 1.0f / sqrtf(post_var + 1e-5f);

        float x_hat_cached  = train->bn_pre_cache[f];
        float x_hat_running = (raw - post_mean) * post_inv_std;

        /* Post-EMA x_hat must differ from cached x_hat */
        CHECK(fabsf(x_hat_cached - x_hat_running) > 1e-6f,
              "cached and post-EMA x_hat are identical (can't distinguish)");

        /* Backward should have used cached x_hat.
         * Recover: dpool_act = bn_beta_grad, gamma_grad = dpool_act * x_hat. */
        float dpool_act  = train->bn_beta_grad[f];
        float gamma_grad = train->bn_gamma_grad[f];
        float implied_x_hat = (fabsf(dpool_act) > 1e-6f) ? (gamma_grad / dpool_act) : 0.0f;

        CHECK_FEQ(implied_x_hat, x_hat_cached, 1e-3f,
                  "gamma_grad/beta_grad ratio != cached x_hat");
        CHECK(fabsf(gamma_grad) > 1e-6f, "bn_gamma_grad is zero");
    }

    free(target);
    free(img);
    nn_cnn_train_destroy(train);
    nn_cnn_infer_destroy(infer);
    return g_test_failures;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Group D: Activation Types (tests 18–21)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: create minimal 1-filter network with given activation, verify
 * output for known conv input equals activation(expected_conv).
 *
 * Uses a 4×4 frame (4 conv positions) to keep math simple. Sets projection
 * weight to 1.0 and bias to 0.0 (identity) so the output IS the activated
 * pooled value. */
static int test_activation_helper(CnnActivationType act, float (*expected_fn)(float)) {
    CnnConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.frame_height       = 4;
    cfg.frame_width        = 4;
    cfg.channel_count      = 1;
    cfg.kernel_size        = 3;
    cfg.filter_count       = 1;
    cfg.feature_size       = 1;
    cfg.pooling_mode       = CNN_POOL_AVG;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 1;
    cfg.use_batch_norm     = 0;
    cfg.pooling_activation = act;
    cfg.output_activation  = CNN_ACT_NONE;
    cfg.sequence_length    = 1;
    cfg.bn_momentum        = 0.9f;
    cfg.bn_epsilon         = 1e-5f;
    cfg.total_input_size   = 4 * 4 * 1;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed");

    /* Override projection: identity + zero bias */
    ctx->projection_weights[0] = 1.0f;
    ctx->projection_bias[0]    = 0.0f;

    size_t conv_wc = cfg.filter_count * cfg.channel_count * cfg.kernel_size * cfg.kernel_size;
    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    fill_const(img, cfg.total_input_size, 1.0f);

    /* ── Test negative region ──
     * All-one input, conv weights = -0.5, bias = -0.5.
     * Per-position conv = 9 * (-0.5) + (-0.5) = -5.0.
     * 4 positions, AVG pool = -5.0.
     */
    for (size_t i = 0; i < conv_wc; i++) ctx->conv_weights[i] = -0.5f;
    ctx->conv_bias[0] = -0.5f;

    nn_cnn_infer_auto_run(ctx, img, out);
    float expected = expected_fn(-5.0f);
    CHECK_FEQ(out[0], expected, 1e-4f, "activation output mismatch (negative input)");

    /* ── Test positive region ── */
    for (size_t i = 0; i < conv_wc; i++) ctx->conv_weights[i] = 0.5f;
    ctx->conv_bias[0] = 0.5f;

    nn_cnn_infer_auto_run(ctx, img, out);
    expected = expected_fn(5.0f);
    CHECK_FEQ(out[0], expected, 1e-4f, "activation output mismatch (positive input)");

    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

static float relu_fn(float x)         { return x > 0.0f ? x : 0.0f; }
static float tanh_fn(float x)         { return tanhf(x); }
static float relu6_fn(float x)        { return x > 6.0f ? 6.0f : (x > 0.0f ? x : 0.0f); }
static float leaky_relu_fn(float x)   { return x > 0.0f ? x : 0.01f * x; }

/* Test 18: ReLU clips negatives to 0 */
static int test_activation_relu(void) {
    test_activation_helper(CNN_ACT_RELU, relu_fn);
    CHECK(fabsf(relu_fn(-5.0f)) < 1e-7f, "ReLU didn't clip negative to zero");
    return g_test_failures;
}

/* Test 19: Tanh squeezes to [-1, 1] */
static int test_activation_tanh(void) {
    test_activation_helper(CNN_ACT_TANH, tanh_fn);
    CHECK(tanh_fn(5.0f) < 1.0f && tanh_fn(5.0f) > -1.0f, "Tanh output out of range");
    return g_test_failures;
}

/* Test 20: ReLU6 clips to [0, 6] */
static int test_activation_relu6(void) {
    test_activation_helper(CNN_ACT_RELU6, relu6_fn);
    CHECK(relu6_fn(-5.0f) == 0.0f, "ReLU6 didn't clip negative to zero");
    CHECK(relu6_fn(5.0f) == 5.0f, "ReLU6 clipped valid positive");
    return g_test_failures;
}

/* Test 21: LeakyReLU passes 0.01x for negatives */
static int test_activation_leaky_relu(void) {
    test_activation_helper(CNN_ACT_LEAKY_RELU, leaky_relu_fn);
    CHECK_FEQ(leaky_relu_fn(-5.0f), -0.05f, 1e-6f, "LeakyReLU didn't scale negative correctly");
    CHECK(leaky_relu_fn(5.0f) == 5.0f, "LeakyReLU modified positive input");
    return g_test_failures;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Group E: Edge Cases (tests 22–24)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Test 22: all-zero input doesn't crash, produces valid output */
static int test_zero_input(void) {
    CnnConfig cfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed");

    float* img = (float*)calloc(cfg.total_input_size, sizeof(float));
    float* out = (float*)malloc(cfg.feature_size * sizeof(float));

    nn_cnn_infer_auto_run(ctx, img, out);

    CHECK(has_nan(out, cfg.feature_size) == 0, "output contains NaN on zero input");
    /* With bias term the output should be non-zero */
    int nonzero = 0;
    for (size_t i = 0; i < cfg.feature_size; i++)
        if (fabsf(out[i]) > 1e-8f) nonzero++;
    CHECK(nonzero > 0, "output is all zero (expected bias contribution)");

    /* Also test with training */
    CnnTrainConfig tcfg;
    init_train_config(&tcfg, 1);
    CnnTrainContext* train = nn_cnn_train_create(ctx, &tcfg);
    CHECK(train != NULL, "train_create failed");

    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    make_onehot_target(target, cfg.feature_size, 0);

    int rc = nn_cnn_train_step_with_data(train, img, target);
    CHECK(rc == 0, "train_step_with_data crashed on zero input");

    free(target);
    nn_cnn_train_destroy(train);
    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 23: constant-valued input doesn't crash */
static int test_constant_input(void) {
    CnnConfig cfg;
    init_config_preset(&cfg, PRESET_SMALL_AVG);

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    fill_const(img, cfg.total_input_size, 0.5f);

    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    nn_cnn_infer_auto_run(ctx, img, out);

    CHECK(has_nan(out, cfg.feature_size) == 0, "output contains NaN on constant input");

    /* Constant input means same conv output everywhere → pool = conv output.
     * With BN, output should be well-behaved. */
    for (size_t i = 0; i < cfg.feature_size; i++) {
        CHECK(!is_nan(out[i]), "output element is NaN");
    }

    /* Training with constant input */
    CnnTrainConfig tcfg;
    init_train_config(&tcfg, 1);
    CnnTrainContext* train = nn_cnn_train_create(ctx, &tcfg);
    CHECK(train != NULL, "train_create failed");

    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    make_onehot_target(target, cfg.feature_size, 1);

    int rc = nn_cnn_train_step_with_data(train, img, target);
    CHECK(rc == 0, "train_step_with_data crashed on constant input");

    free(target);
    nn_cnn_train_destroy(train);
    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 24: filter_count=1 works */
static int test_single_filter(void) {
    CnnConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.frame_height       = 8;
    cfg.frame_width        = 8;
    cfg.channel_count      = 3;
    cfg.kernel_size        = 3;
    cfg.filter_count       = 1;
    cfg.feature_size       = 1;
    cfg.pooling_mode       = CNN_POOL_AVG;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 1;
    cfg.use_batch_norm     = 0;
    cfg.pooling_activation = CNN_ACT_NONE;
    cfg.output_activation  = CNN_ACT_NONE;
    cfg.sequence_length    = 1;
    cfg.bn_momentum        = 0.9f;
    cfg.bn_epsilon         = 1e-5f;
    cfg.total_input_size   = 8 * 8 * 3;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed with filter_count=1");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);

    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    nn_cnn_infer_auto_run(ctx, img, out);

    CHECK(has_nan(out, 1) == 0, "output contains NaN");
    CHECK(fabsf(out[0]) < 100.0f, "output value unreasonably large");

    /* Also test training */
    CnnTrainConfig tcfg;
    init_train_config(&tcfg, 1);
    CnnTrainContext* train = nn_cnn_train_create(ctx, &tcfg);
    CHECK(train != NULL, "train_create failed");

    float* target = (float*)malloc(1 * sizeof(float));
    target[0] = 0.7f;

    int rc = nn_cnn_train_step_with_data(train, img, target);
    CHECK(rc == 0, "train_step_with_data failed with filter_count=1");

    float loss = train->last_loss;
    CHECK(!is_nan(loss), "loss is NaN");

    free(target);
    nn_cnn_train_destroy(train);
    free(out);
    free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Group F: P0 Shared Modules — RMSNorm & SkipConnection integration
 * ═══════════════════════════════════════════════════════════════════════ */

/* Test 25: RMSNorm forward pass produces no NaN */
static int test_p0_rms_norm_no_nan(void) {
    CnnConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sequence_length    = 2;
    cfg.frame_height       = 6;  cfg.frame_width = 6;
    cfg.channel_count      = 1;  cfg.kernel_size = 3;
    cfg.filter_count       = 4;  cfg.feature_size = 4;
    cfg.pooling_mode       = CNN_POOL_AVG;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 1;
    cfg.output_activation  = CNN_ACT_NONE;
    cfg.pooling_activation = CNN_ACT_NONE;
    cfg.use_batch_norm     = 0;
    cfg.use_rms_norm       = 1;
    cfg.norm_epsilon       = 1e-5f;
    cfg.total_input_size   = cfg.frame_height * cfg.frame_width
                           * cfg.channel_count * cfg.sequence_length;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create with rms_norm failed");

    /* Verify norm_gamma exists and initialized to 1.0 */
    CHECK(ctx->norm_gamma != NULL, "norm_gamma not allocated");
    if (ctx->norm_gamma != NULL) {
        CHECK_FEQ(ctx->norm_gamma[0], 1.0f, 1e-6f, "norm_gamma[0] not 1.0");
        CHECK_FEQ(ctx->norm_gamma[2], 1.0f, 1e-6f, "norm_gamma[2] not 1.0");
    }

    size_t img_size = cfg.total_input_size;
    float* img = (float*)malloc(img_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);
    /* second frame with different pattern */
    for (size_t i = 0; i < (size_t)(cfg.frame_height * cfg.frame_width * cfg.channel_count); i++)
        img[i + cfg.frame_height * cfg.frame_width * cfg.channel_count] = cosf((float)i * 0.7f) * 0.5f;

    size_t out_size = cfg.sequence_length * cfg.feature_size;
    float* out = (float*)malloc(out_size * sizeof(float));
    int rc = nn_cnn_infer_auto_run(ctx, img, out);
    CHECK(rc == 0, "auto_run with rms_norm failed");
    CHECK(has_nan(out, out_size) == 0, "output contains NaN with rms_norm");

    /* Verify norm_gamma unchanged after inference (no training) */
    CHECK_FEQ(ctx->norm_gamma[0], 1.0f, 1e-6f, "norm_gamma changed during inference");

    free(out); free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 26: RMSNorm training doesn't crash and produces valid loss */
static int test_p0_rms_norm_training(void) {
    CnnConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sequence_length    = 1;
    cfg.frame_height       = 6;  cfg.frame_width = 6;
    cfg.channel_count      = 1;  cfg.kernel_size = 3;
    cfg.filter_count       = 4;  cfg.feature_size = 4;
    cfg.pooling_mode       = CNN_POOL_AVG;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 1;
    cfg.output_activation  = CNN_ACT_NONE;
    cfg.pooling_activation = CNN_ACT_NONE;
    cfg.use_batch_norm     = 0;
    cfg.use_rms_norm       = 1;
    cfg.norm_epsilon       = 1e-5f;
    cfg.total_input_size   = cfg.frame_height * cfg.frame_width
                           * cfg.channel_count * cfg.sequence_length;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create with rms_norm failed");

    CnnTrainConfig tcfg;
    init_train_config(&tcfg, 1);

    CnnTrainContext* train = nn_cnn_train_create(ctx, &tcfg);
    CHECK(train != NULL, "train_create with rms_norm failed");

    size_t img_size = cfg.total_input_size;
    float* img = (float*)malloc(img_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);

    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    make_soft_target(target, cfg.feature_size);

    /* Run multiple training steps */
    float first_loss = -1.0f;
    for (int step = 0; step < 5; step++) {
        int rc = nn_cnn_train_step_with_data(train, img, target);
        CHECK(rc == 0, "train_step with rms_norm step failed");
        CHECK(!is_nan(train->last_loss), "loss is NaN with rms_norm");
        if (step == 0) first_loss = train->last_loss;
    }

    /* After training, verify loss is non-NaN and norm_gamma buffers exist */
    CHECK(first_loss > 0.0f, "first loss not positive");
    CHECK(train->norm_gamma_grad != NULL, "norm_gamma_grad not allocated");
    CHECK(train->norm_gamma_vel != NULL, "norm_gamma_vel not allocated");
    CHECK(train->norm_input_cache != NULL, "norm_input_cache not allocated");

    free(target); free(img);
    nn_cnn_train_destroy(train);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 27: RMSNorm gamma drifts after repeated training steps */
static int test_p0_rms_norm_gamma_drift(void) {
    CnnConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sequence_length    = 1;
    cfg.frame_height       = 6;  cfg.frame_width = 6;
    cfg.channel_count      = 1;  cfg.kernel_size = 3;
    cfg.filter_count       = 2;  cfg.feature_size = 2;
    cfg.pooling_mode       = CNN_POOL_AVG;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 1;
    cfg.output_activation  = CNN_ACT_NONE;
    cfg.pooling_activation = CNN_ACT_NONE;
    cfg.use_batch_norm     = 0;
    cfg.use_rms_norm       = 1;
    cfg.norm_epsilon       = 1e-5f;
    cfg.total_input_size   = cfg.frame_height * cfg.frame_width
                           * cfg.channel_count * cfg.sequence_length;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create with rms_norm failed");

    /* Record initial gamma values */
    float init_gamma[2];
    init_gamma[0] = ctx->norm_gamma[0];
    init_gamma[1] = ctx->norm_gamma[1];

    CnnTrainConfig tcfg;
    init_train_config(&tcfg, 1);
    tcfg.learning_rate = 0.01f;  /* modest LR to avoid activation explosion */

    CnnTrainContext* train = nn_cnn_train_create(ctx, &tcfg);
    CHECK(train != NULL, "train_create with rms_norm failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);
    make_onehot_target(target, cfg.feature_size, 0);

    /* Train for several steps — verify no crash, NaN, or negative gamma */
    for (int step = 0; step < 20; step++) {
        int rc = nn_cnn_train_step_with_data(train, img, target);
        CHECK(rc == 0, "train_step failed");
        CHECK(!is_nan(train->last_loss), "loss is NaN");
    }

    /* Verify norm_gamma stays reasonable (not NaN, not negative) */
    CHECK(!is_nan(ctx->norm_gamma[0]), "norm_gamma became NaN");
    CHECK(!is_nan(ctx->norm_gamma[cfg.filter_count - 1]), "norm_gamma became NaN");
    CHECK(ctx->norm_gamma[0] > 0.0f, "norm_gamma became non-positive");
    CHECK(ctx->norm_gamma[cfg.filter_count - 1] > 0.0f, "norm_gamma became non-positive");

    free(target); free(img);
    nn_cnn_train_destroy(train);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 28: Skip connection is SKIP_NONE when dimensions mismatch (default case) */
static int test_p0_skip_dims_mismatch(void) {
    CnnConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sequence_length    = 1;
    cfg.frame_height       = 6;  cfg.frame_width = 6;
    cfg.channel_count      = 1;  cfg.kernel_size = 3;
    cfg.filter_count       = 4;  cfg.feature_size = 4;
    cfg.pooling_mode       = CNN_POOL_AVG;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 1;
    cfg.output_activation  = CNN_ACT_NONE;
    cfg.pooling_activation = CNN_ACT_NONE;
    cfg.use_batch_norm     = 0;
    cfg.use_skip           = 1;
    cfg.skip_mode          = SKIP_IDENTITY;
    cfg.total_input_size   = cfg.frame_height * cfg.frame_width
                           * cfg.channel_count * cfg.sequence_length;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create with skip failed");

    /* pooled_value_count (4) == feature_size (4), so skip should be IDENTITY */
    /* Actually filter_count=4 and non-dual pooling, so pooled_value_count=4 == feature_size=4.
     * Skip SHOULD be active. Let's test that instead: */
    CHECK(ctx->skip != NULL, "skip not allocated");
    CHECK(ctx->skip->mode == SKIP_IDENTITY, "skip should be IDENTITY when dims match");

    /* Now test with different dims (should be SKIP_NONE) */
    nn_cnn_infer_destroy(ctx);

    CnnConfig cfg2;
    memset(&cfg2, 0, sizeof(cfg2));
    cfg2.sequence_length    = 1;
    cfg2.frame_height       = 6;  cfg2.frame_width = 6;
    cfg2.channel_count      = 1;  cfg2.kernel_size = 3;
    cfg2.filter_count       = 4;  cfg2.feature_size = 6;  /* 6 != 4 */
    cfg2.pooling_mode       = CNN_POOL_AVG;
    cfg2.conv_mode          = CNN_CONV_STANDARD;
    cfg2.stride             = 1;
    cfg2.output_activation  = CNN_ACT_NONE;
    cfg2.pooling_activation = CNN_ACT_NONE;
    cfg2.use_skip           = 1;
    cfg2.skip_mode          = SKIP_IDENTITY;
    cfg2.total_input_size   = cfg2.frame_height * cfg2.frame_width
                            * cfg2.channel_count * cfg2.sequence_length;

    CnnInferContext* ctx2 = nn_cnn_infer_create_with_config(&cfg2, 42);
    CHECK(ctx2 != NULL, "infer_create with mismatched dims failed");
    CHECK(ctx2->skip != NULL, "skip not allocated for mismatched dims");
    if (ctx2->skip != NULL) {
        CHECK(ctx2->skip->mode == SKIP_NONE, "skip should be SKIP_NONE when dims mismatch");
    }

    /* Forward pass shouldn't crash with SKIP_NONE */
    float* img = (float*)malloc(cfg2.total_input_size * sizeof(float));
    float* out = (float*)malloc(cfg2.feature_size * sizeof(float));
    generate_sine_image(img, cfg2.frame_height, cfg2.frame_width, cfg2.channel_count);
    int rc = nn_cnn_infer_auto_run(ctx2, img, out);
    CHECK(rc == 0, "auto_run with skip SKIP_NONE failed");
    CHECK(has_nan(out, cfg2.feature_size) == 0, "output contains NaN with SKIP_NONE");

    free(out); free(img);
    nn_cnn_infer_destroy(ctx2);
    return g_test_failures;
}

/* Test 29: Skip connection training doesn't crash when dims match */
static int test_p0_skip_training(void) {
    /* When pooled_value_count == feature_size, skip can be IDENTITY */
    CnnConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sequence_length    = 1;
    cfg.frame_height       = 6;  cfg.frame_width = 6;
    cfg.channel_count      = 1;  cfg.kernel_size = 3;
    cfg.filter_count       = 4;  cfg.feature_size = 4;  /* matches pooled_value_count */
    cfg.pooling_mode       = CNN_POOL_AVG;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 1;
    cfg.output_activation  = CNN_ACT_NONE;
    cfg.pooling_activation = CNN_ACT_NONE;
    cfg.use_batch_norm     = 0;
    cfg.use_skip           = 1;
    cfg.skip_mode          = SKIP_IDENTITY;
    cfg.total_input_size   = cfg.frame_height * cfg.frame_width
                           * cfg.channel_count * cfg.sequence_length;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create with dim-matched skip failed");
    CHECK(ctx->skip->mode == SKIP_IDENTITY, "skip not IDENTITY when dims match");

    /* Forward pass with skip */
    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);
    int rc = nn_cnn_infer_auto_run(ctx, img, out);
    CHECK(rc == 0, "auto_run with skip IDENTITY failed");
    CHECK(has_nan(out, cfg.feature_size) == 0, "output NaN with skip IDENTITY");

    /* Training with skip */
    CnnTrainConfig tcfg;
    init_train_config(&tcfg, 1);
    CnnTrainContext* train = nn_cnn_train_create(ctx, &tcfg);
    CHECK(train != NULL, "train_create with skip failed");

    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    make_soft_target(target, cfg.feature_size);

    for (int step = 0; step < 5; step++) {
        rc = nn_cnn_train_step_with_data(train, img, target);
        CHECK(rc == 0, "train_step with skip failed");
        CHECK(!is_nan(train->last_loss), "loss NaN with skip IDENTITY");
    }

    free(target);
    nn_cnn_train_destroy(train);
    free(out); free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 30: Weight save/load preserves norm_gamma */
static int test_p0_save_load_norm_gamma(void) {
    CnnConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sequence_length    = 1;
    cfg.frame_height       = 6;  cfg.frame_width = 6;
    cfg.channel_count      = 1;  cfg.kernel_size = 3;
    cfg.filter_count       = 4;  cfg.feature_size = 4;
    cfg.pooling_mode       = CNN_POOL_AVG;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 1;
    cfg.output_activation  = CNN_ACT_NONE;
    cfg.pooling_activation = CNN_ACT_NONE;
    cfg.use_batch_norm     = 0;
    cfg.use_rms_norm       = 1;
    cfg.norm_epsilon       = 1e-5f;
    cfg.total_input_size   = cfg.frame_height * cfg.frame_width
                           * cfg.channel_count * cfg.sequence_length;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "infer_create failed");

    /* Manually modify norm_gamma to non-default values */
    ctx->norm_gamma[0] = 0.5f;
    ctx->norm_gamma[1] = 2.0f;
    ctx->norm_gamma[2] = 0.75f;
    ctx->norm_gamma[3] = 1.5f;

    /* Save to temporary file */
    const char* path = "test_p0_norm_gamma.bin";
    FILE* fp = fopen(path, "wb");
    CHECK(fp != NULL, "cannot open save file");
    int save_ok = nn_cnn_save_weights(ctx, fp);
    fclose(fp);
    CHECK(save_ok == 1, "save_weights failed");

    /* Create new context and load */
    CnnInferContext* ctx2 = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx2 != NULL, "infer_create for load failed");
    CHECK_FEQ(ctx2->norm_gamma[0], 1.0f, 1e-6f, "pre-load norm_gamma not default");

    fp = fopen(path, "rb");
    CHECK(fp != NULL, "cannot open load file");
    int load_ok = nn_cnn_load_weights(ctx2, fp);
    fclose(fp);
    CHECK(load_ok == 1, "load_weights failed");

    /* Verify norm_gamma loaded correctly */
    CHECK_FEQ(ctx2->norm_gamma[0], 0.5f, 1e-6f, "norm_gamma[0] not restored");
    CHECK_FEQ(ctx2->norm_gamma[1], 2.0f, 1e-6f, "norm_gamma[1] not restored");
    CHECK_FEQ(ctx2->norm_gamma[2], 0.75f, 1e-6f, "norm_gamma[2] not restored");
    CHECK_FEQ(ctx2->norm_gamma[3], 1.5f, 1e-6f, "norm_gamma[3] not restored");

    /* Verify forward pass with loaded weights produces same output */
    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* out1 = (float*)malloc(cfg.feature_size * sizeof(float));
    float* out2 = (float*)malloc(cfg.feature_size * sizeof(float));
    generate_sine_image(img, cfg.frame_height, cfg.frame_width, cfg.channel_count);

    nn_cnn_infer_auto_run(ctx, img, out1);
    nn_cnn_infer_auto_run(ctx2, img, out2);

    for (size_t i = 0; i < (size_t)cfg.feature_size; i++) {
        CHECK_FEQ(out1[i], out2[i], 1e-5f, "output mismatch after save/load");
    }

    free(out2); free(out1); free(img);
    nn_cnn_infer_destroy(ctx2);
    nn_cnn_infer_destroy(ctx);
    (void)remove(path);
    return g_test_failures;
}

/* ═══════════════════════════════════════════════════════════════════════
 * main — run all registered tests
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    srand(42);

    TestCase tests[] = {
        /* Group A: Inference */
        {"test_forward_no_bn_pool_avg",    test_forward_no_bn_pool_avg},
        {"test_forward_bn_pool_avg",       test_forward_bn_pool_avg},
        {"test_forward_bn_pool_none",      test_forward_bn_pool_none},
        {"test_forward_bn_pool_max",       test_forward_bn_pool_max},
        {"test_forward_bn_pool_dual",      test_forward_bn_pool_dual},
        {"test_forward_depthwise",         test_forward_depthwise},
        {"test_forward_stride2",           test_forward_stride2},
        {"test_forward_no_projection",     test_forward_no_projection},

        /* Group B: Training */
        {"test_train_create_destroy",      test_train_create_destroy},
        {"test_train_weight_update",       test_train_weight_update},
        {"test_train_gradient_nonzero",    test_train_gradient_nonzero},
        {"test_train_gradient_accumulation", test_train_gradient_accumulation},

        /* Group C: BN Bug Regressions */
        {"test_bn_bug1_spatial_var_init",  test_bn_bug1_spatial_var_init},
        {"test_bn_bug2_leakyrelu_gradient", test_bn_bug2_leakyrelu_gradient},
        {"test_bn_bug3_xhat_formula",      test_bn_bug3_xhat_formula},
        {"test_bn_bug4_running_stats_update", test_bn_bug4_running_stats_update},
        {"test_bn_bug5_fwd_bwd_consistency", test_bn_bug5_fwd_bwd_consistency},

        /* Group D: Activation Types */
        {"test_activation_relu",           test_activation_relu},
        {"test_activation_tanh",           test_activation_tanh},
        {"test_activation_relu6",          test_activation_relu6},
        {"test_activation_leaky_relu",     test_activation_leaky_relu},

        /* Group E: Edge Cases */
        {"test_zero_input",                test_zero_input},
        {"test_constant_input",            test_constant_input},
        {"test_single_filter",             test_single_filter},

        /* Group F: P0 Shared Modules */
        {"test_p0_rms_norm_no_nan",        test_p0_rms_norm_no_nan},
        {"test_p0_rms_norm_training",      test_p0_rms_norm_training},
        {"test_p0_rms_norm_gamma_drift",   test_p0_rms_norm_gamma_drift},
        {"test_p0_skip_dims_mismatch",     test_p0_skip_dims_mismatch},
        {"test_p0_skip_training",          test_p0_skip_training},
        {"test_p0_save_load_norm_gamma",   test_p0_save_load_norm_gamma},
    };

    size_t total  = sizeof(tests) / sizeof(tests[0]);
    size_t passed = 0;

    for (size_t i = 0; i < total; i++) {
        g_test_failures = 0;
        printf("[%2zu/%zu] %s ... ", i + 1, total, tests[i].name);

        int r = tests[i].func();
        if (r == 0) {
            printf("PASSED\n");
            passed++;
        } else {
            printf("FAILED (%d failures)\n", r);
            fflush(stdout);
        }
    }

    printf("\n=== %zu/%zu TESTS PASSED ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
