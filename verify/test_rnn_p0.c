/**
 * @file test_rnn_p0.c
 * @brief P0 shared-module integration tests for RNN (14 tests).
 *
 * Covers RMSNorm, Dropout, SkipConnection (LAuReL-RW) integration
 * in the RNN backend across forward, backward, save/load, and edge cases.
 */

#include "../src/nn/types/rnn/rnn_config.h"
#include "../src/nn/types/rnn/rnn_infer_ops.h"
#include "../src/nn/types/rnn/rnn_train_ops.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Test framework ─────────────────────────────────────────────────── */
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "  FAIL: %s\n", msg); g_fail++; } \
} while(0)

#define CHECK_FEQ(a, b, tol, msg) do { \
    float va = (float)(a), vb = (float)(b); \
    if (fabsf(va - vb) > (float)(tol)) { \
        fprintf(stderr, "  FAIL: %s (%.6f vs %.6f)\n", msg, (double)va, (double)vb); \
        g_fail++; \
    } \
} while(0)

typedef int (*TestFunc)(void);
typedef struct { const char* name; TestFunc func; } TestCase;

/* ── Helpers ────────────────────────────────────────────────────────── */
static int is_nan(float x) { return x != x; }
static int has_nan(const float* a, size_t n) {
    for (size_t i = 0; i < n; i++) if (is_nan(a[i])) return 1;
    return 0;
}
static void fill_sine(float* a, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] = sinf((float)i * 0.7f) * 0.8f + 0.3f;
}
static void make_onehot(float* t, size_t n, size_t c) {
    for (size_t i = 0; i < n; i++) t[i] = 0.0f;
    if (c < n) t[c] = 1.0f;
}

/* ── RNN config helpers ─────────────────────────────────────────────── */
static void init_rnn_cfg(RnnConfig* cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->sequence_length    = 4;
    cfg->input_feature_size = 3;
    cfg->hidden_size        = 6;
    cfg->output_size        = 3;
    cfg->hidden_activation  = RNN_ACT_TANH;
    cfg->output_activation  = RNN_ACT_NONE;
    cfg->seed               = 42;
}

static void init_rnn_train_cfg(RnnTrainConfig* tcfg) {
    memset(tcfg, 0, sizeof(*tcfg));
    tcfg->learning_rate = 0.01f;
    tcfg->momentum      = 0.0f;
    tcfg->weight_decay  = 0.0f;
    tcfg->batch_size    = 1;
    tcfg->seed          = 42;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 1: Backward compatibility — P0 disabled
 * ═══════════════════════════════════════════════════════════════════════ */
static int test_rnn_backward_compat(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    RnnInferContext* ctx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ctx != NULL, "create failed");
    CHECK(!ctx->use_dropout, "dropout on when P0 off");
    CHECK(!ctx->use_skip, "skip on when P0 off");

    float in[12]; float out[3];
    fill_sine(in, 12);
    int rc = nn_rnn_forward_pass(ctx, in, out, NULL, NULL);
    CHECK(rc == 0, "forward P0-off failed");
    CHECK(!has_nan(out, 3), "output NaN P0-off");
    CHECK(out[0] != 0.0f || out[1] != 0.0f, "output all zero P0-off");

    nn_rnn_infer_destroy(ctx);
    return g_fail;
}

/* Test 2: RMSNorm forward — no NaN, gamma=1.0 */
static int test_rnn_rms_norm_forward(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_rms_norm = 1;
    cfg.norm_epsilon = 1e-5f;
    RnnInferContext* ctx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ctx != NULL, "create rms_norm failed");
    CHECK(ctx->norm_gamma_h != NULL, "norm_gamma_h not allocated");
    CHECK_FEQ(ctx->norm_gamma_h[0], 1.0f, 1e-6f, "gamma[0]!=1");

    float in[12]; float out[3];
    fill_sine(in, 12);
    int rc = nn_rnn_forward_pass(ctx, in, out, NULL, NULL);
    CHECK(rc == 0, "forward rms_norm failed");
    CHECK(!has_nan(out, 3), "output NaN rms_norm");

    nn_rnn_infer_destroy(ctx);
    return g_fail;
}

/* Test 3: RMSNorm training — no crash, buffers allocated */
static int test_rnn_rms_norm_training(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_rms_norm = 1; cfg.norm_epsilon = 1e-5f;
    RnnInferContext* ictx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ictx != NULL, "create failed");

    RnnTrainConfig tcfg; init_rnn_train_cfg(&tcfg);
    RnnTrainContext* tctx = nn_rnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create rms_norm failed");
    CHECK(tctx->norm_gamma_grad != NULL, "norm_gamma_grad missing");
    CHECK(tctx->pre_norm_cache != NULL, "pre_norm_cache missing");

    float in[12]; float target[3];
    fill_sine(in, 12); make_onehot(target, 3, 0);
    for (int i = 0; i < 5; i++) {
        int rc = nn_rnn_train_step_with_data(tctx, in, target);
        CHECK(rc == 0, "train_step rms_norm failed");
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }
    CHECK(tctx->last_loss > 0.0f, "loss non-positive");

    nn_rnn_train_destroy(tctx);
    nn_rnn_infer_destroy(ictx);
    return g_fail;
}

/* Test 4: RMSNorm gamma drifts after training */
static int test_rnn_rms_norm_gamma_drift(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_rms_norm = 1; cfg.norm_epsilon = 1e-5f;
    RnnInferContext* ictx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ictx != NULL, "create failed");

    RnnTrainConfig tcfg; init_rnn_train_cfg(&tcfg);
    tcfg.learning_rate = 0.1f;
    RnnTrainContext* tctx = nn_rnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create failed");

    float in[12]; float target[3];
    fill_sine(in, 12); make_onehot(target, 3, 0);
    for (int i = 0; i < 50; i++) {
        nn_rnn_train_step_with_data(tctx, in, target);
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }
    for (size_t i = 0; i < cfg.hidden_size; i++) {
        CHECK(!is_nan(ictx->norm_gamma_h[i]), "gamma NaN");
        CHECK(ictx->norm_gamma_h[i] > 0.0f, "gamma non-positive");
    }
    int drifted = 0;
    for (size_t i = 0; i < cfg.hidden_size; i++)
        if (fabsf(ictx->norm_gamma_h[i] - 1.0f) > 1e-6f) { drifted = 1; break; }
    CHECK(drifted, "gamma did not drift");

    nn_rnn_train_destroy(tctx);
    nn_rnn_infer_destroy(ictx);
    return g_fail;
}

/* Test 5: Dropout forward — no crash */
static int test_rnn_dropout_forward(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_dropout = 1; cfg.dropout_rate = 0.3f;
    RnnInferContext* ctx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ctx != NULL, "create dropout failed");
    CHECK_FEQ(ctx->dropout_rec.keep_prob, 0.3f, 1e-6f, "keep_prob mismatch");

    float in[12]; float out[3];
    fill_sine(in, 12);
    int rc = nn_rnn_forward_pass(ctx, in, out, NULL, NULL);
    CHECK(rc == 0, "forward dropout failed");
    CHECK(!has_nan(out, 3), "output NaN dropout");

    nn_rnn_infer_destroy(ctx);
    return g_fail;
}

/* Test 6: Dropout training — inference deterministic, loss valid */
static int test_rnn_dropout_training(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_dropout = 1; cfg.dropout_rate = 0.3f;
    RnnInferContext* ictx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ictx != NULL, "create failed");

    RnnTrainConfig tcfg; init_rnn_train_cfg(&tcfg);
    RnnTrainContext* tctx = nn_rnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create dropout failed");

    float in[12]; float target[3];
    fill_sine(in, 12); make_onehot(target, 3, 0);
    for (int i = 0; i < 5; i++) {
        int rc = nn_rnn_train_step_with_data(tctx, in, target);
        CHECK(rc == 0, "train_step dropout failed");
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }
    /* Inference determinism */
    float oa[3], ob[3];
    nn_rnn_forward_pass(ictx, in, oa, NULL, NULL);
    nn_rnn_forward_pass(ictx, in, ob, NULL, NULL);
    for (size_t i = 0; i < 3; i++) CHECK_FEQ(oa[i], ob[i], 1e-6f, "inf non-deterministic");

    nn_rnn_train_destroy(tctx);
    nn_rnn_infer_destroy(ictx);
    return g_fail;
}

/* Test 7: Skip IDENTITY forward and training */
static int test_rnn_skip_identity(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_skip = 1; cfg.skip_mode = SKIP_IDENTITY;
    RnnInferContext* ictx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ictx != NULL, "create skip failed");
    CHECK(ictx->skip_recurrent.mode == SKIP_IDENTITY, "skip mode mismatch");

    float in[12]; float out[3];
    fill_sine(in, 12);
    int rc = nn_rnn_forward_pass(ictx, in, out, NULL, NULL);
    CHECK(rc == 0, "forward skip failed");
    CHECK(!has_nan(out, 3), "output NaN skip");

    RnnTrainConfig tcfg; init_rnn_train_cfg(&tcfg);
    RnnTrainContext* tctx = nn_rnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create skip failed");
    CHECK(tctx->skip_temp_dF != NULL, "skip_temp_dF missing");
    CHECK(tctx->skip_temp_dX != NULL, "skip_temp_dX missing");

    float target[3];
    make_onehot(target, 3, 0);
    for (int i = 0; i < 5; i++) {
        rc = nn_rnn_train_step_with_data(tctx, in, target);
        CHECK(rc == 0, "train_step skip failed");
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }
    nn_rnn_train_destroy(tctx);
    nn_rnn_infer_destroy(ictx);
    return g_fail;
}

/* Test 8: Skip LAuReL-RW — alpha_raw drifts */
static int test_rnn_skip_laurel_rw(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_skip = 1; cfg.skip_mode = SKIP_LAUREL_RW;
    RnnInferContext* ictx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ictx != NULL, "create laurel failed");
    CHECK_FEQ(ictx->skip_recurrent.alpha_raw, 0.0f, 1e-6f, "alpha_raw not 0");

    RnnTrainConfig tcfg; init_rnn_train_cfg(&tcfg);
    tcfg.learning_rate = 0.5f;
    RnnTrainContext* tctx = nn_rnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create laurel failed");

    float in[12]; float target[3];
    fill_sine(in, 12); make_onehot(target, 3, 0);
    for (int i = 0; i < 20; i++) {
        nn_rnn_train_step_with_data(tctx, in, target);
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }
    CHECK(!is_nan(ictx->skip_recurrent.alpha_raw), "alpha_raw NaN");
    int drifted = fabsf(ictx->skip_recurrent.alpha_raw) > 1e-7f;
    CHECK(drifted, "alpha_raw did not drift");

    nn_rnn_train_destroy(tctx);
    nn_rnn_infer_destroy(ictx);
    return g_fail;
}

/* Test 9: Combined — RMSNorm + Dropout + Skip LAuReL-RW */
static int test_rnn_combined_all(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_rms_norm = 1; cfg.norm_epsilon = 1e-5f;
    cfg.use_dropout  = 1; cfg.dropout_rate  = 0.2f;
    cfg.use_skip     = 1; cfg.skip_mode     = SKIP_LAUREL_RW;
    RnnInferContext* ictx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ictx != NULL, "create combined failed");
    CHECK(ictx->norm_gamma_h != NULL, "gamma missing");
    CHECK(ictx->skip_recurrent.mode == SKIP_LAUREL_RW, "skip mode wrong");

    float in[12]; float out[3];
    fill_sine(in, 12);
    int rc = nn_rnn_forward_pass(ictx, in, out, NULL, NULL);
    CHECK(rc == 0, "forward combined failed");
    CHECK(!has_nan(out, 3), "output NaN combined");

    RnnTrainConfig tcfg; init_rnn_train_cfg(&tcfg);
    RnnTrainContext* tctx = nn_rnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create combined failed");
    CHECK(tctx->norm_gamma_grad != NULL, "gamma grad missing");
    CHECK(tctx->skip_temp_dF != NULL, "skip temp missing");

    float target[3];
    make_onehot(target, 3, 0);
    float first_loss = -1;
    for (int i = 0; i < 10; i++) {
        nn_rnn_train_step_with_data(tctx, in, target);
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
        if (i == 0) first_loss = tctx->last_loss;
    }
    CHECK(first_loss > 0.0f, "loss non-positive");

    for (size_t i = 0; i < cfg.hidden_size; i++)
        CHECK(ictx->norm_gamma_h[i] > 0.0f, "gamma non-positive after");

    nn_rnn_train_destroy(tctx);
    nn_rnn_infer_destroy(ictx);
    return g_fail;
}

/* Test 10: Loss decreases over training steps */
static int test_rnn_loss_decreases(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_rms_norm = 1; cfg.norm_epsilon = 1e-5f;
    RnnInferContext* ictx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ictx != NULL, "create failed");

    RnnTrainConfig tcfg; init_rnn_train_cfg(&tcfg);
    tcfg.learning_rate = 0.1f;
    RnnTrainContext* tctx = nn_rnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create failed");

    float in[12]; float target[3];
    fill_sine(in, 12); make_onehot(target, 3, 0);
    nn_rnn_train_step_with_data(tctx, in, target);
    float l0 = tctx->last_loss;
    for (int i = 0; i < 50; i++) nn_rnn_train_step_with_data(tctx, in, target);
    CHECK(tctx->last_loss < l0, "loss did not decrease");
    CHECK(!is_nan(tctx->last_loss), "final loss NaN");

    nn_rnn_train_destroy(tctx);
    nn_rnn_infer_destroy(ictx);
    return g_fail;
}

/* Test 11: Save/load preserves norm_gamma */
static int test_rnn_save_load_norm_gamma(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_rms_norm = 1; cfg.norm_epsilon = 1e-5f;
    RnnInferContext* ctx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ctx != NULL, "create failed");
    ctx->norm_gamma_h[0] = 0.5f;
    ctx->norm_gamma_h[cfg.hidden_size - 1] = 2.0f;

    const char* path = "test_rnn_p0_gamma.bin";
    FILE* fp = fopen(path, "wb");
    CHECK(fp != NULL, "fopen write");
    int ok = nn_rnn_save_weights(ctx, fp);
    fclose(fp);
    CHECK(ok == 1, "save failed");

    RnnInferContext* ctx2 = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ctx2 != NULL, "create2 failed");
    fp = fopen(path, "rb");
    CHECK(fp != NULL, "fopen read");
    ok = nn_rnn_load_weights(ctx2, fp);
    fclose(fp);
    CHECK(ok == 1, "load failed");
    CHECK_FEQ(ctx2->norm_gamma_h[0], 0.5f, 1e-6f, "gamma[0] mismatch");
    CHECK_FEQ(ctx2->norm_gamma_h[cfg.hidden_size - 1], 2.0f, 1e-6f, "gamma[last] mismatch");

    nn_rnn_infer_destroy(ctx2);
    nn_rnn_infer_destroy(ctx);
    remove(path);
    return g_fail;
}

/* Test 12: Save/load — forward output matches */
static int test_rnn_save_load_forward(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_rms_norm = 1; cfg.use_dropout = 1; cfg.use_skip = 1;
    cfg.norm_epsilon = 1e-5f; cfg.dropout_rate = 0.1f; cfg.skip_mode = SKIP_IDENTITY;
    RnnInferContext* ctx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ctx != NULL, "create failed");

    const char* path = "test_rnn_p0_fwd.bin";
    FILE* fp = fopen(path, "wb");
    CHECK(fp != NULL, "fopen write");
    int ok = nn_rnn_save_weights(ctx, fp);
    fclose(fp);
    CHECK(ok == 1, "save failed");

    RnnInferContext* ctx2 = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ctx2 != NULL, "create2 failed");
    fp = fopen(path, "rb");
    CHECK(fp != NULL, "fopen read");
    ok = nn_rnn_load_weights(ctx2, fp);
    fclose(fp);
    CHECK(ok == 1, "load failed");

    float in[12]; float oa[3], ob[3];
    fill_sine(in, 12);
    nn_rnn_forward_pass(ctx, in, oa, NULL, NULL);
    nn_rnn_forward_pass(ctx2, in, ob, NULL, NULL);
    for (size_t i = 0; i < 3; i++) CHECK_FEQ(oa[i], ob[i], 1e-5f, "output mismatch");

    nn_rnn_infer_destroy(ctx2);
    nn_rnn_infer_destroy(ctx);
    remove(path);
    return g_fail;
}

/* Test 13: P0 gradient non-zero after training step */
static int test_rnn_gradient_nonzero(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_rms_norm = 1; cfg.norm_epsilon = 1e-5f;
    RnnInferContext* ictx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ictx != NULL, "create failed");

    RnnTrainConfig tcfg; init_rnn_train_cfg(&tcfg);
    tcfg.learning_rate = 0.1f;
    RnnTrainContext* tctx = nn_rnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create failed");

    float in[12]; float target[3];
    fill_sine(in, 12); make_onehot(target, 3, 0);
    nn_rnn_train_step_with_data(tctx, in, target);

    float gsum = 0.0f;
    for (size_t i = 0; i < cfg.hidden_size; i++) gsum += fabsf(tctx->norm_gamma_grad[i]);
    CHECK(gsum > 0.0f, "norm_gamma_grad all zero");

    nn_rnn_train_destroy(tctx);
    nn_rnn_infer_destroy(ictx);
    return g_fail;
}

/* Test 14: Multiple forward passes are deterministic (after warm-up) */
static int test_rnn_multiple_passes(void) {
    RnnConfig cfg; init_rnn_cfg(&cfg);
    cfg.use_rms_norm = 1; cfg.use_dropout = 1; cfg.use_skip = 1;
    cfg.norm_epsilon = 1e-5f; cfg.dropout_rate = 0.2f; cfg.skip_mode = SKIP_IDENTITY;
    RnnInferContext* ctx = nn_rnn_infer_create_with_config(&cfg, cfg.seed);
    CHECK(ctx != NULL, "create failed");

    float in[12]; float out[3];
    fill_sine(in, 12);
    /* Warm-up */
    nn_rnn_forward_pass(ctx, in, out, NULL, NULL);
    float baseline[3];
    memcpy(baseline, out, sizeof(out));
    for (int pass = 1; pass < 10; pass++) {
        nn_rnn_forward_pass(ctx, in, out, NULL, NULL);
        for (size_t i = 0; i < 3; i++)
            CHECK_FEQ(out[i], baseline[i], 1e-6f, "multi-pass not deterministic");
    }
    nn_rnn_infer_destroy(ctx);
    return g_fail;
}

/* ═══════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════ */
int main(void) {
    TestCase tests[] = {
        {"rnn_backward_compat",         test_rnn_backward_compat},
        {"rnn_rms_norm_forward",        test_rnn_rms_norm_forward},
        {"rnn_rms_norm_training",       test_rnn_rms_norm_training},
        {"rnn_rms_norm_gamma_drift",    test_rnn_rms_norm_gamma_drift},
        {"rnn_dropout_forward",         test_rnn_dropout_forward},
        {"rnn_dropout_training",        test_rnn_dropout_training},
        {"rnn_skip_identity",           test_rnn_skip_identity},
        {"rnn_skip_laurel_rw",          test_rnn_skip_laurel_rw},
        {"rnn_combined_all",            test_rnn_combined_all},
        {"rnn_loss_decreases",          test_rnn_loss_decreases},
        {"rnn_save_load_norm_gamma",    test_rnn_save_load_norm_gamma},
        {"rnn_save_load_forward",       test_rnn_save_load_forward},
        {"rnn_gradient_nonzero",        test_rnn_gradient_nonzero},
        {"rnn_multiple_passes",         test_rnn_multiple_passes},
    };
    size_t total = sizeof(tests) / sizeof(tests[0]);
    size_t passed = 0;
    for (size_t i = 0; i < total; i++) {
        g_fail = 0;
        printf("[%2zu/%zu] %s ... ", i + 1, total, tests[i].name);
        int r = tests[i].func();
        if (r == 0) { printf("PASSED\n"); passed++; }
        else { printf("FAILED (%d failures)\n", r); fflush(stdout); }
    }
    printf("\n=== %zu/%zu TESTS PASSED ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
