/**
 * @file test_gnn_p0.c
 * @brief P0 shared-module integration tests for GNN (14 tests).
 *
 * Covers RMSNorm, Dropout, SkipConnection (LAuReL-RW) integration
 * in the GNN backend across forward, backward, save/load, and edge cases.
 */

#include "../src/nn/types/gnn/gnn_config.h"
#include "../src/nn/types/gnn/gnn_infer_ops.h"
#include "../src/nn/types/gnn/gnn_train_ops.h"

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

/* ── GNN config helpers ─────────────────────────────────────────────── */
#define GNN_NODES  4
#define GNN_SLOTS  2
#define GNN_FEAT   3
#define GNN_HIDDEN 6
#define GNN_OUTPUT 3
#define GNN_PASSES 2

static GnnConfig* make_gnn_config(void) {
    size_t sz = gnn_config_size_for_topology(GNN_NODES, GNN_SLOTS);
    GnnConfig* cfg = (GnnConfig*)calloc(1, sz);
    if (!cfg) return NULL;
    cfg->node_count              = GNN_NODES;
    cfg->node_feature_size       = GNN_FEAT;
    cfg->hidden_size             = GNN_HIDDEN;
    cfg->output_size             = GNN_OUTPUT;
    cfg->message_passes          = GNN_PASSES;
    cfg->slot_count              = GNN_SLOTS;
    cfg->node_mask_feature_index = (size_t)(-1);
    cfg->primary_anchor_feature_index = (size_t)(-1);
    cfg->secondary_anchor_feature_index = (size_t)(-1);
    cfg->aggregator_type         = GNN_AGG_MEAN;
    cfg->readout_type            = GNN_READOUT_GRAPH_POOL;
    cfg->hidden_activation       = GNN_ACT_TANH;
    cfg->output_activation       = GNN_ACT_NONE;
    cfg->seed                    = 42;
    /* Simple chain: 0-1-2-3 */
    cfg->neighbor_index[0] = 1;  cfg->neighbor_index[1] = 2;
    cfg->neighbor_index[2] = 0;  cfg->neighbor_index[3] = 3;
    cfg->neighbor_index[4] = 0;  cfg->neighbor_index[5] = 1;
    cfg->neighbor_index[6] = 1;  cfg->neighbor_index[7] = 2;
    return cfg;
}

static GnnTrainConfig make_gnn_train_cfg(void) {
    GnnTrainConfig t; memset(&t, 0, sizeof(t));
    t.learning_rate = 0.01f; t.momentum = 0.0f;
    t.weight_decay = 0.0f;   t.batch_size = 1; t.seed = 42;
    return t;
}

static int gnn_run_forward(GnnInferContext* ctx, const float* in, float* out) {
    nn_gnn_infer_set_input(ctx, in, GNN_NODES * GNN_FEAT);
    int rc = nn_gnn_infer_step(ctx);
    if (rc != 0) return rc;
    nn_gnn_infer_get_output(ctx, out, GNN_OUTPUT);
    return 0;
}

static int gnn_train_one_step(GnnTrainContext* tctx, const float* in, const float* tgt) {
    return nn_gnn_train_step_with_data(tctx, in, tgt);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Test 1: Backward compatibility — P0 disabled
 * ═══════════════════════════════════════════════════════════════════════ */
static int test_gnn_backward_compat(void) {
    GnnConfig* cfg = make_gnn_config();
    GnnInferContext* ctx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ctx != NULL, "create failed");
    CHECK(ctx->norm_gamma_h == NULL, "gamma allocated when P0 off");
    CHECK(!ctx->use_dropout, "dropout on when P0 off");
    CHECK(!ctx->use_skip, "skip on when P0 off");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float out[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT);
    int rc = gnn_run_forward(ctx, in, out);
    CHECK(rc == 0, "forward P0-off failed");
    CHECK(!has_nan(out, GNN_OUTPUT), "output NaN P0-off");
    int nonzero = 0;
    for (size_t i = 0; i < GNN_OUTPUT; i++)
        if (fabsf(out[i]) > 1e-6f) nonzero = 1;
    CHECK(nonzero, "output all zero P0-off");

    free(in); nn_gnn_infer_destroy(ctx); free(cfg);
    return g_fail;
}

/* Test 2: RMSNorm forward — no NaN, gamma=1.0 */
static int test_gnn_rms_norm_forward(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_rms_norm = 1; cfg->norm_epsilon = 1e-5f;
    GnnInferContext* ctx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ctx != NULL, "create rms_norm failed");
    CHECK(ctx->norm_gamma_h != NULL, "norm_gamma_h not allocated");
    CHECK_FEQ(ctx->norm_gamma_h[0], 1.0f, 1e-6f, "gamma[0]!=1");
    CHECK(ctx->use_rms_norm, "rms_norm flag not set");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float out[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT);
    int rc = gnn_run_forward(ctx, in, out);
    CHECK(rc == 0, "forward rms_norm failed");
    CHECK(!has_nan(out, GNN_OUTPUT), "output NaN rms_norm");

    free(in); nn_gnn_infer_destroy(ctx); free(cfg);
    return g_fail;
}

/* Test 3: RMSNorm training — no crash, buffers allocated */
static int test_gnn_rms_norm_training(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_rms_norm = 1; cfg->norm_epsilon = 1e-5f;
    GnnInferContext* ictx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "create failed");

    GnnTrainConfig tcfg = make_gnn_train_cfg();
    GnnTrainContext* tctx = nn_gnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create rms_norm failed");
    CHECK(tctx->norm_gamma_grad != NULL, "norm_gamma_grad missing");
    CHECK(tctx->pre_norm_cache != NULL, "pre_norm_cache missing");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float target[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT); make_onehot(target, GNN_OUTPUT, 0);
    for (int i = 0; i < 5; i++) {
        int rc = gnn_train_one_step(tctx, in, target);
        CHECK(rc == 0, "train_step rms_norm failed");
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }
    CHECK(tctx->last_loss > 0.0f, "loss non-positive");

    free(in); nn_gnn_train_destroy(tctx); nn_gnn_infer_destroy(ictx); free(cfg);
    return g_fail;
}

/* Test 4: RMSNorm gamma drifts after training */
static int test_gnn_rms_norm_gamma_drift(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_rms_norm = 1; cfg->norm_epsilon = 1e-5f;
    GnnInferContext* ictx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "create failed");

    GnnTrainConfig tcfg = make_gnn_train_cfg();
    tcfg.learning_rate = 0.1f;
    GnnTrainContext* tctx = nn_gnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create failed");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float target[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT); make_onehot(target, GNN_OUTPUT, 0);
    for (int i = 0; i < 50; i++) {
        gnn_train_one_step(tctx, in, target);
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }
    for (size_t i = 0; i < GNN_HIDDEN; i++) {
        CHECK(!is_nan(ictx->norm_gamma_h[i]), "gamma NaN");
        CHECK(ictx->norm_gamma_h[i] > 0.0f, "gamma non-positive");
    }
    int drifted = 0;
    for (size_t i = 0; i < GNN_HIDDEN; i++)
        if (fabsf(ictx->norm_gamma_h[i] - 1.0f) > 1e-6f) { drifted = 1; break; }
    CHECK(drifted, "gamma did not drift");

    free(in); nn_gnn_train_destroy(tctx); nn_gnn_infer_destroy(ictx); free(cfg);
    return g_fail;
}

/* Test 5: Dropout forward — no crash */
static int test_gnn_dropout_forward(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_dropout = 1; cfg->dropout_rate = 0.3f;
    GnnInferContext* ctx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ctx != NULL, "create dropout failed");
    CHECK_FEQ(ctx->dropout_msg.keep_prob, 0.3f, 1e-6f, "keep_prob mismatch");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float out[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT);
    int rc = gnn_run_forward(ctx, in, out);
    CHECK(rc == 0, "forward dropout failed");
    CHECK(!has_nan(out, GNN_OUTPUT), "output NaN dropout");

    free(in); nn_gnn_infer_destroy(ctx); free(cfg);
    return g_fail;
}

/* Test 6: Dropout forward — no NaN, keep_prob matches */
static int test_gnn_dropout_training(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_dropout = 1; cfg->dropout_rate = 0.3f;
    GnnInferContext* ictx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "create failed");
    CHECK_FEQ(ictx->dropout_msg.keep_prob, 0.3f, 1e-6f, "keep_prob mismatch");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float out[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT);
    int rc = gnn_run_forward(ictx, in, out);
    CHECK(rc == 0, "forward dropout failed");
    CHECK(!has_nan(out, GNN_OUTPUT), "output NaN dropout");

    free(in); nn_gnn_infer_destroy(ictx); free(cfg);
    return g_fail;
}

/* Test 7: Skip IDENTITY forward and flag checks */
static int test_gnn_skip_identity(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_skip = 1; cfg->skip_mode = SKIP_IDENTITY;
    GnnInferContext* ictx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "create skip failed");
    CHECK(ictx->skip_msg.mode == SKIP_IDENTITY, "skip mode mismatch");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float out[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT);
    int rc = gnn_run_forward(ictx, in, out);
    CHECK(rc == 0, "forward skip failed");
    CHECK(!has_nan(out, GNN_OUTPUT), "output NaN skip");

    free(in); nn_gnn_infer_destroy(ictx); free(cfg);
    return g_fail;
}

/* Test 8: Skip LAuReL-RW — alpha_raw starts at 0, forward works */
static int test_gnn_skip_laurel_rw(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_skip = 1; cfg->skip_mode = SKIP_LAUREL_RW;
    GnnInferContext* ictx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "create laurel failed");
    CHECK_FEQ(ictx->skip_msg.alpha_raw, 0.0f, 1e-6f, "alpha_raw not 0");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float out[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT);
    int rc = gnn_run_forward(ictx, in, out);
    CHECK(rc == 0, "forward laurel failed");
    CHECK(!has_nan(out, GNN_OUTPUT), "output NaN laurel");

    free(in); nn_gnn_infer_destroy(ictx); free(cfg);
    return g_fail;
}

/* Test 9: Combined — RMSNorm + Dropout + Skip LAuReL-RW */
static int test_gnn_combined_all(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_rms_norm = 1; cfg->norm_epsilon = 1e-5f;
    cfg->use_dropout  = 1; cfg->dropout_rate  = 0.2f;
    cfg->use_skip     = 1; cfg->skip_mode     = SKIP_LAUREL_RW;
    GnnInferContext* ictx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "create combined failed");
    CHECK(ictx->norm_gamma_h != NULL, "gamma missing");
    CHECK(ictx->skip_msg.mode == SKIP_LAUREL_RW, "skip mode wrong");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float out[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT);
    int rc = gnn_run_forward(ictx, in, out);
    CHECK(rc == 0, "forward combined failed");
    CHECK(!has_nan(out, GNN_OUTPUT), "output NaN combined");

    GnnTrainConfig tcfg = make_gnn_train_cfg();
    GnnTrainContext* tctx = nn_gnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create combined failed");
    CHECK(tctx->norm_gamma_grad != NULL, "gamma grad missing");
    CHECK(tctx->skip_temp_dF != NULL, "skip temp missing");

    float target[GNN_OUTPUT];
    make_onehot(target, GNN_OUTPUT, 0);
    float first_loss = -1;
    for (int i = 0; i < 10; i++) {
        gnn_train_one_step(tctx, in, target);
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
        if (i == 0) first_loss = tctx->last_loss;
    }
    CHECK(first_loss > 0.0f, "loss non-positive");

    for (size_t i = 0; i < GNN_HIDDEN; i++)
        CHECK(ictx->norm_gamma_h[i] > 0.0f, "gamma non-positive after");

    free(in); nn_gnn_train_destroy(tctx); nn_gnn_infer_destroy(ictx); free(cfg);
    return g_fail;
}

/* Test 10: Loss decreases over training steps */
static int test_gnn_loss_decreases(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_rms_norm = 1; cfg->norm_epsilon = 1e-5f;
    GnnInferContext* ictx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "create failed");

    GnnTrainConfig tcfg = make_gnn_train_cfg();
    tcfg.learning_rate = 0.1f;
    GnnTrainContext* tctx = nn_gnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create failed");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float target[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT); make_onehot(target, GNN_OUTPUT, 0);
    gnn_train_one_step(tctx, in, target);
    float l0 = tctx->last_loss;
    for (int i = 0; i < 50; i++) gnn_train_one_step(tctx, in, target);
    CHECK(tctx->last_loss < l0, "loss did not decrease");
    CHECK(!is_nan(tctx->last_loss), "final loss NaN");

    free(in); nn_gnn_train_destroy(tctx); nn_gnn_infer_destroy(ictx); free(cfg);
    return g_fail;
}

/* Test 11: Save/load preserves norm_gamma */
static int test_gnn_save_load_norm_gamma(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_rms_norm = 1; cfg->norm_epsilon = 1e-5f;
    GnnInferContext* ctx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ctx != NULL, "create failed");
    ctx->norm_gamma_h[0] = 0.5f;
    ctx->norm_gamma_h[GNN_HIDDEN - 1] = 2.0f;

    const char* path = "test_gnn_p0_gamma.bin";
    FILE* fp = fopen(path, "wb");
    CHECK(fp != NULL, "fopen write");
    int ok = nn_gnn_save_weights(ctx, fp);
    fclose(fp);
    CHECK(ok == 1, "save failed");

    GnnInferContext* ctx2 = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ctx2 != NULL, "create2 failed");
    fp = fopen(path, "rb");
    CHECK(fp != NULL, "fopen read");
    ok = nn_gnn_load_weights(ctx2, fp);
    fclose(fp);
    CHECK(ok == 1, "load failed");
    CHECK_FEQ(ctx2->norm_gamma_h[0], 0.5f, 1e-6f, "gamma[0] mismatch");
    CHECK_FEQ(ctx2->norm_gamma_h[GNN_HIDDEN - 1], 2.0f, 1e-6f, "gamma[last] mismatch");

    nn_gnn_infer_destroy(ctx2); nn_gnn_infer_destroy(ctx);
    free(cfg); remove(path);
    return g_fail;
}

/* Test 12: Save/load — forward output matches */
static int test_gnn_save_load_forward(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_rms_norm = 1; cfg->use_dropout = 1; cfg->use_skip = 1;
    cfg->norm_epsilon = 1e-5f; cfg->dropout_rate = 0.1f; cfg->skip_mode = SKIP_IDENTITY;
    GnnInferContext* ctx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ctx != NULL, "create failed");

    const char* path = "test_gnn_p0_fwd.bin";
    FILE* fp = fopen(path, "wb");
    CHECK(fp != NULL, "fopen write");
    int ok = nn_gnn_save_weights(ctx, fp);
    fclose(fp);
    CHECK(ok == 1, "save failed");

    GnnInferContext* ctx2 = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ctx2 != NULL, "create2 failed");
    fp = fopen(path, "rb");
    CHECK(fp != NULL, "fopen read");
    ok = nn_gnn_load_weights(ctx2, fp);
    fclose(fp);
    CHECK(ok == 1, "load failed");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float oa[GNN_OUTPUT], ob[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT);
    gnn_run_forward(ctx, in, oa);
    gnn_run_forward(ctx2, in, ob);
    for (size_t i = 0; i < GNN_OUTPUT; i++) CHECK_FEQ(oa[i], ob[i], 1e-5f, "output mismatch");

    free(in); nn_gnn_infer_destroy(ctx2); nn_gnn_infer_destroy(ctx);
    free(cfg); remove(path);
    return g_fail;
}

/* Test 13: P0 gradient non-zero after training step */
static int test_gnn_gradient_nonzero(void) {
    GnnConfig* cfg = make_gnn_config();
    cfg->use_rms_norm = 1; cfg->norm_epsilon = 1e-5f;
    GnnInferContext* ictx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "create failed");

    GnnTrainConfig tcfg = make_gnn_train_cfg();
    tcfg.learning_rate = 0.1f;
    GnnTrainContext* tctx = nn_gnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create failed");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    float target[GNN_OUTPUT];
    fill_sine(in, GNN_NODES * GNN_FEAT); make_onehot(target, GNN_OUTPUT, 0);
    gnn_train_one_step(tctx, in, target);

    float gsum = 0.0f;
    for (size_t i = 0; i < GNN_HIDDEN; i++) gsum += fabsf(tctx->norm_gamma_grad[i]);
    CHECK(gsum > 0.0f, "norm_gamma_grad all zero");

    free(in); nn_gnn_train_destroy(tctx); nn_gnn_infer_destroy(ictx); free(cfg);
    return g_fail;
}

/* Test 14: Anchor-slots readout with P0 */
static int test_gnn_anchor_slots_p0(void) {
    /* Build a config with ANCHOR_SLOTS readout */
    GnnConfig* cfg = make_gnn_config();
    cfg->readout_type                 = GNN_READOUT_ANCHOR_SLOTS;
    cfg->primary_anchor_feature_index = 0; /* use feature[0] as anchor selector */
    cfg->use_rms_norm                 = 1;
    cfg->norm_epsilon                 = 1e-5f;

    /* Make first node the anchor by giving it a very high feature[0] */
    GnnInferContext* ictx = nn_gnn_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "create anchor failed");
    CHECK(ictx->norm_gamma_h != NULL, "gamma missing");

    float* in = (float*)calloc(GNN_NODES * GNN_FEAT, sizeof(float));
    /* Node 0: high feature[0], rest sine */
    in[0] = 100.0f;
    fill_sine(in, GNN_NODES * GNN_FEAT);

    float out[GNN_OUTPUT];
    int rc = gnn_run_forward(ictx, in, out);
    CHECK(rc == 0, "forward anchor failed");
    CHECK(!has_nan(out, GNN_OUTPUT), "output NaN anchor");

    GnnTrainConfig tcfg = make_gnn_train_cfg();
    GnnTrainContext* tctx = nn_gnn_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create anchor failed");

    float target[GNN_OUTPUT];
    make_onehot(target, GNN_OUTPUT, 0);
    for (int i = 0; i < 5; i++) {
        rc = gnn_train_one_step(tctx, in, target);
        CHECK(rc == 0, "train_step anchor failed");
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }

    free(in); nn_gnn_train_destroy(tctx); nn_gnn_infer_destroy(ictx); free(cfg);
    return g_fail;
}

/* ═══════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════ */
int main(void) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    TestCase tests[] = {
        {"gnn_backward_compat",         test_gnn_backward_compat},
        {"gnn_rms_norm_forward",        test_gnn_rms_norm_forward},
        {"gnn_rms_norm_training",       test_gnn_rms_norm_training},
        {"gnn_rms_norm_gamma_drift",    test_gnn_rms_norm_gamma_drift},
        {"gnn_dropout_forward",         test_gnn_dropout_forward},
        {"gnn_dropout_training",        test_gnn_dropout_training},
        {"gnn_skip_identity",           test_gnn_skip_identity},
        {"gnn_skip_laurel_rw",          test_gnn_skip_laurel_rw},
        {"gnn_combined_all",            test_gnn_combined_all},
        {"gnn_loss_decreases",          test_gnn_loss_decreases},
        {"gnn_save_load_norm_gamma",    test_gnn_save_load_norm_gamma},
        {"gnn_save_load_forward",       test_gnn_save_load_forward},
        {"gnn_gradient_nonzero",        test_gnn_gradient_nonzero},
        {"gnn_anchor_slots_p0",         test_gnn_anchor_slots_p0},
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
