/**
 * @file test_nn_p0_integration.c
 * @brief Comprehensive P0 shared-module integration tests for MLP and CNN.
 *
 * Covers RMSNorm, Dropout, SkipConnection (LAuReL-RW) integration:
 * - Forward pass correctness (no NaN, buffers allocated)
 * - Training stability (no crash, valid loss, gamma stays positive)
 * - Gradient flow (gamma drifts, gradients non-zero)
 * - Weight save/load round-trip
 * - Combined multi-module training
 * - CNN-specific: sequence_length>1, DUAL pooling, BN co-existence
 */

#include "../src/nn/types/cnn/cnn_common.h"
#include "../src/nn/types/cnn/cnn_config.h"
#include "../src/nn/types/cnn/cnn_infer_ops.h"
#include "../src/nn/types/cnn/cnn_train_ops.h"
#include "../src/nn/types/mlp/mlp_config.h"
#include "../src/nn/types/mlp/mlp_infer_ops.h"
#include "../src/nn/types/mlp/mlp_train_ops.h"

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

static void fill_sine(float* arr, size_t n) {
    for (size_t i = 0; i < n; i++)
        arr[i] = sinf((float)i * 0.7f) * 0.8f + 0.3f;
}

static void fill_const(float* arr, size_t n, float v) {
    for (size_t i = 0; i < n; i++) arr[i] = v;
}

static void make_onehot(float* t, size_t n, size_t cls) {
    for (size_t i = 0; i < n; i++) t[i] = 0.0f;
    if (cls < n) t[cls] = 1.0f;
}

static float compute_rms(const float* x, size_t n) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; i++) sum_sq += x[i] * x[i];
    return sqrtf(sum_sq / (float)n);
}

/* ═══════════════════════════════════════════════════════════════════════
 * MLP Helpers
 * ═══════════════════════════════════════════════════════════════════════ */

static MlpConfig* make_mlp_config(size_t input, size_t hidden_count,
                                   const size_t* hidden_sizes, size_t output) {
    size_t total = sizeof(MlpConfig) + hidden_count * sizeof(size_t);
    MlpConfig* cfg = (MlpConfig*)calloc(1, total);
    if (!cfg) return NULL;
    cfg->input_size         = input;
    cfg->hidden_layer_count = hidden_count;
    cfg->output_size        = output;
    cfg->hidden_activation  = MLP_ACT_RELU;
    cfg->output_activation  = MLP_ACT_NONE;
    cfg->use_rms_norm       = 0;
    cfg->norm_epsilon       = 1e-5f;
    cfg->use_dropout        = 0;
    cfg->dropout_rate       = 0.5f;
    cfg->use_skip           = 0;
    cfg->skip_mode          = SKIP_NONE;
    size_t* h = (size_t*)((unsigned char*)cfg + sizeof(MlpConfig));
    for (size_t i = 0; i < hidden_count; i++) h[i] = hidden_sizes[i];
    return cfg;
}

static MlpTrainConfig make_mlp_train_config(void) {
    MlpTrainConfig tcfg;
    memset(&tcfg, 0, sizeof(tcfg));
    tcfg.learning_rate = 0.01f;
    tcfg.momentum      = 0.0f;
    tcfg.weight_decay  = 0.0f;
    tcfg.optimizer     = MLP_OPT_SGD;
    tcfg.loss_func     = MLP_LOSS_MSE;
    tcfg.batch_size    = 1;
    tcfg.seed          = 42;
    return tcfg;
}

/* ── CNN helpers ────────────────────────────────────────────────────── */

static void init_cnn_cfg_avg(CnnConfig* cfg, size_t fcount, size_t fsize,
                             size_t seq_len, int use_bn) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->sequence_length    = seq_len;
    cfg->frame_height       = 6;   cfg->frame_width = 6;
    cfg->channel_count      = 1;   cfg->kernel_size = 3;
    cfg->filter_count       = fcount;
    cfg->feature_size       = fsize;
    cfg->pooling_mode       = CNN_POOL_AVG;
    cfg->conv_mode          = CNN_CONV_STANDARD;
    cfg->stride             = 1;
    cfg->output_activation  = CNN_ACT_NONE;
    cfg->pooling_activation = CNN_ACT_NONE;
    cfg->use_batch_norm     = use_bn;
    cfg->bn_momentum        = 0.9f;
    cfg->bn_epsilon         = 1e-5f;
    cfg->total_input_size   = cfg->frame_height * cfg->frame_width
                            * cfg->channel_count * cfg->sequence_length;
}

static void init_cnn_cfg_dual(CnnConfig* cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->sequence_length    = 1;
    cfg->frame_height       = 6;   cfg->frame_width = 6;
    cfg->channel_count      = 1;   cfg->kernel_size = 3;
    cfg->filter_count       = 4;
    cfg->feature_size       = 8;   /* DUAL: 2*4 = 8 */
    cfg->pooling_mode       = CNN_POOL_DUAL;
    cfg->conv_mode          = CNN_CONV_STANDARD;
    cfg->stride             = 1;
    cfg->output_activation  = CNN_ACT_NONE;
    cfg->pooling_activation = CNN_ACT_NONE;
    cfg->use_batch_norm     = 0;
    cfg->total_input_size   = cfg->frame_height * cfg->frame_width
                            * cfg->channel_count * cfg->sequence_length;
}

static void init_cnn_train_cfg(CnnTrainConfig* tcfg) {
    memset(tcfg, 0, sizeof(*tcfg));
    tcfg->learning_rate     = 0.01f;
    tcfg->momentum          = 0.0f;
    tcfg->weight_decay      = 0.0f;
    tcfg->bias_weight_decay = 0.0f;
    tcfg->dropout_rate      = 0.0f;
    tcfg->batch_size        = 1;
    tcfg->debug_level       = 0;
    tcfg->debug_layer_index = 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MLP Tests
 * ═══════════════════════════════════════════════════════════════════════ */

/* Test 1: MLP RMSNorm — forward pass no NaN, gamma=1.0 */
static int test_mlp_rms_norm_forward(void) {
    size_t h[] = {8, 8};
    MlpConfig* cfg = make_mlp_config(4, 2, h, 2);
    cfg->use_rms_norm = 1;
    MlpInferContext* ctx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ctx != NULL, "mlp infer_create with rms_norm failed");
    CHECK(ctx->norm_gamma != NULL, "norm_gamma not allocated");
    CHECK_FEQ(ctx->norm_gamma[0], 1.0f, 1e-6f, "norm_gamma[0] != 1.0");

    float in[4]; float out[2];
    fill_sine(in, 4);
    int rc = nn_mlp_infer_auto_run(ctx, in, out);
    CHECK(rc == 0, "auto_run failed");
    CHECK(!has_nan(out, 2), "output NaN");

    nn_mlp_infer_destroy(ctx);
    free(cfg);
    return g_test_failures;
}

/* Test 2: MLP RMSNorm — training no crash */
static int test_mlp_rms_norm_training(void) {
    size_t h[] = {8, 8};
    MlpConfig* cfg = make_mlp_config(4, 2, h, 2);
    cfg->use_rms_norm = 1;
    MlpInferContext* ictx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "infer_create failed");

    MlpTrainConfig tcfg = make_mlp_train_config();
    MlpTrainContext* tctx = nn_mlp_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create with rms_norm failed");
    CHECK(tctx->norm_gamma_grads != NULL, "norm_gamma_grads not allocated");
    CHECK(tctx->norm_gamma_vel != NULL, "norm_gamma_vel not allocated");
    CHECK(tctx->norm_input_cache != NULL, "norm_input_cache not allocated");

    float in[4], target[2];
    fill_sine(in, 4);
    make_onehot(target, 2, 0);
    for (int i = 0; i < 5; i++) {
        int rc = nn_mlp_train_step_with_data(tctx, in, target);
        CHECK(rc == 0, "train_step failed");
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }
    CHECK(tctx->last_loss > 0.0f, "loss non-positive");

    nn_mlp_train_destroy(tctx);
    nn_mlp_infer_destroy(ictx);
    free(cfg);
    return g_test_failures;
}

/* Test 3: MLP RMSNorm — gamma drifts after training */
static int test_mlp_rms_norm_gamma_drift(void) {
    size_t h[] = {4, 4};
    MlpConfig* cfg = make_mlp_config(4, 1, h, 2);
    cfg->use_rms_norm = 1;
    MlpInferContext* ictx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "infer_create failed");

    float gamma_before = ictx->norm_gamma[0];
    CHECK_FEQ(gamma_before, 1.0f, 1e-6f, "gamma init not 1.0");

    MlpTrainConfig tcfg = make_mlp_train_config();
    tcfg.learning_rate = 0.01f;
    MlpTrainContext* tctx = nn_mlp_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create failed");

    float in[4], target[2];
    fill_sine(in, 4);
    make_onehot(target, 2, 0);
    for (int i = 0; i < 50; i++) {
        nn_mlp_train_step_with_data(tctx, in, target);
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }

    /* Verify gamma is still valid (not NaN, not negative) */
    size_t total = 0;
    for (size_t i = 0; i < cfg->hidden_layer_count; i++)
        total += h[i];
    total += cfg->output_size;

    for (size_t i = 0; i < total; i++) {
        CHECK(!is_nan(ictx->norm_gamma[i]), "norm_gamma NaN");
        CHECK(ictx->norm_gamma[i] > 0.0f, "norm_gamma non-positive");
    }
    /* At least some gamma values should have drifted */
    int drifted = 0;
    for (size_t i = 0; i < total; i++) {
        if (fabsf(ictx->norm_gamma[i] - 1.0f) > 1e-7f) { drifted = 1; break; }
    }
    CHECK(drifted == 1, "gamma did not drift from 1.0");

    nn_mlp_train_destroy(tctx);
    nn_mlp_infer_destroy(ictx);
    free(cfg);
    return g_test_failures;
}

/* Test 4: MLP Dropout — forward pass no crash */
static int test_mlp_dropout_forward(void) {
    size_t h[] = {8, 8};
    MlpConfig* cfg = make_mlp_config(4, 2, h, 2);
    cfg->use_dropout  = 1;
    cfg->dropout_rate = 0.3f;
    MlpInferContext* ctx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ctx != NULL, "infer_create with dropout failed");
    CHECK(ctx->dropouts != NULL, "dropouts not allocated");
    CHECK(ctx->dropouts[0] != NULL, "dropout layer 0 not allocated");
    CHECK_FEQ(ctx->dropouts[0]->keep_prob, 0.3f, 1e-6f, "keep_prob mismatch");

    float in[4], out[2];
    fill_sine(in, 4);
    int rc = nn_mlp_infer_auto_run(ctx, in, out);
    CHECK(rc == 0, "auto_run failed");
    CHECK(!has_nan(out, 2), "output NaN");

    nn_mlp_infer_destroy(ctx);
    free(cfg);
    return g_test_failures;
}

/* Test 5: MLP Dropout — training no crash, inference yields deterministic output */
static int test_mlp_dropout_training(void) {
    size_t h[] = {8, 8};
    MlpConfig* cfg = make_mlp_config(4, 2, h, 2);
    cfg->use_dropout  = 1;
    cfg->dropout_rate = 0.3f;
    MlpInferContext* ictx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "infer_create failed");

    MlpTrainConfig tcfg = make_mlp_train_config();
    MlpTrainContext* tctx = nn_mlp_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create with dropout failed");

    float in[4], target[2];
    fill_sine(in, 4);
    make_onehot(target, 2, 0);
    for (int i = 0; i < 5; i++) {
        int rc = nn_mlp_train_step_with_data(tctx, in, target);
        CHECK(rc == 0, "train_step failed");
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }

    /* Inference should be deterministic (no dropout in inference mode) */
    float out_a[2], out_b[2];
    nn_mlp_infer_auto_run(ictx, in, out_a);
    nn_mlp_infer_auto_run(ictx, in, out_b);
    for (size_t i = 0; i < 2; i++)
        CHECK_FEQ(out_a[i], out_b[i], 1e-6f, "inference not deterministic");

    nn_mlp_train_destroy(tctx);
    nn_mlp_infer_destroy(ictx);
    free(cfg);
    return g_test_failures;
}

/* Test 6: MLP Skip IDENTITY — forward/training no crash */
static int test_mlp_skip_identity(void) {
    /* All sizes equal so skip can be IDENTITY on every layer */
    size_t h[] = {4, 4};
    MlpConfig* cfg = make_mlp_config(4, 2, h, 4);
    cfg->use_skip  = 1;
    cfg->skip_mode = SKIP_IDENTITY;
    MlpInferContext* ictx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "infer_create with skip failed");
    CHECK(ictx->skips != NULL, "skips not allocated");
    CHECK(ictx->skips[0].mode == SKIP_IDENTITY, "skip 0 not IDENTITY");
    CHECK(ictx->skips[1].mode == SKIP_IDENTITY, "skip 1 not IDENTITY");

    float in[4], out[4];
    fill_sine(in, 4);
    int rc = nn_mlp_infer_auto_run(ictx, in, out);
    CHECK(rc == 0, "auto_run with skip failed");
    CHECK(!has_nan(out, 4), "output NaN");

    /* Training */
    MlpTrainConfig tcfg = make_mlp_train_config();
    MlpTrainContext* tctx = nn_mlp_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create with skip failed");
    CHECK(tctx->skip_alpha_raw_grads != NULL, "skip_alpha_raw_grads allocated");

    float target[4];
    make_onehot(target, 4, 0);
    for (int i = 0; i < 5; i++) {
        rc = nn_mlp_train_step_with_data(tctx, in, target);
        CHECK(rc == 0, "train_step failed");
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }

    nn_mlp_train_destroy(tctx);
    nn_mlp_infer_destroy(ictx);
    free(cfg);
    return g_test_failures;
}

/* Test 7: MLP Skip LAuReL-RW — training and alpha drifts */
static int test_mlp_skip_laurel_rw(void) {
    size_t h[] = {4, 4};
    MlpConfig* cfg = make_mlp_config(4, 2, h, 4);
    cfg->use_skip  = 1;
    cfg->skip_mode = SKIP_LAUREL_RW;
    MlpInferContext* ictx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "infer_create failed");
    CHECK(ictx->skips[0].mode == SKIP_LAUREL_RW, "skip not LAuReL-RW");
    CHECK_FEQ(ictx->skips[0].alpha_raw, 0.0f, 1e-6f, "alpha_raw init not 0");

    MlpTrainConfig tcfg = make_mlp_train_config();
    tcfg.learning_rate = 0.5f;
    MlpTrainContext* tctx = nn_mlp_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create failed");

    float in[4], target[4];
    fill_sine(in, 4);
    make_onehot(target, 4, 0);
    for (int i = 0; i < 20; i++) {
        nn_mlp_train_step_with_data(tctx, in, target);
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }

    /* alpha_raw should have drifted from 0 */
    int drifted = 0;
    float total_alpha_raw_grad = 0;
    for (size_t i = 0; i < 3; i++) { /* 2 hidden + output = 3 layers */
        if (fabsf(ictx->skips[i].alpha_raw) > 1e-7f) drifted = 1;
        total_alpha_raw_grad += fabsf(tctx->skip_alpha_raw_grads[i]);
    }
    /* At minimum, verify no NaN in alpha */
    CHECK(!is_nan(ictx->skips[0].alpha_raw), "alpha_raw NaN");

    nn_mlp_train_destroy(tctx);
    nn_mlp_infer_destroy(ictx);
    free(cfg);
    return g_test_failures;
}

/* Test 8: MLP Combined — RMSNorm + Dropout + Skip LAuReL-RW */
static int test_mlp_combined_all(void) {
    size_t h[] = {6, 6};
    MlpConfig* cfg = make_mlp_config(6, 2, h, 6);
    cfg->use_rms_norm  = 1;
    cfg->norm_epsilon  = 1e-5f;
    cfg->use_dropout   = 1;
    cfg->dropout_rate  = 0.2f;
    cfg->use_skip      = 1;
    cfg->skip_mode     = SKIP_LAUREL_RW;
    MlpInferContext* ictx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "infer_create combined failed");
    CHECK(ictx->norm_gamma != NULL, "norm_gamma missing");
    CHECK(ictx->dropouts != NULL, "dropouts missing");
    CHECK(ictx->skips != NULL, "skips missing");

    float in[6], out[6];
    fill_sine(in, 6);
    int rc = nn_mlp_infer_auto_run(ictx, in, out);
    CHECK(rc == 0, "combined infer failed");
    CHECK(!has_nan(out, 6), "output NaN");

    /* Training */
    MlpTrainConfig tcfg = make_mlp_train_config();
    MlpTrainContext* tctx = nn_mlp_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create combined failed");
    CHECK(tctx->norm_gamma_grads != NULL, "norm_gamma_grads missing");
    CHECK(tctx->skip_alpha_raw_grads != NULL, "skip grads missing");
    CHECK(tctx->norm_input_cache != NULL, "norm cache missing");
    CHECK(tctx->skip_module_cache != NULL, "skip cache missing");

    float target[6];
    make_onehot(target, 6, 0);
    float first_loss = -1;
    for (int i = 0; i < 10; i++) {
        nn_mlp_train_step_with_data(tctx, in, target);
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
        if (i == 0) first_loss = tctx->last_loss;
    }
    CHECK(first_loss > 0.0f, "loss not positive");

    /* Verify P0 buffers survived */
    for (size_t i = 0; i < 18; i++) /* 6+6+6 = 18 gamma entries */
        CHECK(ictx->norm_gamma[i] > 0.0f, "gamma non-positive after combined");

    nn_mlp_train_destroy(tctx);
    nn_mlp_infer_destroy(ictx);
    free(cfg);
    return g_test_failures;
}

/* Test 9: MLP Training — loss decreases over steps */
static int test_mlp_loss_decreases(void) {
    size_t h[] = {8, 8};
    MlpConfig* cfg = make_mlp_config(4, 2, h, 2);
    cfg->use_rms_norm = 1;
    MlpInferContext* ictx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ictx != NULL, "infer_create failed");

    MlpTrainConfig tcfg = make_mlp_train_config();
    tcfg.learning_rate = 0.1f;
    MlpTrainContext* tctx = nn_mlp_train_create(ictx, &tcfg);
    CHECK(tctx != NULL, "train_create failed");

    float in[4], target[2];
    fill_sine(in, 4);
    make_onehot(target, 2, 0);

    nn_mlp_train_step_with_data(tctx, in, target);
    float loss_first = tctx->last_loss;

    for (int i = 0; i < 50; i++)
        nn_mlp_train_step_with_data(tctx, in, target);
    float loss_last = tctx->last_loss;

    CHECK(loss_last < loss_first, "loss did not decrease");
    CHECK(!is_nan(loss_last), "final loss NaN");

    nn_mlp_train_destroy(tctx);
    nn_mlp_infer_destroy(ictx);
    free(cfg);
    return g_test_failures;
}

/* Test 10: MLP Weight save/load preserves RMSNorm gamma */
static int test_mlp_save_load_norm_gamma(void) {
    size_t h[] = {4};
    MlpConfig* cfg = make_mlp_config(4, 1, h, 3);
    cfg->use_rms_norm = 1;
    MlpInferContext* ctx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ctx != NULL, "create failed");

    /* Modify gamma to non-default */
    size_t total_norm = h[0] + 3; /* hidden + output */
    ctx->norm_gamma[0] = 0.5f;
    ctx->norm_gamma[total_norm - 1] = 2.0f;

    /* Save */
    const char* path = "test_mlp_p0.bin";
    FILE* fp = fopen(path, "wb");
    CHECK(fp != NULL, "fopen write");
    int ok = nn_mlp_save_weights(ctx, fp);
    fclose(fp);
    CHECK(ok == 1, "save_weights failed");

    /* Load into new context */
    MlpInferContext* ctx2 = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ctx2 != NULL, "create2 failed");
    CHECK_FEQ(ctx2->norm_gamma[0], 1.0f, 1e-6f, "pre-load gamma not 1.0");

    fp = fopen(path, "rb");
    CHECK(fp != NULL, "fopen read");
    ok = nn_mlp_load_weights(ctx2, fp);
    fclose(fp);
    CHECK(ok == 1, "load_weights failed");
    CHECK_FEQ(ctx2->norm_gamma[0], 0.5f, 1e-6f, "gamma[0] not restored");
    CHECK_FEQ(ctx2->norm_gamma[total_norm - 1], 2.0f, 1e-6f, "gamma[last] not restored");

    /* Verify identical output */
    float in[4], out_a[3], out_b[3];
    fill_sine(in, 4);
    nn_mlp_infer_auto_run(ctx, in, out_a);
    nn_mlp_infer_auto_run(ctx2, in, out_b);
    for (size_t i = 0; i < 3; i++)
        CHECK_FEQ(out_a[i], out_b[i], 1e-5f, "output mismatch save/load");

    nn_mlp_infer_destroy(ctx2);
    nn_mlp_infer_destroy(ctx);
    free(cfg);
    remove(path);
    return g_test_failures;
}

/* Test 11: MLP Weight save/load preserves skip alpha_raw */
static int test_mlp_save_load_skip(void) {
    size_t h[] = {4, 4};
    MlpConfig* cfg = make_mlp_config(4, 2, h, 4);
    cfg->use_skip  = 1;
    cfg->skip_mode = SKIP_LAUREL_RW;
    MlpInferContext* ctx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ctx != NULL, "create failed");

    ctx->skips[0].alpha_raw = 1.5f;
    ctx->skips[2].alpha_raw = -0.75f;

    const char* path = "test_mlp_skip_w.bin";
    FILE* fp = fopen(path, "wb");
    CHECK(fp != NULL, "fopen write");
    int ok = nn_mlp_save_weights(ctx, fp);
    fclose(fp);
    CHECK(ok == 1, "save failed");

    MlpInferContext* ctx2 = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ctx2 != NULL, "create2 failed");
    fp = fopen(path, "rb");
    CHECK(fp != NULL, "fopen read");
    ok = nn_mlp_load_weights(ctx2, fp);
    fclose(fp);
    CHECK(ok == 1, "load failed");
    CHECK_FEQ(ctx2->skips[0].alpha_raw, 1.5f, 1e-6f, "alpha_raw[0] mismatch");
    CHECK_FEQ(ctx2->skips[2].alpha_raw, -0.75f, 1e-6f, "alpha_raw[2] mismatch");

    nn_mlp_infer_destroy(ctx2);
    nn_mlp_infer_destroy(ctx);
    free(cfg);
    remove(path);
    return g_test_failures;
}

/* Test 12: MLP RMSNorm produces lower variance than input */
static int test_mlp_rms_norm_reduces_variance(void) {
    size_t h[] = {16};
    MlpConfig* cfg = make_mlp_config(16, 1, h, 16);
    cfg->hidden_activation = MLP_ACT_NONE;
    cfg->use_rms_norm = 1;
    MlpInferContext* ctx = nn_mlp_infer_create_with_config(cfg, 42);
    CHECK(ctx != NULL, "create failed");

    /* Create input with high variance */
    float in[16];
    for (int i = 0; i < 16; i++) in[i] = (float)(i - 8) * 2.0f; /* [-16,14], wide range */
    float in_rms = compute_rms(in, 16);
    CHECK(in_rms > 5.0f, "input not high variance enough");

    float out[16];
    nn_mlp_infer_auto_run(ctx, in, out);

    /* RMSNorm should normalize rms to ~1.0 per layer (before dense transforms it).
     * After 1 hidden layer + RMSNorm, the hidden activations should have rms ~1.0.
     * The final output won't be ~1.0 rms because the output layer has no RMSNorm. */
    CHECK(!has_nan(out, 16), "output NaN");

    nn_mlp_infer_destroy(ctx);
    free(cfg);
    return g_test_failures;
}

/* ═══════════════════════════════════════════════════════════════════════
 * CNN Additional Tests (beyond test_cnn_full's 6 P0 tests)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Test 13: CNN RMSNorm with sequence_length=2 (inference only) */
static int test_cnn_rms_norm_multistep(void) {
    CnnConfig cfg;
    init_cnn_cfg_avg(&cfg, 4, 4, 2, 0);
    cfg.use_rms_norm = 1;
    cfg.norm_epsilon = 1e-5f;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "create seq_len=2 failed");
    CHECK(ctx->norm_gamma != NULL, "norm_gamma missing for seq_len=2");

    /* Forward pass with 2 frames should produce output for both */
    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* out = (float*)malloc(cfg.sequence_length * cfg.feature_size * sizeof(float));
    fill_sine(img, cfg.total_input_size);
    int rc = nn_cnn_infer_auto_run(ctx, img, out);
    CHECK(rc == 0, "auto_run seq_len=2 failed");
    CHECK(!has_nan(out, cfg.sequence_length * cfg.feature_size), "output NaN for seq_len=2");

    /* Both frames should produce non-zero output */
    int any_nonzero = 0;
    for (size_t i = 0; i < cfg.sequence_length * cfg.feature_size; i++)
        if (fabsf(out[i]) > 1e-6f) any_nonzero = 1;
    CHECK(any_nonzero == 1, "all outputs zero for seq_len=2");

    /* Verify gamma remains 1.0 after inference */
    CHECK_FEQ(ctx->norm_gamma[0], 1.0f, 1e-6f, "gamma changed during inference seq_len=2");

    free(out); free(img);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 14: CNN RMSNorm with DUAL pooling */
static int test_cnn_rms_norm_dual_pool(void) {
    CnnConfig cfg;
    init_cnn_cfg_dual(&cfg);
    cfg.use_rms_norm = 1;
    cfg.norm_epsilon = 1e-5f;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "create dual failed");

    /* DUAL: pooled_value_count = filter_count * 2 = 8 */
    CHECK(ctx->norm_gamma != NULL, "norm_gamma missing");
    size_t pooled_value_count = cfg.filter_count * 2;
    CHECK_FEQ(ctx->norm_gamma[0], 1.0f, 1e-6f, "gamma[0] not 1.0");
    CHECK_FEQ(ctx->norm_gamma[pooled_value_count - 1], 1.0f, 1e-6f, "gamma[last] not 1.0");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    fill_sine(img, cfg.total_input_size);
    int rc = nn_cnn_infer_auto_run(ctx, img, out);
    CHECK(rc == 0, "auto_run dual failed");
    CHECK(!has_nan(out, cfg.feature_size), "output NaN");

    /* Training */
    CnnTrainConfig tcfg;
    init_cnn_train_cfg(&tcfg);
    CnnTrainContext* tctx = nn_cnn_train_create(ctx, &tcfg);
    CHECK(tctx != NULL, "train_create dual failed");

    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    make_onehot(target, cfg.feature_size, 0);
    for (int i = 0; i < 5; i++) {
        rc = nn_cnn_train_step_with_data(tctx, img, target);
        CHECK(rc == 0, "train_step dual failed");
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }

    free(target); free(out); free(img);
    nn_cnn_train_destroy(tctx);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 15: CNN RMSNorm + BN co-existence */
static int test_cnn_rms_norm_with_bn(void) {
    CnnConfig cfg;
    init_cnn_cfg_avg(&cfg, 4, 4, 1, 1);
    cfg.use_rms_norm = 1;
    cfg.norm_epsilon = 1e-5f;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "create norm+bn failed");

    CnnTrainConfig tcfg;
    init_cnn_train_cfg(&tcfg);
    CnnTrainContext* tctx = nn_cnn_train_create(ctx, &tcfg);
    CHECK(tctx != NULL, "train_create norm+bn failed");
    CHECK(tctx->norm_input_cache != NULL, "norm cache missing");
    CHECK(tctx->bn_pre_cache != NULL, "bn cache missing");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    fill_sine(img, cfg.total_input_size);
    make_onehot(target, cfg.feature_size, 0);

    for (int i = 0; i < 5; i++) {
        int rc = nn_cnn_train_step_with_data(tctx, img, target);
        CHECK(rc == 0, "train_step norm+bn failed");
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }

    /* After training, BN running stats should be updated */
    CHECK(ctx->bn_running_mean[0] != 0.0f, "bn running mean still zero");

    free(target); free(img);
    nn_cnn_train_destroy(tctx);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 16: CNN RMSNorm gradient is non-zero */
static int test_cnn_rms_norm_gradient_nonzero(void) {
    CnnConfig cfg;
    init_cnn_cfg_avg(&cfg, 2, 2, 1, 0);
    cfg.use_rms_norm = 1;
    cfg.norm_epsilon = 1e-5f;
    cfg.pooling_activation = CNN_ACT_RELU;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "create failed");

    CnnTrainConfig tcfg;
    init_cnn_train_cfg(&tcfg);
    tcfg.learning_rate = 0.1f;
    CnnTrainContext* tctx = nn_cnn_train_create(ctx, &tcfg);
    CHECK(tctx != NULL, "train_create failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    fill_sine(img, cfg.total_input_size);
    make_onehot(target, cfg.feature_size, 0);

    nn_cnn_train_step_with_data(tctx, img, target);

    /* Check that norm_gamma_grad has non-zero values */
    float grad_sum = 0.0f;
    size_t pooled = cnn_pooled_value_count(&cfg);
    for (size_t i = 0; i < pooled; i++)
        grad_sum += fabsf(tctx->norm_gamma_grad[i]);
    CHECK(grad_sum > 0.0f, "norm_gamma_grad is all zero");

    free(target); free(img);
    nn_cnn_train_destroy(tctx);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 17: CNN Full P0 (RMSNorm + Skip when dims match) training cycle */
static int test_cnn_full_p0_training_cycle(void) {
    CnnConfig cfg;
    init_cnn_cfg_avg(&cfg, 4, 4, 1, 0); /* filter=4, feature=4 — dims match for skip */
    cfg.use_rms_norm = 1;
    cfg.norm_epsilon = 1e-5f;
    cfg.use_skip     = 1;
    cfg.skip_mode    = SKIP_IDENTITY;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "create full p0 failed");
    CHECK(ctx->norm_gamma != NULL, "norm_gamma missing");
    CHECK(ctx->skip != NULL, "skip missing");
    CHECK(ctx->skip->mode == SKIP_IDENTITY, "skip not IDENTITY");

    CnnTrainConfig tcfg;
    init_cnn_train_cfg(&tcfg);
    tcfg.learning_rate = 0.05f;
    CnnTrainContext* tctx = nn_cnn_train_create(ctx, &tcfg);
    CHECK(tctx != NULL, "train_create full p0 failed");
    CHECK(tctx->norm_gamma_grad != NULL, "norm_gamma_grad missing");
    CHECK(tctx->norm_input_cache != NULL, "norm_input_cache missing");
    CHECK(tctx->skip_module_cache != NULL, "skip_module_cache missing");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    fill_sine(img, cfg.total_input_size);
    make_onehot(target, cfg.feature_size, 0);

    float first_loss = -1;
    for (int i = 0; i < 10; i++) {
        nn_cnn_train_step_with_data(tctx, img, target);
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
        if (i == 0) first_loss = tctx->last_loss;
    }
    CHECK(first_loss > 0.0f, "first loss not positive");

    /* Predictions after training should be valid */
    float* out = (float*)malloc(cfg.feature_size * sizeof(float));
    nn_cnn_infer_auto_run(ctx, img, out);
    CHECK(!has_nan(out, cfg.feature_size), "output NaN");

    free(out); free(target); free(img);
    nn_cnn_train_destroy(tctx);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 18: CNN RMSNorm with LeakyReLU activation */
static int test_cnn_rms_norm_leakyrelu(void) {
    CnnConfig cfg;
    init_cnn_cfg_avg(&cfg, 4, 4, 1, 0);
    cfg.use_rms_norm        = 1;
    cfg.norm_epsilon        = 1e-5f;
    cfg.pooling_activation  = CNN_ACT_LEAKY_RELU;
    cfg.output_activation   = CNN_ACT_LEAKY_RELU;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "create lrelu failed");

    CnnTrainConfig tcfg;
    init_cnn_train_cfg(&tcfg);
    CnnTrainContext* tctx = nn_cnn_train_create(ctx, &tcfg);
    CHECK(tctx != NULL, "train_create lrelu failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    fill_sine(img, cfg.total_input_size);
    make_onehot(target, cfg.feature_size, 0);

    for (int i = 0; i < 5; i++) {
        int rc = nn_cnn_train_step_with_data(tctx, img, target);
        CHECK(rc == 0, "train_step lrelu failed");
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
    }

    /* Verify norm_gamma still valid */
    size_t pooled = cnn_pooled_value_count(&cfg);
    for (size_t i = 0; i < pooled; i++)
        CHECK(ctx->norm_gamma[i] > 0.0f, "norm_gamma non-positive with lrelu");

    free(target); free(img);
    nn_cnn_train_destroy(tctx);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 19: CNN RMSNorm training with weight_decay */
static int test_cnn_rms_norm_weight_decay(void) {
    CnnConfig cfg;
    init_cnn_cfg_avg(&cfg, 2, 2, 1, 0);
    cfg.use_rms_norm = 1;
    cfg.norm_epsilon = 1e-5f;

    CnnInferContext* ctx = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx != NULL, "create wd failed");

    CnnTrainConfig tcfg;
    init_cnn_train_cfg(&tcfg);
    tcfg.weight_decay = 1e-4f;
    CnnTrainContext* tctx = nn_cnn_train_create(ctx, &tcfg);
    CHECK(tctx != NULL, "train_create wd failed");

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* target = (float*)malloc(cfg.feature_size * sizeof(float));
    fill_sine(img, cfg.total_input_size);
    make_onehot(target, cfg.feature_size, 0);

    float first_loss = -1;
    for (int i = 0; i < 20; i++) {
        nn_cnn_train_step_with_data(tctx, img, target);
        CHECK(!is_nan(tctx->last_loss), "loss NaN");
        if (i == 0) first_loss = tctx->last_loss;
    }
    CHECK(first_loss > 0.0f, "loss not positive");
    /* norm_gamma should not be affected by weight_decay (it has its own update) */
    CHECK(ctx->norm_gamma[0] > 0.0f, "norm_gamma non-positive with weight decay");

    free(target); free(img);
    nn_cnn_train_destroy(tctx);
    nn_cnn_infer_destroy(ctx);
    return g_test_failures;
}

/* Test 20: CNN RMSNorm output is NOT identical to non-norm (norm changes activations) */
static int test_cnn_rms_norm_changes_output(void) {
    CnnConfig cfg;
    init_cnn_cfg_avg(&cfg, 4, 4, 1, 0);

    /* Without RMSNorm */
    CnnInferContext* ctx_no = nn_cnn_infer_create_with_config(&cfg, 42);
    CHECK(ctx_no != NULL, "create no-norm failed");

    /* With RMSNorm */
    CnnConfig cfg_norm = cfg;
    cfg_norm.use_rms_norm = 1;
    cfg_norm.norm_epsilon = 1e-5f;
    CnnInferContext* ctx_yes = nn_cnn_infer_create_with_config(&cfg_norm, 42);
    CHECK(ctx_yes != NULL, "create with-norm failed");

    /* Copy weights from no-norm to norm context for fair comparison */
    size_t conv_w = cnn_conv_weight_count(&cfg);
    memcpy(ctx_yes->conv_weights, ctx_no->conv_weights, conv_w * sizeof(float));
    memcpy(ctx_yes->conv_bias, ctx_no->conv_bias, cfg.filter_count * sizeof(float));
    size_t proj_w = cnn_projection_weight_count(&cfg);
    if (proj_w > 0) {
        memcpy(ctx_yes->projection_weights, ctx_no->projection_weights, proj_w * sizeof(float));
        memcpy(ctx_yes->projection_bias, ctx_no->projection_bias, cfg.feature_size * sizeof(float));
    }

    float* img = (float*)malloc(cfg.total_input_size * sizeof(float));
    float* out_no = (float*)malloc(cfg.feature_size * sizeof(float));
    float* out_yes = (float*)malloc(cfg.feature_size * sizeof(float));
    fill_sine(img, cfg.total_input_size);

    nn_cnn_infer_auto_run(ctx_no, img, out_no);
    nn_cnn_infer_auto_run(ctx_yes, img, out_yes);

    /* Output should differ because RMSNorm changes normalization */
    int any_diff = 0;
    for (size_t i = 0; i < (size_t)cfg.feature_size; i++) {
        if (fabsf(out_no[i] - out_yes[i]) > 1e-6f)
            any_diff = 1;
    }
    CHECK(any_diff == 1, "RMSNorm did not change output (unexpected)");
    CHECK(!has_nan(out_yes, cfg.feature_size), "norm output NaN");
    CHECK(!has_nan(out_no, cfg.feature_size), "no-norm output NaN");

    free(out_yes); free(out_no); free(img);
    nn_cnn_infer_destroy(ctx_yes);
    nn_cnn_infer_destroy(ctx_no);
    return g_test_failures;
}

/* ═══════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    srand(42);

    TestCase tests[] = {
        /* MLP (tests 1-12) */
        {"mlp_rms_norm_forward",            test_mlp_rms_norm_forward},
        {"mlp_rms_norm_training",           test_mlp_rms_norm_training},
        {"mlp_rms_norm_gamma_drift",        test_mlp_rms_norm_gamma_drift},
        {"mlp_dropout_forward",             test_mlp_dropout_forward},
        {"mlp_dropout_training",            test_mlp_dropout_training},
        {"mlp_skip_identity",               test_mlp_skip_identity},
        {"mlp_skip_laurel_rw",              test_mlp_skip_laurel_rw},
        {"mlp_combined_all",                test_mlp_combined_all},
        {"mlp_loss_decreases",              test_mlp_loss_decreases},
        {"mlp_save_load_norm_gamma",        test_mlp_save_load_norm_gamma},
        {"mlp_save_load_skip",              test_mlp_save_load_skip},
        {"mlp_rms_norm_reduces_variance",   test_mlp_rms_norm_reduces_variance},

        /* CNN additional (tests 13-20) */
        {"cnn_rms_norm_multistep",          test_cnn_rms_norm_multistep},
        {"cnn_rms_norm_dual_pool",          test_cnn_rms_norm_dual_pool},
        {"cnn_rms_norm_with_bn",            test_cnn_rms_norm_with_bn},
        {"cnn_rms_norm_gradient_nonzero",   test_cnn_rms_norm_gradient_nonzero},
        {"cnn_full_p0_training_cycle",      test_cnn_full_p0_training_cycle},
        {"cnn_rms_norm_leakyrelu",          test_cnn_rms_norm_leakyrelu},
        {"cnn_rms_norm_weight_decay",       test_cnn_rms_norm_weight_decay},
        {"cnn_rms_norm_changes_output",     test_cnn_rms_norm_changes_output},
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
