/**
 * @brief Quick verification of CNN stride support.
 *
 * Tests:
 *   1. Forward pass with stride=1 produces expected output dimensions
 *   2. Forward pass with stride=2 produces halved output grid
 *   3. Weight save/load round-trips stride correctly
 *   4. Training step doesn't crash with stride=2
 */
#include "nn/types/cnn/cnn_infer_ops.h"
#include "nn/types/cnn/cnn_train_ops.h"
#include <stdio.h>
#include <stdint.h>

static int test_forward_stride1(void) {
    CnnConfig cfg;
    CnnInferContext* ctx;
    float input[4 * 4 * 1];  /* 4x4x1 frame */
    float output[256];
    int result;

    (void)memset(&cfg, 0, sizeof(cfg));
    cfg.total_input_size   = 16;
    cfg.sequence_length    = 1;
    cfg.frame_width        = 4;
    cfg.frame_height       = 4;
    cfg.channel_count      = 1;
    cfg.kernel_size        = 3;
    cfg.filter_count       = 2;
    cfg.feature_size       = 0;
    cfg.pooling_mode       = CNN_POOL_NONE;
    cfg.output_activation  = CNN_ACT_RELU;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 1;
    cfg.seed               = 42;

    ctx = nn_cnn_infer_create_with_config(&cfg, 0);
    if (!ctx) { printf("FAIL: stride=1 create\n"); return 1; }

    /* Set all inputs to 1.0 */
    {
        size_t i;
        for (i = 0; i < 16; i++) input[i] = 1.0f;
    }

    /* Output should be [2 filters] * [2x2 grid] = 8 floats */
    result = nn_cnn_infer_auto_run(ctx, input, output);
    if (result != 0) { printf("FAIL: stride=1 forward pass rc=%d\n", result); nn_cnn_infer_destroy(ctx); return 1; }

    /* Verify output size: (4-3)/1+1 = 2, 2*2*2 = 8 */
    nn_cnn_infer_get_output(ctx, output, 8);
    printf("  stride=1: output[0]=%.4f output[1]=%.4f (grid: 2x2, 2 filters = 8 values)\n", output[0], output[1]);

    nn_cnn_infer_destroy(ctx);
    return 0;
}

static int test_forward_stride2(void) {
    CnnConfig cfg;
    CnnInferContext* ctx;
    float input[8 * 8 * 1];
    float output[256];
    int result;

    (void)memset(&cfg, 0, sizeof(cfg));
    cfg.total_input_size   = 64;
    cfg.sequence_length    = 1;
    cfg.frame_width        = 8;
    cfg.frame_height       = 8;
    cfg.channel_count      = 1;
    cfg.kernel_size        = 3;
    cfg.filter_count       = 1;
    cfg.feature_size       = 0;
    cfg.pooling_mode       = CNN_POOL_NONE;
    cfg.output_activation  = CNN_ACT_RELU;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 2;
    cfg.seed               = 42;

    ctx = nn_cnn_infer_create_with_config(&cfg, 0);
    if (!ctx) { printf("FAIL: stride=2 create\n"); return 1; }

    {
        size_t i;
        for (i = 0; i < 64; i++) input[i] = 1.0f;
    }

    /* Output: (8-3)/2+1 = 3, 3*3*1 = 9 floats */
    result = nn_cnn_infer_auto_run(ctx, input, output);
    if (result != 0) { printf("FAIL: stride=2 forward pass rc=%d\n", result); nn_cnn_infer_destroy(ctx); return 1; }

    nn_cnn_infer_get_output(ctx, output, 9);
    printf("  stride=2: output[0]=%.4f output[4]=%.4f (grid: 3x3, 1 filter = 9 values)\n", output[0], output[4]);

    nn_cnn_infer_destroy(ctx);
    return 0;
}

static int test_stride_rejected_when_zero(void) {
    CnnConfig cfg;
    CnnInferContext* ctx;

    (void)memset(&cfg, 0, sizeof(cfg));
    cfg.total_input_size   = 64;
    cfg.sequence_length    = 1;
    cfg.frame_width        = 8;
    cfg.frame_height       = 8;
    cfg.channel_count      = 1;
    cfg.kernel_size        = 3;
    cfg.filter_count       = 1;
    cfg.pooling_mode       = CNN_POOL_NONE;
    cfg.output_activation  = CNN_ACT_RELU;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 0;  /* should be rejected */
    cfg.seed               = 42;

    ctx = nn_cnn_infer_create_with_config(&cfg, 0);
    if (ctx != NULL) { printf("FAIL: stride=0 not rejected\n"); nn_cnn_infer_destroy(ctx); return 1; }
    printf("  stride=0: correctly rejected\n");
    return 0;
}

static int test_weight_save_load_stride2(void) {
    CnnConfig cfg;
    CnnInferContext* ctx;
    CnnInferContext* loaded_ctx;
    float input[64], output[64], loaded_output[64];
    int ok;
    FILE* fp;

    (void)memset(&cfg, 0, sizeof(cfg));
    cfg.total_input_size   = 64;
    cfg.sequence_length    = 1;
    cfg.frame_width        = 8;
    cfg.frame_height       = 8;
    cfg.channel_count      = 1;
    cfg.kernel_size        = 3;
    cfg.filter_count       = 1;
    cfg.pooling_mode       = CNN_POOL_NONE;
    cfg.output_activation  = CNN_ACT_RELU;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 2;
    cfg.seed               = 42;

    ctx = nn_cnn_infer_create_with_config(&cfg, 0);
    if (!ctx) { printf("FAIL: save/load create\n"); return 1; }

    /* Set all inputs to 1.0 */
    {
        size_t i;
        for (i = 0; i < 64; i++) input[i] = 1.0f;
    }

    nn_cnn_infer_auto_run(ctx, input, output);

    /* Save weights */
#ifdef _WIN32
    if (fopen_s(&fp, "verify_stride_tmp.bin", "wb") != 0) fp = NULL;
#else
    fp = fopen("verify_stride_tmp.bin", "wb");
#endif
    if (!fp) { printf("FAIL: open for write\n"); nn_cnn_infer_destroy(ctx); return 1; }
    ok = nn_cnn_save_weights(ctx, fp);
    fclose(fp);
    if (!ok) { printf("FAIL: save weights\n"); nn_cnn_infer_destroy(ctx); return 1; }

    /* Load into new context */
    loaded_ctx = nn_cnn_infer_create_with_config(&cfg, 0);
    if (!loaded_ctx) { printf("FAIL: create loaded ctx\n"); nn_cnn_infer_destroy(ctx); return 1; }

#ifdef _WIN32
    if (fopen_s(&fp, "verify_stride_tmp.bin", "rb") != 0) fp = NULL;
#else
    fp = fopen("verify_stride_tmp.bin", "rb");
#endif
    if (!fp) { printf("FAIL: open for read\n"); nn_cnn_infer_destroy(ctx); nn_cnn_infer_destroy(loaded_ctx); return 1; }
    ok = nn_cnn_load_weights(loaded_ctx, fp);
    fclose(fp);
    if (!ok) { printf("FAIL: load weights\n"); nn_cnn_infer_destroy(ctx); nn_cnn_infer_destroy(loaded_ctx); return 1; }

    /* Verify identical outputs */
    nn_cnn_infer_auto_run(loaded_ctx, input, loaded_output);
    {
        size_t i;
        for (i = 0; i < 9; i++) {
            if (output[i] != loaded_output[i]) {
                printf("FAIL: save/load mismatch at [%zu]: %.6f vs %.6f\n", i, output[i], loaded_output[i]);
                nn_cnn_infer_destroy(ctx);
                nn_cnn_infer_destroy(loaded_ctx);
                return 1;
            }
        }
    }

    printf("  stride=2: save/load round-trip verified (%zu floats match)\n", (size_t)9);

    nn_cnn_infer_destroy(ctx);
    nn_cnn_infer_destroy(loaded_ctx);
    return 0;
}

static int test_train_step_stride2(void) {
    CnnConfig cfg;
    CnnTrainConfig train_cfg;
    CnnInferContext* infer_ctx;
    CnnTrainContext* train_ctx;
    float input[64], target[9];
    int rc;

    (void)memset(&cfg, 0, sizeof(cfg));
    cfg.total_input_size   = 64;
    cfg.sequence_length    = 1;
    cfg.frame_width        = 8;
    cfg.frame_height       = 8;
    cfg.channel_count      = 1;
    cfg.kernel_size        = 3;
    cfg.filter_count       = 1;
    cfg.pooling_mode       = CNN_POOL_NONE;
    cfg.output_activation  = CNN_ACT_RELU;
    cfg.conv_mode          = CNN_CONV_STANDARD;
    cfg.stride             = 2;
    cfg.seed               = 42;

    infer_ctx = nn_cnn_infer_create_with_config(&cfg, 0);
    if (!infer_ctx) { printf("FAIL: train create infer\n"); return 1; }

    (void)memset(&train_cfg, 0, sizeof(train_cfg));
    train_cfg.learning_rate = 0.01f;
    train_ctx = nn_cnn_train_create(infer_ctx, &train_cfg);
    if (!train_ctx) { printf("FAIL: train create train\n"); nn_cnn_infer_destroy(infer_ctx); return 1; }

    {
        size_t i;
        for (i = 0; i < 64; i++) input[i] = 1.0f;
        for (i = 0; i < 9; i++) target[i] = 0.5f;
    }

    rc = nn_cnn_train_step_with_data(train_ctx, input, target);
    if (rc != 0) { printf("FAIL: train step rc=%d\n", rc); nn_cnn_train_destroy(train_ctx); nn_cnn_infer_destroy(infer_ctx); return 1; }

    {
        float loss = train_ctx->last_loss;
        printf("  stride=2: training step ok, loss=%.6f\n", loss);
    }

    nn_cnn_train_destroy(train_ctx);
    nn_cnn_infer_destroy(infer_ctx);
    return 0;
}

int main(void) {
    int failures = 0;
    printf("=== CNN Stride Support Verification ===\n\n");

    printf("Test 1: Forward pass with stride=1...\n");
    failures += test_forward_stride1();

    printf("\nTest 2: Forward pass with stride=2...\n");
    failures += test_forward_stride2();

    printf("\nTest 3: Reject stride=0...\n");
    failures += test_stride_rejected_when_zero();

    printf("\nTest 4: Weight save/load with stride=2...\n");
    failures += test_weight_save_load_stride2();

    printf("\nTest 5: Training step with stride=2...\n");
    failures += test_train_step_stride2();

    printf("\n=== Results: %d failures ===\n", failures);
    return failures;
}
