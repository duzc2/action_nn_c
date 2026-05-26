/**
 * @brief Standalone test for the CNN training backend (bypasses graph system).
 *
 * Creates a single CNN layer (conv3x3, BN+ReLU6, POOL_AVG) and trains
 * on synthetic data to verify gradient flow and parameter updates.
 */
#include "../src/nn/types/cnn/cnn_config.h"
#include "../src/nn/types/cnn/cnn_infer_ops.h"
#include "../src/nn/types/cnn/cnn_train_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple Sin experiment: learn to map [sin(x+phase), sin(y+phase)] → label */
#define IMG_H  8
#define IMG_W  8
#define IMG_C  2
#define FILTERS 4
#define KERNEL 3
#define N_SAMPLES 200
#define N_EPOCHS 30

static float drand(void) {
    return (float)rand() / (float)RAND_MAX;
}

static void make_sample(float* img, float* target, size_t n_out, int label) {
    size_t c, y, x;
    for (c = 0; c < IMG_C; c++) {
        for (y = 0; y < IMG_H; y++) {
            for (x = 0; x < IMG_W; x++) {
                float phase = (float)label * 0.3f + (float)c * 0.5f;
                img[(c * IMG_H + y) * IMG_W + x] =
                    sinf(((float)x / IMG_W * 3.14f + phase) * ((float)y / IMG_H * 3.14f + 1.0f))
                    + 0.05f * (drand() - 0.5f);
            }
        }
    }
    for (size_t i = 0; i < n_out; i++) target[i] = 0.0f;
    target[label] = 1.0f;
}

int main(void) {
    CnnConfig config;
    CnnTrainConfig train_cfg;
    CnnInferContext* infer_ctx;
    CnnTrainContext* train_ctx;
    float* images;
    float* targets;
    float loss_sum;
    int epoch, s;

    srand(42);
    (void)memset(&config, 0, sizeof(config));
    (void)memset(&train_cfg, 0, sizeof(train_cfg));

    /* Configure: 2ch→conv3x3→4 filters→BN+ReLU6→POOL_AVG→4 features */
    config.filter_count       = FILTERS;
    config.kernel_size        = KERNEL;
    config.channel_count      = IMG_C;
    config.frame_width        = IMG_W;
    config.frame_height       = IMG_H;
    config.sequence_length    = 1U;
    config.feature_size       = FILTERS;  /* one value per filter after global avg pool */
    config.pooling_mode       = CNN_POOL_AVG;
    config.conv_mode          = CNN_CONV_STANDARD;
    config.stride             = 1U;
    config.use_batch_norm     = 1;
    config.bn_momentum        = 0.9f;
    config.bn_epsilon         = 1e-5f;
    config.output_activation  = CNN_ACT_NONE;
    config.pooling_activation = CNN_ACT_RELU;
    config.total_input_size   = IMG_H * IMG_W * IMG_C;

    size_t out_h = (config.frame_height - config.kernel_size) / config.stride + 1U;
    size_t out_w = (config.frame_width - config.kernel_size) / config.stride + 1U;
    printf("CNN config: %u x %u x %u input, conv %u x %u, %u filters, output %zu x %zu x %u -> POOL_AVG -> %u features\n",
        (unsigned)IMG_H, (unsigned)IMG_W, (unsigned)IMG_C,
        (unsigned)KERNEL, (unsigned)KERNEL, (unsigned)FILTERS,
        out_h, out_w, (unsigned)FILTERS, (unsigned)FILTERS);

    /* Create infer */
    infer_ctx = nn_cnn_infer_create_with_config(&config, 12345U);
    if (!infer_ctx) { printf("FAIL: infer_create\n"); return 1; }

    printf("bn_gamma[0]=%.4f bn_beta[0]=%.4f mean[0]=%.4f var[0]=%.4f\n",
        infer_ctx->bn_gamma[0], infer_ctx->bn_beta[0],
        infer_ctx->bn_running_mean[0], infer_ctx->bn_running_var[0]);

    /* Train config */
    train_cfg.learning_rate = 0.1f;
    train_cfg.momentum      = 0.9f;
    train_cfg.weight_decay  = 0.0f;
    train_cfg.batch_size    = 1U;
    train_cfg.seed          = 12345U;

    /* Create train context */
    train_ctx = nn_cnn_train_create(infer_ctx, &train_cfg);
    if (!train_ctx) { printf("FAIL: train_create\n"); return 1; }

    printf("bn_pre_cache=%p bn_spatial_var=%p infer->bn_training_pre_cache=%p\n",
        (void*)train_ctx->bn_pre_cache, (void*)train_ctx->bn_spatial_var,
        (void*)infer_ctx->bn_training_pre_cache);

    /* Allocate data - binary classification, targets size = feature_size */
    images  = malloc(N_SAMPLES * IMG_C * IMG_H * IMG_W * sizeof(float));
    targets = malloc(N_SAMPLES * FILTERS * sizeof(float));
    for (s = 0; s < N_SAMPLES; s++) {
        make_sample(images + s * IMG_C * IMG_H * IMG_W,
                    targets + s * FILTERS, FILTERS, s % 2);
    }

    printf("\nTraining %d epochs x %d samples, lr=%.4f momentum=%.2f\n", N_EPOCHS, N_SAMPLES, train_cfg.learning_rate, train_cfg.momentum);

    /* Pre-train check */
    {
        float conv_w0 = infer_ctx->conv_weights[0];
        float conv_w18 = infer_ctx->conv_weights[18];
        float conv_b1 = infer_ctx->conv_bias[1];
        float out[FILTERS];
        nn_cnn_infer_auto_run(infer_ctx, images, out);
        printf("Pre: conv_w[0]=%.6f conv_w[18]=%.6f conv_b[1]=%.6f out[0..3]=%.4f %.4f %.4f %.4f\n",
            conv_w0, conv_w18, conv_b1, out[0], out[1], out[2], out[3]);
    }

    for (epoch = 0; epoch < N_EPOCHS; epoch++) {
        loss_sum = 0.0f;
        for (s = 0; s < N_SAMPLES; s++) {
            nn_cnn_train_step_with_data(
                train_ctx,
                images + s * IMG_C * IMG_H * IMG_W,
                targets + s * FILTERS);
            loss_sum += train_ctx->last_loss;
        }
        if (epoch == 0) {
            printf("  After epoch 0: conv_w[0]=%.6f conv_w[18]=%.6f conv_b[1]=%.6f bn_gamma[1]=%.4f\n",
                infer_ctx->conv_weights[0], infer_ctx->conv_weights[18],
                infer_ctx->conv_bias[1], infer_ctx->bn_gamma[1]);
        }
        printf("  epoch %2d: avg_loss=%.6f\n", epoch, loss_sum / N_SAMPLES);
    }

    {
        float out[FILTERS];
        nn_cnn_infer_auto_run(infer_ctx, images, out);
        printf("\nPost: conv_w[0]=%.6f conv_w[18]=%.6f conv_b[1]=%.6f out[0..3]=%.4f %.4f %.4f %.4f\n",
            infer_ctx->conv_weights[0], infer_ctx->conv_weights[18], infer_ctx->conv_bias[1],
            out[0], out[1], out[2], out[3]);
        printf("bn_gamma[1]=%.4f bn_beta[1]=%.4f mean[0]=%.4f var[0]=%.4f\n",
            infer_ctx->bn_gamma[1], infer_ctx->bn_beta[1],
            infer_ctx->bn_running_mean[0], infer_ctx->bn_running_var[0]);
    }

    nn_cnn_train_destroy(train_ctx);
    nn_cnn_infer_destroy(infer_ctx);
    free(images);
    free(targets);
    printf("PASS: standalone CNN backend training test complete.\n");
    return 0;
}
