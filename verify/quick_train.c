/**
 * @brief Quick trainer for demo. Trains on a subset of CIFAR-10 to produce weights.bin.
 *
 * Uses the generated MobileNetV2 network code.
 * Trains on 200 samples for 5 epochs, saves weights, runs inference demo.
 */
#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "cifar10_dataset.h"

#include "../demo/demo_runtime_paths.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#define QUICK_TRAIN_SAMPLES  1000
#define QUICK_TEST_SAMPLES   500
#define QUICK_EPOCHS         6
#define QUICK_CLASS_COUNT    10
#define QUICK_IMAGE_SIZE     (32*32*3)

static const char* kQuickTrainBatch = "../../demo/edge_video_preprocess/dataset/data_batch_1.bin";
static const char* kQuickTestBatch  = "../../demo/edge_video_preprocess/dataset/test_batch.bin";
static const char* kWeightsOut      = "../demo/edge_video_preprocess/data/weights.bin";

static float eval_accuracy(void* infer_ctx, Cifar10Dataset* ds) {
    size_t i, correct = 0;
    float output[QUICK_CLASS_COUNT];
    for (i = 0; i < ds->sample_count; i++) {
        infer_auto_run(infer_ctx,
            &ds->images[i * QUICK_IMAGE_SIZE], output);
        if ((uint8_t)cifar10_argmax(output, QUICK_CLASS_COUNT) == ds->labels[i])
            correct++;
    }
    return (float)correct * 100.0f / (float)ds->sample_count;
}

int main(void) {
    Cifar10Dataset train_ds, test_ds;
    void* infer_ctx, *train_ctx;
    char err_buf[256];
    float target[QUICK_CLASS_COUNT];
    int epoch, sample;
    float loss_sum, avg_loss;
    float epoch_losses[QUICK_EPOCHS];
    time_t t0, t1;

    (void)memset(&train_ds, 0, sizeof(train_ds));
    (void)memset(&test_ds, 0, sizeof(test_ds));
    err_buf[0] = '\0';

    /* Set working directory so relative paths resolve correctly */
    demo_set_working_directory_to_executable();

    printf("=== Quick Trainer (CIFAR-10 subset) ===\n\n");

    /* Load datasets */
    if (cifar10_dataset_load_batch(kQuickTrainBatch, QUICK_TRAIN_SAMPLES,
            &train_ds, err_buf, sizeof(err_buf)) != 0) {
        fprintf(stderr, "FAIL: load train: %s\n", err_buf);
        return 1;
    }
    printf("Train: %zu samples loaded\n", train_ds.sample_count);

    if (cifar10_dataset_load_batch(kQuickTestBatch, QUICK_TEST_SAMPLES,
            &test_ds, err_buf, sizeof(err_buf)) != 0) {
        fprintf(stderr, "FAIL: load test: %s\n", err_buf);
        cifar10_dataset_free(&train_ds);
        return 1;
    }
    printf("Test:  %zu samples loaded\n\n", test_ds.sample_count);

    /* Create contexts */
    infer_ctx = infer_create();
    train_ctx = train_create(infer_ctx);
    if (!infer_ctx || !train_ctx) {
        fprintf(stderr, "FAIL: context creation\n");
        cifar10_dataset_free(&train_ds);
        cifar10_dataset_free(&test_ds);
        return 1;
    }
    printf("Network: MobileNetV2 (17 CNN + 1 MLP), depthwise stride-2, He init\n");
    printf("Training: %d epochs x %zu samples\n\n", QUICK_EPOCHS, train_ds.sample_count);

    /* Check initial random accuracy */
    {
        float init_acc = eval_accuracy(infer_ctx, &test_ds);
        printf("Initial accuracy (random): %.1f%% (baseline ~10%%)\n\n", init_acc);
    }

    printf("%-6s %-10s %-10s %-12s\n", "Epoch", "Avg Loss", "Accuracy", "Elapsed");
    printf("%-6s %-10s %-10s %-12s\n", "-----", "--------", "--------", "--------");

    t0 = time(NULL);

    /* Training loop */
    for (epoch = 0; epoch < QUICK_EPOCHS; epoch++) {
        loss_sum = 0.0f;

        for (sample = 0; sample < (int)train_ds.sample_count; sample++) {
            const float* img = &train_ds.images[sample * QUICK_IMAGE_SIZE];
            cifar10_make_one_hot(train_ds.labels[sample], target, QUICK_CLASS_COUNT);

            if (train_step(train_ctx, img, target) != 0) {
                fprintf(stderr, "FAIL: step at epoch %d sample %d\n", epoch + 1, sample + 1);
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                cifar10_dataset_free(&train_ds);
                cifar10_dataset_free(&test_ds);
                return 1;
            }
            loss_sum += train_get_loss(train_ctx);

            /* Progress indicator */
            if ((sample + 1) % 50 == 0) {
                printf("  epoch %d/%d  sample %d/%zu  current loss: %.4f\r",
                    epoch + 1, QUICK_EPOCHS, sample + 1, train_ds.sample_count,
                    train_get_loss(train_ctx));
                (void)fflush(stdout);
            }
        }

        avg_loss = loss_sum / (float)train_ds.sample_count;
        epoch_losses[epoch] = avg_loss;
        t1 = time(NULL);

        /* Evaluate */
        {
            float acc = eval_accuracy(infer_ctx, &test_ds);
            printf("  epoch %d/%d  sample %zu/%zu                          \n",
                epoch + 1, QUICK_EPOCHS, train_ds.sample_count, train_ds.sample_count);
            printf("%-6d %-10.4f %-10.1f%% %-12lld\n",
                epoch + 1, avg_loss, acc, (long long)(t1 - t0));
        }
    }

    printf("\n=== Results ===\n");
    printf("  Epoch losses: [%.4f", epoch_losses[0]);
    {
        int e;
        for (e = 1; e < QUICK_EPOCHS; e++) {
            printf(", %.4f", epoch_losses[e]);
        }
    }
    printf("]\n");

    /* Save weights */
    printf("\nSaving weights to: %s\n", kWeightsOut);
    if (weights_save_to_file(infer_ctx, kWeightsOut) != 0) {
        fprintf(stderr, "FAIL: weight save\n");
    } else {
        printf("Weights saved successfully.\n");
    }

    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    cifar10_dataset_free(&train_ds);
    cifar10_dataset_free(&test_ds);
    return 0;
}
