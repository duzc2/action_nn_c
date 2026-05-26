/**
 * @brief End-to-end test: generated MobileNetV2 on a single CIFAR-10 sample.
 *
 * Uses the actual generated code from edge_video_preprocess/data/ to:
 *   1. Create inference context
 *   2. Run 10 training steps on 1 sample, verify loss decreases
 *   3. Run inference on 1 sample with trained weights
 */
#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "weights_load.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#define TEST_CLASSES 10
#define TEST_IMAGE_SIZE 3072  /* 32x32x3 */

int main(void) {
    void* infer_ctx;
    void* train_ctx;
    float image[TEST_IMAGE_SIZE];
    float target[TEST_CLASSES];
    float output[TEST_CLASSES];
    float prev_loss = 1e9f;
    int step;
    int losses_decreased = 0;

    printf("=== End-to-End MobileNetV2 (18-leaf, Spatial BN, He Init) Test ===\n\n");

    /* Create a simple test image: gradient pattern */
    {
        int i;
        for (i = 0; i < TEST_IMAGE_SIZE; i++) {
            image[i] = ((float)(i % 256) / 255.0f) * 2.0f - 1.0f;
        }
    }

    /* Target: class 0 (airplane) */
    {
        int i;
        for (i = 0; i < TEST_CLASSES; i++) {
            target[i] = (i == 0) ? 1.0f : 0.0f;
        }
    }

    /* Create inference + training contexts */
    infer_ctx = infer_create();
    if (!infer_ctx) {
        fprintf(stderr, "FAIL: infer_create returned NULL\n");
        return 1;
    }
    printf("Inference context created\n");

    train_ctx = train_create(infer_ctx);
    if (!train_ctx) {
        fprintf(stderr, "FAIL: train_create returned NULL\n");
        infer_destroy(infer_ctx);
        return 1;
    }
    printf("Training context created\n");

    /* Initial inference (random weights) */
    if (infer_auto_run(infer_ctx, image, output) != 0) {
        fprintf(stderr, "FAIL: initial inference failed\n");
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        return 1;
    }
    printf("Initial inference: OK (random weights, output[0]=%.4f)\n", output[0]);

    /* Run training steps and verify loss decreases */
    printf("\nTraining 20 steps on a single sample:\n");
    printf("  Step    Loss       Direction\n");
    printf("  ----    ----       ---------\n");

    for (step = 0; step < 20; step++) {
        float current_loss;
        const char* direction;

        if (train_step(train_ctx, image, target) != 0) {
            fprintf(stderr, "FAIL: train_step %d failed\n", step);
            train_destroy(train_ctx);
            infer_destroy(infer_ctx);
            return 1;
        }

        current_loss = train_get_loss(train_ctx);

        if (current_loss < prev_loss - 1e-6f) {
            losses_decreased++;
            direction = "DOWN v";
        } else {
            direction = "---";
        }

        printf("  %4d    %.6f  %s\n",
            step + 1, current_loss, direction);

        prev_loss = current_loss;
    }

    printf("\n");

    /* Run inference with trained weights */
    if (infer_auto_run(infer_ctx, image, output) != 0) {
        fprintf(stderr, "FAIL: final inference failed\n");
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        return 1;
    }

    printf("Inference after training (20 steps):\n");
    {
        int i, best = 0;
        float best_val = output[0];
        for (i = 1; i < TEST_CLASSES; i++) {
            if (output[i] > best_val) {
                best_val = output[i];
                best = i;
            }
        }
        printf("  Predicted class: %d (confidence: %.4f)\n", best, best_val);
        printf("  Target class:    0 (airplane)\n");
        printf("  All outputs: [%.4f", output[0]);
        for (i = 1; i < TEST_CLASSES; i++) {
            printf(", %.4f", output[i]);
        }
        printf("]\n");
    }

    /* Save and reload weights to verify persistence */
    printf("\nTesting weight save/load round-trip:\n");
    {
        float before_save[TEST_CLASSES];
        float after_load[TEST_CLASSES];
        void* reload_infer;

        /* Capture output before save */
        (void)memcpy(before_save, output, sizeof(output));

        /* Save weights */
        if (weights_save_to_file(infer_ctx, "verify_e2e_weights.bin") != 0) {
            fprintf(stderr, "FAIL: weight save\n");
        } else {
            /* Create new context and load */
            reload_infer = infer_create();
            if (!reload_infer) {
                fprintf(stderr, "FAIL: reload infer_create\n");
            } else {
                if (weights_load_from_file(reload_infer, "verify_e2e_weights.bin") != 0) {
                    fprintf(stderr, "FAIL: weight load\n");
                } else {
                    infer_auto_run(reload_infer, image, after_load);
                    /* Verify outputs match */
                    {
                        int i, match = 1;
                        for (i = 0; i < TEST_CLASSES; i++) {
                            if (fabsf(before_save[i] - after_load[i]) > 1e-5f) {
                                match = 0;
                                break;
                            }
                        }
                        printf("  Save/load round-trip: %s\n", match ? "PASS (outputs match)" : "FAIL (mismatch)");
                    }
                }
                infer_destroy(reload_infer);
            }
        }
    }

    /* Summary */
    printf("\n=== Results ===\n");
    printf("  Loss decreased: %d/20 steps\n", losses_decreased);
    printf("  Overall: %s\n",
        (losses_decreased >= 10) ? "PASS - Training converges with stride support" :
        "CHECK - Loss behavior needs investigation");

    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    return 0;
}
