/**
 * @file train_main.c
 * @brief SevenSeg Demo training entry
 *
 * Demonstrates usage of profiler-generated training API.
 * This demo ONLY includes the generated train.h header,
 * NOT any src/nn implementation headers.
 */

#include "infer.h"
#include "train.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Seven segment display segment map
 */
static const float SEG_MAP[10][7] = {
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f},
    {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
    {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f},
    {1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}
};

static void encode_digit(int digit, float* input) {
    size_t i;
    for (i = 0; i < 10; i++) {
        input[i] = (i == (size_t)digit) ? 1.0f : 0.0f;
    }
}

int main(void) {
    void* infer_ctx;
    void* train_ctx;
    const char* output_file = "../../data/weights.bin";
    size_t epoch;
    size_t digit;
    int epoch_count = 100;
    float total_loss = 0.0f;
    float input[10];
    float target[7];
    float output[7];

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== SevenSeg Network Training ===\n\n");

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        return 1;
    }

    train_ctx = train_create(infer_ctx);
    if (train_ctx == NULL) {
        fprintf(stderr, "Failed to create training context\n");
        infer_destroy(infer_ctx);
        return 1;
    }

    printf("Training for %d epochs...\n\n", epoch_count);

    for (epoch = 0; epoch < (size_t)epoch_count; epoch++) {
        float epoch_loss = 0.0f;

        for (digit = 0; digit < 10; digit++) {
            encode_digit((int)digit, input);
            memcpy(target, SEG_MAP[digit], sizeof(target));

            train_step(train_ctx, input, target);

            infer_auto_run(infer_ctx, input, output);
        }

        total_loss += train_get_loss(train_ctx);

        if ((epoch + 1) % 20 == 0) {
            printf("Epoch %zu/%d - Loss: %.4f\n",
                   epoch + 1, epoch_count, train_get_loss(train_ctx));
        }
    }

    printf("\nTraining completed! Average loss: %.4f\n", total_loss / (float)epoch_count);
    printf("Weights should be saved to: %s\n", output_file);

    train_destroy(train_ctx);
    infer_destroy(infer_ctx);

    return 0;
}
