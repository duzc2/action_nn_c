/**
 * @file train_main.c
 * @brief Move Demo training entry
 *
 * Demonstrates usage of profiler-generated training API.
 * This demo ONLY includes the generated train.h header,
 * NOT any src/nn implementation headers.
 *
 * Trains MLP to learn move commands:
 * - Input: (x, y, command)
 * - Output: (new_x, new_y)
 *
 * Network structure: 3 -> [16, 8] -> 2
 */

#include "infer.h"
#include "train.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LEARNING_RATE 0.1f

/**
 * @brief Normalize input to [0, 1] range
 */
static void normalize_input(float* input, int x, int y, int cmd) {
    input[0] = (float)x / 10.0f;
    input[1] = (float)y / 10.0f;
    input[2] = (float)cmd / 3.0f;
}

/**
 * @brief Denormalize output from [0, 1] range
 */
static void denormalize_output(float x, float y, int* out_x, int* out_y) {
    *out_x = (int)roundf(x * 10.0f);
    *out_y = (int)roundf(y * 10.0f);

    if (*out_x < 0) *out_x = 0;
    if (*out_x > 10) *out_x = 10;
    if (*out_y < 0) *out_y = 0;
    if (*out_y > 10) *out_y = 10;
}

/**
 * @brief Compute expected output for given move
 */
static void compute_expected(float* expected, int x, int y, int cmd) {
    float norm_x = (float)x / 10.0f;
    float norm_y = (float)y / 10.0f;
    float norm_dx = 0.0f;
    float norm_dy = 0.0f;

    switch (cmd) {
        case 0: norm_dy = 0.1f; break;
        case 1: norm_dy = -0.1f; break;
        case 2: norm_dx = -0.1f; break;
        case 3: norm_dx = 0.1f; break;
    }

    expected[0] = norm_x + norm_dx;
    expected[1] = norm_y + norm_dy;
}

/**
 * @brief Compute MSE loss
 */
static float compute_loss(const float* output, const float* expected, size_t size) {
    float loss = 0.0f;
    size_t i;
    for (i = 0; i < size; i++) {
        float diff = output[i] - expected[i];
        loss += diff * diff;
    }
    return loss / 2.0f;
}

/**
 * @brief Run one training sample
 */
static void train_sample(void* train_ctx, void* infer_ctx,
                        const float* input, const float* expected) {
    float output[2];

    train_step(train_ctx, input, expected);
    infer_auto_run(infer_ctx, input, output);

    printf("  Input: (%.2f, %.2f, %.2f) -> Output: (%.3f, %.3f), Expected: (%.3f, %.3f), Loss: %.4f\n",
           input[0], input[1], input[2],
           output[0], output[1], expected[0], expected[1],
           compute_loss(output, expected, 2));
}

int main(void) {
    const char* output_file = "../../data/weights.bin";
    void* infer_ctx;
    void* train_ctx;
    int epoch;
    int sample;
    int total_epochs = 20;
    int total_samples = 50;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Move Network Training ===\n");
    printf("Network structure: 3 -> [16, 8] -> 2\n");
    printf("Learning rate: %.2f\n\n", LEARNING_RATE);

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

    printf("Training for %d epochs, %d samples each...\n\n", total_epochs, total_samples);

    for (epoch = 0; epoch < total_epochs; epoch++) {
        printf("Epoch %d/%d:\n", epoch + 1, total_epochs);

        for (sample = 0; sample < total_samples; sample++) {
            int x, y, cmd;
            float input[3];
            float expected[2];

            x = rand() % 10;
            y = rand() % 10;
            cmd = rand() % 4;

            normalize_input(input, x, y, cmd);
            compute_expected(expected, x, y, cmd);

            train_sample(train_ctx, infer_ctx, input, expected);
        }

        if ((epoch + 1) % 5 == 0) {
            printf("  [Checkpoint at epoch %d, Loss: %.4f]\n",
                   epoch + 1, train_get_loss(train_ctx));
        }
    }

    printf("\n=== Training Complete ===\n");
    printf("Average loss: %.4f\n", train_get_loss(train_ctx));
    printf("Weights should be saved to: %s\n", output_file);
    train_destroy(train_ctx);
    infer_destroy(infer_ctx);

    return 0;
}
