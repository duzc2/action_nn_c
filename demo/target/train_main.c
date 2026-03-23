/**
 * @file train_main.c
 * @brief Target demo training entry based on generated APIs
 */

#include "infer.h"
#include "train.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief Normalize target demo input
 */
static void normalize_input(
    float* input,
    float current_x,
    float current_y,
    float target_x,
    float target_y
) {
    input[0] = current_x / 50.0f;
    input[1] = current_y / 50.0f;
    input[2] = target_x / 50.0f;
    input[3] = target_y / 50.0f;
}

/**
 * @brief Compute normalized expected move vector
 */
static void compute_expected_output(
    float* output,
    float current_x,
    float current_y,
    float target_x,
    float target_y,
    float max_speed
) {
    float dx = target_x - current_x;
    float dy = target_y - current_y;
    float dist = sqrtf(dx * dx + dy * dy);
    float move_scale;

    if (dist <= 1e-6f || max_speed <= 0.0f) {
        output[0] = 0.0f;
        output[1] = 0.0f;
        return;
    }

    move_scale = (dist < max_speed) ? dist : max_speed;
    output[0] = (dx / dist) * (move_scale / max_speed);
    output[1] = (dy / dist) * (move_scale / max_speed);
}

int main(void) {
    const char* output_file = "../../data/weights.bin";
    const float max_speed = 5.0f;
    void* infer_ctx;
    void* train_ctx;
    int epoch;
    int sample;
    int total_samples = 20;
    int total_epochs = 10;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Target Network Training ===\n");
    printf("Network structure: 4 -> [16, 8] -> 2\n");
    printf("Training samples per epoch: %d\n\n", total_samples);

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "failed to create inference context\n");
        return 1;
    }

    train_ctx = train_create(infer_ctx);
    if (train_ctx == NULL) {
        fprintf(stderr, "failed to create training context\n");
        infer_destroy(infer_ctx);
        return 1;
    }

    for (epoch = 0; epoch < total_epochs; epoch++) {
        if ((epoch % 2) == 0) {
            printf("Epoch %d/%d\n", epoch + 1, total_epochs);
        }

        for (sample = 0; sample < total_samples; sample++) {
            float current_x = (float)((rand() % 101) - 50);
            float current_y = (float)((rand() % 101) - 50);
            float target_x = (float)((rand() % 101) - 50);
            float target_y = (float)((rand() % 101) - 50);
            float input[4];
            float expected[2];
            float output[2];

            normalize_input(input, current_x, current_y, target_x, target_y);
            compute_expected_output(
                expected,
                current_x,
                current_y,
                target_x,
                target_y,
                max_speed
            );

            train_step(train_ctx, input, expected);
            infer_auto_run(infer_ctx, input, output);

            if ((sample % 10) == 0) {
                printf(
                    "  sample %d: current(%.1f, %.1f) target(%.1f, %.1f) -> "
                    "predicted(%.3f, %.3f) expected(%.3f, %.3f)\n",
                    sample,
                    current_x,
                    current_y,
                    target_x,
                    target_y,
                    output[0],
                    output[1],
                    expected[0],
                    expected[1]
                );
            }
        }

        printf("  average loss: %.4f\n", train_get_loss(train_ctx));
    }

    printf("\nTraining completed.\n");
    printf("Average loss: %.4f\n", train_get_loss(train_ctx));
    printf("Weights should be saved to: %s\n", output_file);

    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    return 0;
}
