/**
 * @file train_main.c
 * @brief Target demo training entry based on generated APIs
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "../demo_runtime_paths.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TARGET_EPOCHS 80
#define TARGET_GRID_SIZE 5
#define TARGET_SPEED_COUNT 3

/**
 * @brief Normalize target demo input using documented field order
 */
static void build_input(
    float* input,
    float target_x,
    float target_y,
    float current_x,
    float current_y,
    float max_speed
) {
    input[0] = target_x / 50.0f;
    input[1] = target_y / 50.0f;
    input[2] = current_x / 50.0f;
    input[3] = current_y / 50.0f;
    input[4] = max_speed / 5.0f;
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
    static const float position_grid[TARGET_GRID_SIZE] = {
        -50.0f, -25.0f, 0.0f, 25.0f, 50.0f
    };
    static const float speed_grid[TARGET_SPEED_COUNT] = {
        1.0f, 3.0f, 5.0f
    };
    void* infer_ctx;
    void* train_ctx;
    int save_rc;
    int epoch;
    float output[2];
    float epoch_loss;
    size_t sample_count;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Target Network Training ===\n");
    printf("Network structure: 5 -> [32, 16] -> 2\n");
    printf("Dataset: structured grid positions with variable max speed\n\n");

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

    for (epoch = 0; epoch < TARGET_EPOCHS; ++epoch) {
        size_t speed_index;
        size_t current_y_index;
        size_t current_x_index;
        size_t target_y_index;
        size_t target_x_index;

        epoch_loss = 0.0f;
        sample_count = 0U;

        for (speed_index = 0U; speed_index < TARGET_SPEED_COUNT; ++speed_index) {
            const float max_speed = speed_grid[speed_index];

            for (current_y_index = 0U; current_y_index < TARGET_GRID_SIZE; ++current_y_index) {
                for (current_x_index = 0U; current_x_index < TARGET_GRID_SIZE; ++current_x_index) {
                    const float current_x = position_grid[current_x_index];
                    const float current_y = position_grid[current_y_index];

                    for (target_y_index = 0U; target_y_index < TARGET_GRID_SIZE; ++target_y_index) {
                        for (target_x_index = 0U; target_x_index < TARGET_GRID_SIZE; ++target_x_index) {
                            const float target_x = position_grid[target_x_index];
                            const float target_y = position_grid[target_y_index];
                            float input[5];
                            float expected[2];
                            float diff_x;
                            float diff_y;

                            build_input(
                                input,
                                target_x,
                                target_y,
                                current_x,
                                current_y,
                                max_speed
                            );
                            compute_expected_output(
                                expected,
                                current_x,
                                current_y,
                                target_x,
                                target_y,
                                max_speed
                            );

                            if (train_step(train_ctx, input, expected) != 0) {
                                fprintf(stderr, "target training step failed at epoch %d\n", epoch + 1);
                                train_destroy(train_ctx);
                                infer_destroy(infer_ctx);
                                return 1;
                            }

                            if (infer_auto_run(infer_ctx, input, output) != 0) {
                                fprintf(stderr, "target inference check failed at epoch %d\n", epoch + 1);
                                train_destroy(train_ctx);
                                infer_destroy(infer_ctx);
                                return 1;
                            }

                            diff_x = output[0] - expected[0];
                            diff_y = output[1] - expected[1];
                            epoch_loss += (diff_x * diff_x + diff_y * diff_y) * 0.5f;
                            sample_count += 1U;
                        }
                    }
                }
            }
        }

        if (((epoch + 1) % 10) == 0 || epoch == 0 || epoch == (TARGET_EPOCHS - 1)) {
            printf(
                "Epoch %d/%d - dataset loss: %.4f - trainer loss: %.4f\n",
                epoch + 1,
                TARGET_EPOCHS,
                epoch_loss / (float)sample_count,
                train_get_loss(train_ctx)
            );
        }
    }

    printf("\nTraining completed.\n");
    printf("Average loss: %.4f\n", train_get_loss(train_ctx));
    save_rc = weights_save_to_file(infer_ctx, output_file);
    if (save_rc != 0) {
        fprintf(stderr, "failed to save weights to %s (rc=%d)\n", output_file, save_rc);
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        return 1;
    }
    printf("Weights saved to: %s\n", output_file);

    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    return 0;
}
