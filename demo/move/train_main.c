/**
 * @file train_main.c
 * @brief Move Demo training entry
 *
 * Demonstrates usage of profiler-generated training API.
 * This demo ONLY includes the generated train.h header,
 * NOT any src/nn implementation headers.
 *
 * Trains MLP to learn move commands:
 * - Input: (x, y, one-hot command[5])
 * - Output: (new_x, new_y)
 *
 * Network structure: 7 -> [32, 16] -> 2
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "../demo_runtime_paths.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MOVE_MIN_COORD 0
#define MOVE_MAX_COORD 10
#define MOVE_COMMAND_COUNT 5
#define MOVE_EPOCHS 180

/**
 * @brief Normalize discrete grid coordinate to [-1, 1]
 */
static float normalize_coordinate(int value) {
    return ((float)value - 5.0f) / 5.0f;
}

/**
 * @brief Encode discrete command into a single centered scalar feature
 */
static void build_input(float* input, int x, int y, int cmd) {
    int command_index;

    input[0] = normalize_coordinate(x);
    input[1] = normalize_coordinate(y);
    for (command_index = 0; command_index < MOVE_COMMAND_COUNT; ++command_index) {
        input[2 + command_index] = (command_index == cmd) ? 1.0f : 0.0f;
    }
}

/**
 * @brief Apply one move command on the bounded 2D grid
 */
static void apply_command(int x, int y, int cmd, int* out_x, int* out_y) {
    int next_x = x;
    int next_y = y;

    switch (cmd) {
        case 0:
            if (next_y < MOVE_MAX_COORD) {
                next_y += 1;
            }
            break;
        case 1:
            if (next_y > MOVE_MIN_COORD) {
                next_y -= 1;
            }
            break;
        case 2:
            if (next_x > MOVE_MIN_COORD) {
                next_x -= 1;
            }
            break;
        case 3:
            if (next_x < MOVE_MAX_COORD) {
                next_x += 1;
            }
            break;
        case 4:
        default:
            break;
    }

    *out_x = next_x;
    *out_y = next_y;
}

/**
 * @brief Build normalized absolute-position target for one sample
 */
static void build_expected(float* expected, int x, int y, int cmd) {
    int next_x;
    int next_y;

    apply_command(x, y, cmd, &next_x, &next_y);
    expected[0] = normalize_coordinate(next_x);
    expected[1] = normalize_coordinate(next_y);
}

/**
 * @brief Compute simple mean squared error for reporting
 */
static float compute_loss(const float* output, const float* expected, size_t size) {
    float loss = 0.0f;
    size_t i;

    for (i = 0; i < size; ++i) {
        float diff = output[i] - expected[i];
        loss += diff * diff;
    }

    return loss / (float)size;
}

int main(void) {
    const char* output_file = "../../data/weights.bin";
    void* infer_ctx;
    void* train_ctx;
    int save_rc;
    int epoch;
    float input[7];
    float expected[2];
    float output[2];
    float epoch_loss;
    size_t sample_count;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Move Network Training ===\n");
    printf("Network structure: 7 -> [32, 16] -> 2\n");
    printf("Dataset: exhaustive 11x11 grid with 5 commands\n\n");

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

    printf("Training for %d epochs...\n\n", MOVE_EPOCHS);

    for (epoch = 0; epoch < MOVE_EPOCHS; ++epoch) {
        int cmd;
        int y;
        int x;

        epoch_loss = 0.0f;
        sample_count = 0U;

        for (cmd = 0; cmd < MOVE_COMMAND_COUNT; ++cmd) {
            for (y = MOVE_MIN_COORD; y <= MOVE_MAX_COORD; ++y) {
                for (x = MOVE_MIN_COORD; x <= MOVE_MAX_COORD; ++x) {
                    build_input(input, x, y, cmd);
                    build_expected(expected, x, y, cmd);

                    if (train_step(train_ctx, input, expected) != 0) {
                        fprintf(stderr, "Training step failed at epoch %d\n", epoch + 1);
                        train_destroy(train_ctx);
                        infer_destroy(infer_ctx);
                        return 1;
                    }

                    if (infer_auto_run(infer_ctx, input, output) != 0) {
                        fprintf(stderr, "Inference check failed at epoch %d\n", epoch + 1);
                        train_destroy(train_ctx);
                        infer_destroy(infer_ctx);
                        return 1;
                    }

                    epoch_loss += compute_loss(output, expected, 2U);
                    sample_count += 1U;
                }
            }
        }

        if (((epoch + 1) % 20) == 0 || epoch == 0 || epoch == (MOVE_EPOCHS - 1)) {
            printf(
                "Epoch %d/%d - dataset loss: %.4f - trainer loss: %.4f\n",
                epoch + 1,
                MOVE_EPOCHS,
                epoch_loss / (float)sample_count,
                train_get_loss(train_ctx)
            );
        }
    }

    printf("\n=== Training Complete ===\n");
    printf("Average loss: %.4f\n", train_get_loss(train_ctx));
    save_rc = weights_save_to_file(infer_ctx, output_file);
    if (save_rc != 0) {
        fprintf(stderr, "Failed to save weights to %s (rc=%d)\n", output_file, save_rc);
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        return 1;
    }
    printf("Weights saved to: %s\n", output_file);
    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    return 0;
}
