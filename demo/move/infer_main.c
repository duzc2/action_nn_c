/**
 * @file infer_main.c
 * @brief Move Demo inference entry
 *
 * Demonstrates usage of profiler-generated inference API.
 * This demo ONLY includes the generated infer.h header,
 * NOT any src/nn implementation headers.
 *
 * User input:
 * - Start position (x, y)
 * - Commands: 0=up, 1=down, 2=left, 3=right, 4=stop
 *
 * Output: position after each command
 */

#include "infer.h"
#include "weights_load.h"
#include "../demo_runtime_paths.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MOVE_MIN_COORD 0
#define MOVE_MAX_COORD 10
#define MOVE_COMMAND_COUNT 5

/**
 * @brief Normalize discrete grid coordinate to [-1, 1]
 */
static float normalize_coordinate(int value) {
    return ((float)value - 5.0f) / 5.0f;
}

static void build_input(float* input, int x, int y, int cmd) {
    int command_index;

    input[0] = normalize_coordinate(x);
    input[1] = normalize_coordinate(y);
    for (command_index = 0; command_index < MOVE_COMMAND_COUNT; ++command_index) {
        input[2 + command_index] = (command_index == cmd) ? 1.0f : 0.0f;
    }
}

/**
 * @brief Denormalize network prediction back to bounded grid coordinates
 */
static void denormalize_output(float x, float y, int* out_x, int* out_y) {
    *out_x = (int)lroundf(x * 5.0f + 5.0f);
    *out_y = (int)lroundf(y * 5.0f + 5.0f);

    if (*out_x < MOVE_MIN_COORD) {
        *out_x = MOVE_MIN_COORD;
    }
    if (*out_x > MOVE_MAX_COORD) {
        *out_x = MOVE_MAX_COORD;
    }
    if (*out_y < MOVE_MIN_COORD) {
        *out_y = MOVE_MIN_COORD;
    }
    if (*out_y > MOVE_MAX_COORD) {
        *out_y = MOVE_MAX_COORD;
    }
}

int main(void) {
    int x = 0;
    int y = 0;
    int cmd = 0;
    void* infer_ctx;
    const char* weights_file = "../../data/weights.bin";
    float input[7];
    float output[2];
    int out_x, out_y;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Move Network Inference ===\n");
    printf("Input format: startX startY, then commands\n");
    printf("Commands: 0=up, 1=down, 2=left, 3=right, 4=stop\n\n");

    printf("Enter start position (x y): ");
    if (scanf("%d %d", &x, &y) != 2) {
        return 1;
    }

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        return 1;
    }

    if (weights_load_from_file(infer_ctx, weights_file) != 0) {
        fprintf(stderr, "Failed to load weights from %s\n", weights_file);
        infer_destroy(infer_ctx);
        return 1;
    }

    printf("Enter commands (0-4):\n");

    while (scanf("%d", &cmd) == 1) {
        if (cmd == 4) {
            break;
        }
        if (cmd < 0 || cmd >= MOVE_COMMAND_COUNT) {
            printf("Invalid command: %d\n", cmd);
            continue;
        }

        build_input(input, x, y, cmd);
        if (infer_auto_run(infer_ctx, input, output) != 0) {
            fprintf(stderr, "Inference failed\n");
            infer_destroy(infer_ctx);
            return 1;
        }
        denormalize_output(output[0], output[1], &out_x, &out_y);
        if (out_x > x + 1) {
            out_x = x + 1;
        }
        if (out_x < x - 1) {
            out_x = x - 1;
        }
        if (out_y > y + 1) {
            out_y = y + 1;
        }
        if (out_y < y - 1) {
            out_y = y - 1;
        }

        x = out_x;
        y = out_y;
        printf("x=%d y=%d\n", x, y);
    }

    printf("\nfinal_x=%d final_y=%d\n", x, y);

    infer_destroy(infer_ctx);

    return 0;
}
