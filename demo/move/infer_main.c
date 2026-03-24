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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

int main(void) {
    int x = 0;
    int y = 0;
    int cmd = 0;
    void* infer_ctx;
    const char* weights_file = "../../data/weights.bin";
    float input[3];
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

        normalize_input(input, x, y, cmd);
        infer_auto_run(infer_ctx, input, output);
        denormalize_output(output[0], output[1], &out_x, &out_y);

        x = out_x;
        y = out_y;
        printf("x=%d y=%d\n", x, y);
    }

    printf("\nFinal position: x=%d y=%d\n", x, y);

    infer_destroy(infer_ctx);

    return 0;
}
