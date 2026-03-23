#include "infer.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>

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

int main(void) {
    const float max_speed = 5.0f;
    void* infer_ctx;
    float current_x = 0.0f;
    float current_y = 0.0f;
    float target_x = 0.0f;
    float target_y = 0.0f;
    float input[4];
    float output[2];

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "failed to switch working directory to executable directory\n");
        return 1;
    }

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "failed to create inference context\n");
        return 1;
    }

    printf("input: targetX targetY currentX currentY\n");
    while (scanf("%f %f %f %f", &target_x, &target_y, &current_x, &current_y) == 4) {
        normalize_input(input, current_x, current_y, target_x, target_y);
        if (infer_auto_run(infer_ctx, input, output) != 0) {
            fprintf(stderr, "inference failed\n");
            infer_destroy(infer_ctx);
            return 1;
        }
        printf(
            "move_dx=%.6f move_dy=%.6f\n",
            output[0] * max_speed,
            output[1] * max_speed
        );
    }

    infer_destroy(infer_ctx);
    return 0;
}
