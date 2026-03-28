#include "infer.h"
#include "weights_load.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <stdlib.h>

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
 * @brief Parse a single input line into five float values
 *
 * Uses C99 strtof-based parsing to avoid MSVC scanf deprecation warnings
 * while keeping the console interface unchanged.
 *
 * @return 1 on success, 0 on parse failure
 */
static int parse_input_line(
    const char* line,
    float* target_x,
    float* target_y,
    float* current_x,
    float* current_y,
    float* max_speed
) {
    const char* cursor = line;
    char* end_ptr = NULL;
    float values[5];
    size_t index;

    for (index = 0; index < 5; ++index) {
        values[index] = strtof(cursor, &end_ptr);
        if (end_ptr == cursor) {
            return 0;
        }
        cursor = end_ptr;
    }

    while (*cursor != '\0') {
        if ((*cursor != ' ') &&
            (*cursor != '\t') &&
            (*cursor != '\r') &&
            (*cursor != '\n')) {
            return 0;
        }
        ++cursor;
    }

    *target_x = values[0];
    *target_y = values[1];
    *current_x = values[2];
    *current_y = values[3];
    *max_speed = values[4];
    return 1;
}

int main(void) {
    const char* weights_file = "../../data/weights.bin";
    void* infer_ctx;
    float current_x = 0.0f;
    float current_y = 0.0f;
    float target_x = 0.0f;
    float target_y = 0.0f;
    float max_speed = 0.0f;
    float input[5];
    float output[2];
    float move_dx;
    float move_dy;
    char line[256];

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "failed to switch working directory to executable directory\n");
        return 1;
    }

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "failed to create inference context\n");
        return 1;
    }

    if (weights_load_from_file(infer_ctx, weights_file) != 0) {
        fprintf(stderr, "failed to load weights from %s\n", weights_file);
        infer_destroy(infer_ctx);
        return 1;
    }

    printf("input: targetX targetY currentX currentY maxSpeed\n");
    while (fgets(line, sizeof(line), stdin) != NULL) {
        if (!parse_input_line(
                line,
                &target_x,
                &target_y,
                &current_x,
                &current_y,
                &max_speed)) {
            printf("invalid input, expected: targetX targetY currentX currentY maxSpeed\n");
            continue;
        }

        if (max_speed <= 0.0f) {
            printf("maxSpeed must be positive\n");
            continue;
        }

        build_input(input, target_x, target_y, current_x, current_y, max_speed);
        if (infer_auto_run(infer_ctx, input, output) != 0) {
            fprintf(stderr, "inference failed\n");
            infer_destroy(infer_ctx);
            return 1;
        }

        move_dx = output[0];
        move_dy = output[1];
        if (move_dx > 1.0f) {
            move_dx = 1.0f;
        }
        if (move_dx < -1.0f) {
            move_dx = -1.0f;
        }
        if (move_dy > 1.0f) {
            move_dy = 1.0f;
        }
        if (move_dy < -1.0f) {
            move_dy = -1.0f;
        }
        printf(
            "move_dx=%.6f move_dy=%.6f\n",
            move_dx * max_speed,
            move_dy * max_speed
        );
    }

    infer_destroy(infer_ctx);
    return 0;
}
