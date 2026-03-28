/**
 * @file infer_main.c
 * @brief Hybrid transformer+MLP route demo inference entry
 */

#include "infer.h"
#include "weights_load.h"
#include "../demo_runtime_paths.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define HYBRID_ROUTE_INPUT_SIZE 8U
#define HYBRID_ROUTE_MAP_SIZE 11U
#define HYBRID_ROUTE_MAX_FRAMES 8U
#define HYBRID_ROUTE_STEP_SCALE 0.30f

/**
 * @brief Clamp a raster coordinate into the printable map.
 */
static size_t clamp_map_index(int value) {
    if (value < 0) {
        return 0U;
    }
    if (value >= (int)HYBRID_ROUTE_MAP_SIZE) {
        return HYBRID_ROUTE_MAP_SIZE - 1U;
    }
    return (size_t)value;
}

/**
 * @brief Normalize a 2D vector when possible.
 */
static void normalize_vector(float* x, float* y) {
    float length = sqrtf((*x * *x) + (*y * *y));

    if (length > 0.00001f) {
        *x /= length;
        *y /= length;
    }
}

/**
 * @brief Project a world-space point to the text map.
 */
static void mark_context_point(
    char map[HYBRID_ROUTE_MAP_SIZE][HYBRID_ROUTE_MAP_SIZE + 1U],
    float world_x,
    float world_y,
    char marker
) {
    float normalized_x = (world_x + 1.0f) * 0.5f;
    float normalized_y = (world_y + 1.0f) * 0.5f;
    int map_x = (int)lroundf(normalized_x * (float)(HYBRID_ROUTE_MAP_SIZE - 1U));
    int map_y = (int)lroundf((1.0f - normalized_y) * (float)(HYBRID_ROUTE_MAP_SIZE - 1U));

    map[clamp_map_index(map_y)][clamp_map_index(map_x)] = marker;
}

/**
 * @brief Turn four route cues into a readable mission statement.
 */
static const char* classify_intent(const float* input) {
    float avg_x = (input[0] + input[2] + input[4] + input[6]) * 0.25f;
    float avg_y = (input[1] + input[3] + input[5] + input[7]) * 0.25f;
    float turn_strength = input[6] - input[0];

    if (avg_y > 0.35f && turn_strength > 0.35f) {
        return "Goal: drive forward, then enter the right-hand branch.";
    }
    if (avg_y > 0.35f && turn_strength < -0.35f) {
        return "Goal: drive forward, then enter the left-hand branch.";
    }
    if (avg_y > 0.35f && fabsf(avg_x) < 0.20f) {
        return "Goal: continue straight through the main aisle.";
    }
    if (avg_y < -0.25f) {
        return "Goal: back out and retreat from the current lane.";
    }
    if (avg_x > 0.35f) {
        return "Goal: shift right toward the side corridor.";
    }
    if (avg_x < -0.35f) {
        return "Goal: shift left toward the side corridor.";
    }
    return "Goal: hold the current corridor and smooth the route.";
}

/**
 * @brief Explain the immediate next move in natural language.
 */
static const char* classify_step(float dx, float dy) {
    if (fabsf(dx) < 0.20f && fabsf(dy) < 0.20f) {
        return "short corrective move";
    }
    if (dy > 0.35f && dx > 0.25f) {
        return "commit to a forward-right turn";
    }
    if (dy > 0.35f && dx < -0.25f) {
        return "commit to a forward-left turn";
    }
    if (dy > 0.35f) {
        return "continue forward";
    }
    if (dy < -0.25f) {
        return "move backward / retreat";
    }
    if (dx > 0.25f) {
        return "slide right";
    }
    if (dx < -0.25f) {
        return "slide left";
    }
    return "small diagonal correction";
}

/**
 * @brief Draw a simple intersection backdrop so humans can see the route.
 */
static void mark_reference_road(char map[HYBRID_ROUTE_MAP_SIZE][HYBRID_ROUTE_MAP_SIZE + 1U]) {
    size_t row;
    size_t column;

    for (row = 0U; row < HYBRID_ROUTE_MAP_SIZE; ++row) {
        map[row][5] = '|';
    }
    for (column = 5U; column < HYBRID_ROUTE_MAP_SIZE; ++column) {
        map[2][column] = '-';
    }
    for (column = 0U; column <= 5U; ++column) {
        map[2][column] = '-';
    }
}

/**
 * @brief Parse one CLI line into the fixed 8-float cue vector.
 */
static int read_input_vector(float input[HYBRID_ROUTE_INPUT_SIZE]) {
    char line[256];
    char* cursor;
    char* end_ptr;
    size_t index;

    if (fgets(line, (int)sizeof(line), stdin) == NULL) {
        return 0;
    }

    cursor = line;
    for (index = 0U; index < HYBRID_ROUTE_INPUT_SIZE; ++index) {
        input[index] = strtof(cursor, &end_ptr);
        if (end_ptr == cursor) {
            return 0;
        }
        cursor = end_ptr;
    }

    while (*cursor != '\0') {
        if (!isspace((unsigned char)*cursor)) {
            return 0;
        }
        ++cursor;
    }

    return 1;
}

/**
 * @brief Render the frame map and the human-readable intent.
 */
static void render_context_map(
    const float* history_positions,
    const float current_x,
    const float current_y,
    const float* output,
    const float* input,
    size_t frame_index
) {
    char map[HYBRID_ROUTE_MAP_SIZE][HYBRID_ROUTE_MAP_SIZE + 1U];
    float predicted_x = current_x + (output[0] * HYBRID_ROUTE_STEP_SCALE);
    float predicted_y = current_y + (output[1] * HYBRID_ROUTE_STEP_SCALE);
    size_t row;
    size_t column;

    for (row = 0U; row < HYBRID_ROUTE_MAP_SIZE; ++row) {
        for (column = 0U; column < HYBRID_ROUTE_MAP_SIZE; ++column) {
            map[row][column] = '.';
        }
        map[row][HYBRID_ROUTE_MAP_SIZE] = '\0';
    }

    mark_reference_road(map);
    mark_context_point(map, 0.0f, 0.95f, 'G');
    mark_context_point(map, 0.85f, 0.55f, 'R');
    mark_context_point(map, -0.85f, 0.55f, 'L');

    for (row = 0U; row < 4U; ++row) {
        mark_context_point(map, history_positions[row * 2U], history_positions[row * 2U + 1U], 'o');
    }
    mark_context_point(map, current_x, current_y, 'C');
    mark_context_point(map, predicted_x, predicted_y, 'P');

    printf(
        "\nFrame %zu\n"
        "Map (| main aisle, - branch lane, G=go straight zone, R=right branch, L=left branch, o=recent route, C=current, P=predicted next step)\n",
        frame_index + 1U
    );
    for (row = 0U; row < HYBRID_ROUTE_MAP_SIZE; ++row) {
        printf("%s\n", map[row]);
    }
    printf("%s\n", classify_intent(input));
    for (row = 0U; row < 4U; ++row) {
        printf(
            "Cue %zu -> (%.2f, %.2f)\n",
            row + 1U,
            input[row * 2U],
            input[row * 2U + 1U]
        );
    }
    printf("Current position -> (%.2f, %.2f)\n", current_x, current_y);
    printf("Predicted route vector -> (%.3f, %.3f)\n", output[0], output[1]);
    printf("Immediate action -> %s\n", classify_step(output[0], output[1]));
    printf("Predicted next position -> (%.2f, %.2f)\n", predicted_x, predicted_y);
}

/**
 * @brief Convert route cue vectors into a visible path.
 */
static void build_history_positions(const float* input, float* history_positions, float* out_current_x, float* out_current_y) {
    float current_x = 0.0f;
    float current_y = -0.85f;
    size_t pair_index;

    for (pair_index = 0U; pair_index < 4U; ++pair_index) {
        current_x += input[pair_index * 2U] * HYBRID_ROUTE_STEP_SCALE;
        current_y += input[pair_index * 2U + 1U] * HYBRID_ROUTE_STEP_SCALE;
        history_positions[pair_index * 2U] = current_x;
        history_positions[pair_index * 2U + 1U] = current_y;
    }

    *out_current_x = current_x;
    *out_current_y = current_y;
}

int main(void) {
    void* infer_ctx;
    const char* weights_file = "../../data/weights.bin";
    float input[HYBRID_ROUTE_INPUT_SIZE];
    float output[2];
    float history_positions[8];
    float current_x;
    float current_y;
    size_t frame_index;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Hybrid Route Inference ===\n");
    printf("Scene: a warehouse robot approaches an intersection.\n");
    printf("Input meaning: 4 recent local route cues from the upstream planner.\n");
    printf("Human goal: understand whether the robot should keep straight, turn left, or turn right.\n");
    printf("Enter 8 floats:\n");
    printf("cue1_x cue1_y cue2_x cue2_y cue3_x cue3_y cue4_x cue4_y\n\n");

    if (!read_input_vector(input)) {
        fprintf(stderr, "Invalid input\n");
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

    build_history_positions(input, history_positions, &current_x, &current_y);

    printf("\n=== Route rollout ===\n");
    for (frame_index = 0U; frame_index < HYBRID_ROUTE_MAX_FRAMES; ++frame_index) {
        if (infer_auto_run(infer_ctx, input, output) != 0) {
            fprintf(stderr, "Inference failed during rollout\n");
            infer_destroy(infer_ctx);
            return 1;
        }

        normalize_vector(&output[0], &output[1]);
        render_context_map(history_positions, current_x, current_y, output, input, frame_index);

        current_x += output[0] * HYBRID_ROUTE_STEP_SCALE;
        current_y += output[1] * HYBRID_ROUTE_STEP_SCALE;

        history_positions[0] = history_positions[2];
        history_positions[1] = history_positions[3];
        history_positions[2] = history_positions[4];
        history_positions[3] = history_positions[5];
        history_positions[4] = history_positions[6];
        history_positions[5] = history_positions[7];
        history_positions[6] = current_x;
        history_positions[7] = current_y;

        input[0] = input[2];
        input[1] = input[3];
        input[2] = input[4];
        input[3] = input[5];
        input[4] = input[6];
        input[5] = input[7];
        input[6] = output[0];
        input[7] = output[1];
    }

    infer_destroy(infer_ctx);
    return 0;
}
