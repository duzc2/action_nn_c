#include "infer.h"
#include "weights_load.h"
#include "../demo_runtime_paths.h"

#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TARGET_MAP_SIZE 50U
#define TARGET_REQUIRED_GOALS 3U
#define TARGET_MAX_STEPS 200U
#define TARGET_SPEED 2.0f
#define TARGET_START_X 25.0f
#define TARGET_START_Y 25.0f
#define TARGET_REACH_EPSILON 1.0f
#define TARGET_MIN_RANDOM_DISTANCE 8.0f

/**
 * @brief Normalize target demo input using documented field order.
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
 * @brief Clamp a world coordinate into the printable 50x50 map.
 */
static float clamp_world_coordinate(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > (float)(TARGET_MAP_SIZE - 1U)) {
        return (float)(TARGET_MAP_SIZE - 1U);
    }
    return value;
}

/**
 * @brief Clamp a signed integer index into the printable map bounds.
 */
static size_t clamp_map_index(int value) {
    if (value < 0) {
        return 0U;
    }
    if (value >= (int)TARGET_MAP_SIZE) {
        return TARGET_MAP_SIZE - 1U;
    }
    return (size_t)value;
}

/**
 * @brief Compute Euclidean distance for readable arrival checks.
 */
static float distance_between_points(float ax, float ay, float bx, float by) {
    float dx = bx - ax;
    float dy = by - ay;

    return sqrtf((dx * dx) + (dy * dy));
}

/**
 * @brief Keep the raw network output inside the expected unit-step ball.
 *
 * The trainer teaches a normalized direction vector whose magnitude should
 * stay inside [0, 1]. This clamp keeps the demo readable even if the network
 * predicts a slightly longer vector.
 */
static void clamp_prediction_vector(float* dx, float* dy) {
    float length = sqrtf((*dx * *dx) + (*dy * *dy));

    if (length > 1.0f) {
        *dx /= length;
        *dy /= length;
    }
}

/**
 * @brief Pick a new pseudo-random target that is not too close to the agent.
 *
 * A fixed seed is used by main() so the "random" demo remains reproducible
 * while still exercising multiple target locations automatically.
 */
static void choose_next_target(
    float* target_x,
    float* target_y,
    float current_x,
    float current_y
) {
    size_t attempt;

    for (attempt = 0U; attempt < 64U; ++attempt) {
        float candidate_x = (float)(rand() % (int)TARGET_MAP_SIZE);
        float candidate_y = (float)(rand() % (int)TARGET_MAP_SIZE);

        if (distance_between_points(current_x, current_y, candidate_x, candidate_y) >=
            TARGET_MIN_RANDOM_DISTANCE) {
            *target_x = candidate_x;
            *target_y = candidate_y;
            return;
        }
    }

    *target_x = (float)(rand() % (int)TARGET_MAP_SIZE);
    *target_y = (float)(rand() % (int)TARGET_MAP_SIZE);
}

/**
 * @brief Build a per-run seed so repeated demo launches produce new targets.
 *
 * time(NULL) provides cross-run variation, and the pointer-based xor adds a
 * little more entropy if two launches happen within the same second.
 */
static unsigned int build_random_seed(void* infer_ctx) {
    unsigned int seed = (unsigned int)time(NULL);

    seed ^= (unsigned int)((uintptr_t)infer_ctx & 0xffffffffu);
    seed ^= (unsigned int)(((uintptr_t)&infer_ctx >> 4U) & 0xffffffffu);
    return seed;
}

/**
 * @brief Mark a world-space point on the ASCII map.
 *
 * The map prints y=49 at the top row so the display looks like a normal
 * Cartesian top-down view.
 */
static void mark_point(
    char map[TARGET_MAP_SIZE][TARGET_MAP_SIZE + 1U],
    float world_x,
    float world_y,
    char marker
) {
    int map_x = (int)lroundf(clamp_world_coordinate(world_x));
    int map_y = (int)lroundf(clamp_world_coordinate(world_y));
    size_t row = clamp_map_index((int)(TARGET_MAP_SIZE - 1U) - map_y);
    size_t column = clamp_map_index(map_x);

    map[row][column] = marker;
}

/**
 * @brief Render the 50x50 demo map with trail, target, current and next step.
 */
static void render_map(
    const float* trail_x,
    const float* trail_y,
    size_t trail_count,
    float current_x,
    float current_y,
    float target_x,
    float target_y,
    float next_x,
    float next_y,
    size_t step_index,
    float normalized_dx,
    float normalized_dy,
    float applied_dx,
    float applied_dy,
    int reached_target,
    size_t reached_goal_count
) {
    char map[TARGET_MAP_SIZE][TARGET_MAP_SIZE + 1U];
    size_t row;
    size_t column;
    size_t trail_index;

    for (row = 0U; row < TARGET_MAP_SIZE; ++row) {
        for (column = 0U; column < TARGET_MAP_SIZE; ++column) {
            map[row][column] = '.';
        }
        map[row][TARGET_MAP_SIZE] = '\0';
    }

    for (trail_index = 0U; trail_index < trail_count; ++trail_index) {
        mark_point(map, trail_x[trail_index], trail_y[trail_index], 'o');
    }

    mark_point(map, TARGET_START_X, TARGET_START_Y, 'S');
    mark_point(map, target_x, target_y, 'T');
    mark_point(map, current_x, current_y, 'C');
    mark_point(map, next_x, next_y, reached_target ? 'T' : 'P');

    printf(
        "\nStep %zu/%u  Goals reached: %zu/%u\n"
        "Map legend: .=empty o=trail S=start C=current P=predicted next T=target\n",
        step_index + 1U,
        TARGET_MAX_STEPS,
        reached_goal_count,
        TARGET_REQUIRED_GOALS
    );
    for (row = 0U; row < TARGET_MAP_SIZE; ++row) {
        printf("%s\n", map[row]);
    }
    printf("Current position  -> (%.2f, %.2f)\n", current_x, current_y);
    printf("Target position   -> (%.2f, %.2f)\n", target_x, target_y);
    printf("Normalized output -> (%.4f, %.4f)\n", normalized_dx, normalized_dy);
    printf("Applied delta     -> (%.4f, %.4f)\n", applied_dx, applied_dy);
    printf("Next position     -> (%.2f, %.2f)\n", next_x, next_y);
    printf(
        "Remaining dist.   -> %.4f%s\n",
        distance_between_points(next_x, next_y, target_x, target_y),
        reached_target ? " (target reached)" : ""
    );
}

int main(void) {
    void* infer_ctx;
    const char* weights_file = "../../data/weights.bin";
    float current_x = TARGET_START_X;
    float current_y = TARGET_START_Y;
    float target_x = 0.0f;
    float target_y = 0.0f;
    float input[5];
    float output[2];
    float trail_x[TARGET_MAX_STEPS + 1U];
    float trail_y[TARGET_MAX_STEPS + 1U];
    unsigned int random_seed;
    size_t reached_goal_count = 0U;
    size_t trail_count = 0U;
    size_t step_index;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Target Network Inference ===\n");
    printf("Auto visual rollout: 50x50 map, start=(25,25), speed=2.\n");
    printf(
        "Stop rule: finish after reaching %u targets, or stop at %u steps as a safety cap.\n",
        TARGET_REQUIRED_GOALS,
        TARGET_MAX_STEPS
    );
    printf("A per-run random seed is used so each demo launch gets new target positions.\n\n");

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

    random_seed = build_random_seed(infer_ctx);
    srand(random_seed);
    printf("Random seed -> %u\n\n", random_seed);
    choose_next_target(&target_x, &target_y, current_x, current_y);

    trail_x[trail_count] = current_x;
    trail_y[trail_count] = current_y;
    trail_count += 1U;

    for (step_index = 0U;
         (step_index < TARGET_MAX_STEPS) && (reached_goal_count < TARGET_REQUIRED_GOALS);
         ++step_index) {
        float predicted_dx;
        float predicted_dy;
        float next_x;
        float next_y;
        int reached_target = 0;

        build_input(input, target_x, target_y, current_x, current_y, TARGET_SPEED);
        if (infer_auto_run(infer_ctx, input, output) != 0) {
            fprintf(stderr, "inference failed during step %zu\n", step_index + 1U);
            infer_destroy(infer_ctx);
            return 1;
        }

        predicted_dx = output[0];
        predicted_dy = output[1];
        clamp_prediction_vector(&predicted_dx, &predicted_dy);

        next_x = clamp_world_coordinate(current_x + predicted_dx * TARGET_SPEED);
        next_y = clamp_world_coordinate(current_y + predicted_dy * TARGET_SPEED);
        if (distance_between_points(next_x, next_y, target_x, target_y) <= TARGET_REACH_EPSILON) {
            next_x = target_x;
            next_y = target_y;
            reached_target = 1;
        }

        render_map(
            trail_x,
            trail_y,
            trail_count,
            current_x,
            current_y,
            target_x,
            target_y,
            next_x,
            next_y,
            step_index,
            predicted_dx,
            predicted_dy,
            predicted_dx * TARGET_SPEED,
            predicted_dy * TARGET_SPEED,
            reached_target,
            reached_goal_count + (reached_target ? 1U : 0U)
        );

        current_x = next_x;
        current_y = next_y;
        if (trail_count < (TARGET_MAX_STEPS + 1U)) {
            trail_x[trail_count] = current_x;
            trail_y[trail_count] = current_y;
            trail_count += 1U;
        }

        if (reached_target) {
            float previous_target_x = target_x;
            float previous_target_y = target_y;

            reached_goal_count += 1U;
            if (reached_goal_count < TARGET_REQUIRED_GOALS) {
                choose_next_target(&target_x, &target_y, current_x, current_y);
                printf(
                    "Reached target %zu/%u at (%.2f, %.2f). Next target -> (%.2f, %.2f)\n",
                    reached_goal_count,
                    TARGET_REQUIRED_GOALS,
                    previous_target_x,
                    previous_target_y,
                    target_x,
                    target_y
                );
            } else {
                printf(
                    "Reached target %zu/%u at (%.2f, %.2f). Demo success condition satisfied.\n",
                    reached_goal_count,
                    TARGET_REQUIRED_GOALS,
                    previous_target_x,
                    previous_target_y
                );
            }
        }
    }

    if (reached_goal_count >= TARGET_REQUIRED_GOALS) {
        printf(
            "\nDemo finished successfully after %zu steps. Final position -> (%.2f, %.2f)\n",
            step_index,
            current_x,
            current_y
        );
    } else {
        printf(
            "\nDemo stopped at safety cap: %u steps, %zu/%u targets reached. Final position -> (%.2f, %.2f)\n",
            TARGET_MAX_STEPS,
            reached_goal_count,
            TARGET_REQUIRED_GOALS,
            current_x,
            current_y
        );
    }

    infer_destroy(infer_ctx);
    return 0;
}
