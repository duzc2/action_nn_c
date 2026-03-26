/**
 * @file infer_main.c
 * @brief Nested navigation demo inference entry
 */

#include "infer.h"
#include "weights_load.h"
#include "../demo_runtime_paths.h"
#include "nested_nav_scene.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

#define NESTED_NAV_MAP_SIZE 21U
#define NESTED_NAV_MAX_FRAMES 40U
#define NESTED_NAV_STEP_SCALE 0.15f

/**
 * @brief Clamp a map coordinate into the visible text map.
 */
static size_t clamp_map_index(int value) {
    if (value < 0) {
        return 0U;
    }
    if (value >= (int)NESTED_NAV_MAP_SIZE) {
        return NESTED_NAV_MAP_SIZE - 1U;
    }
    return (size_t)value;
}

/**
 * @brief Normalize the predicted motion so the rollout step length stays bounded.
 */
static void normalize_vector(float* x, float* y) {
    float length = sqrtf((*x * *x) + (*y * *y));

    if (length > 1.0f) {
        *x /= length;
        *y /= length;
    }
}

/**
 * @brief Mark a world-space point onto the ASCII map.
 */
static void mark_world_point(
    char map[NESTED_NAV_MAP_SIZE][NESTED_NAV_MAP_SIZE + 1U],
    float world_x,
    float world_y,
    char marker
) {
    float normalized_x = (world_x - NESTED_NAV_WORLD_MIN) / (NESTED_NAV_WORLD_MAX - NESTED_NAV_WORLD_MIN);
    float normalized_y = (world_y - NESTED_NAV_WORLD_MIN) / (NESTED_NAV_WORLD_MAX - NESTED_NAV_WORLD_MIN);
    int map_x = (int)lroundf(normalized_x * (float)(NESTED_NAV_MAP_SIZE - 1U));
    int map_y = (int)lroundf((1.0f - normalized_y) * (float)(NESTED_NAV_MAP_SIZE - 1U));

    map[clamp_map_index(map_y)][clamp_map_index(map_x)] = marker;
}

/**
 * @brief Rasterize one circular obstacle into the ASCII map.
 */
static void mark_obstacle_area(
    char map[NESTED_NAV_MAP_SIZE][NESTED_NAV_MAP_SIZE + 1U],
    float world_x,
    float world_y,
    float radius
) {
    size_t row;
    size_t column;

    for (row = 0U; row < NESTED_NAV_MAP_SIZE; ++row) {
        for (column = 0U; column < NESTED_NAV_MAP_SIZE; ++column) {
            float sample_x = NESTED_NAV_WORLD_MIN +
                ((float)column * (NESTED_NAV_WORLD_MAX - NESTED_NAV_WORLD_MIN) / (float)(NESTED_NAV_MAP_SIZE - 1U));
            float sample_y = NESTED_NAV_WORLD_MAX -
                ((float)row * (NESTED_NAV_WORLD_MAX - NESTED_NAV_WORLD_MIN) / (float)(NESTED_NAV_MAP_SIZE - 1U));

            if (nested_nav_distance(sample_x, sample_y, world_x, world_y) <= radius) {
                map[row][column] = 'X';
            }
        }
    }
}

/**
 * @brief Choose a compact direction glyph for the predicted next move.
 */
static char direction_marker(float dx, float dy) {
    if (fabsf(dx) < 0.15f && fabsf(dy) < 0.15f) {
        return '*';
    }
    if (fabsf(dx) > fabsf(dy)) {
        return dx >= 0.0f ? '>' : '<';
    }
    return dy >= 0.0f ? '^' : 'v';
}

/**
 * @brief Draw the current rollout frame.
 */
static void render_navigation_map(
    const NestedNavScene* scene,
    const float* output,
    const float (*trail)[2],
    size_t trail_count,
    size_t frame_index,
    int collision_detected
) {
    char map[NESTED_NAV_MAP_SIZE][NESTED_NAV_MAP_SIZE + 1U];
    float predicted_x = scene->current_x + (output[0] * NESTED_NAV_STEP_SCALE);
    float predicted_y = scene->current_y + (output[1] * NESTED_NAV_STEP_SCALE);
    size_t trail_index;
    size_t row;
    size_t column;
    size_t obstacle_index;

    for (row = 0U; row < NESTED_NAV_MAP_SIZE; ++row) {
        for (column = 0U; column < NESTED_NAV_MAP_SIZE; ++column) {
            map[row][column] = '.';
        }
        map[row][NESTED_NAV_MAP_SIZE] = '\0';
    }

    for (obstacle_index = 0U; obstacle_index < NESTED_NAV_OBSTACLE_COUNT; ++obstacle_index) {
        mark_obstacle_area(
            map,
            scene->obstacle_centers[obstacle_index][0],
            scene->obstacle_centers[obstacle_index][1],
            g_nested_nav_obstacle_radii[obstacle_index]
        );
    }

    for (trail_index = 0U; trail_index < trail_count; ++trail_index) {
        mark_world_point(map, trail[trail_index][0], trail[trail_index][1], 'o');
    }

    mark_world_point(map, scene->target_x, scene->target_y, 'T');
    mark_world_point(map, scene->current_x, scene->current_y, 'C');
    mark_world_point(map, predicted_x, predicted_y, collision_detected ? '!' : direction_marker(output[0], output[1]));

    printf(
        "\nFrame %zu\n"
        "Map view (o=trail, C=current, T=target, X=obstacle area, ^=v<>/* next step, !=collision)\n",
        frame_index + 1U
    );
    for (row = 0U; row < NESTED_NAV_MAP_SIZE; ++row) {
        printf("%s\n", map[row]);
    }
    printf(
        "Current=(%.2f, %.2f)  Target=(%.2f, %.2f)\n",
        scene->current_x,
        scene->current_y,
        scene->target_x,
        scene->target_y
    );
    for (obstacle_index = 0U; obstacle_index < NESTED_NAV_OBSTACLE_COUNT; ++obstacle_index) {
        printf(
            "Obstacle %zu=(%.2f, %.2f) r=%.2f\n",
            obstacle_index + 1U,
            scene->obstacle_centers[obstacle_index][0],
            scene->obstacle_centers[obstacle_index][1],
            g_nested_nav_obstacle_radii[obstacle_index]
        );
    }
    printf("Predicted next position=(%.2f, %.2f)\n", predicted_x, predicted_y);
    if (collision_detected) {
        printf("WARNING: predicted segment intersects obstacle occupancy.\n");
    }
}

/**
 * @brief Reject rollout steps that would cross any obstacle.
 */
static int predicted_step_collides(const NestedNavScene* scene, const float* output) {
    float next_x = scene->current_x + (output[0] * NESTED_NAV_STEP_SCALE);
    float next_y = scene->current_y + (output[1] * NESTED_NAV_STEP_SCALE);
    size_t obstacle_index;

    for (obstacle_index = 0U; obstacle_index < NESTED_NAV_OBSTACLE_COUNT; ++obstacle_index) {
        if (nested_nav_segment_hits_obstacle(
                scene->current_x,
                scene->current_y,
                next_x,
                next_y,
                scene->obstacle_centers[obstacle_index][0],
                scene->obstacle_centers[obstacle_index][1],
                g_nested_nav_obstacle_radii[obstacle_index])) {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Return a positive clearance score for the candidate step endpoint.
 */
static float candidate_clearance(const NestedNavScene* scene, const float* output) {
    float next_x = scene->current_x + (output[0] * NESTED_NAV_STEP_SCALE);
    float next_y = scene->current_y + (output[1] * NESTED_NAV_STEP_SCALE);
    float best = 1000.0f;
    size_t obstacle_index;

    for (obstacle_index = 0U; obstacle_index < NESTED_NAV_OBSTACLE_COUNT; ++obstacle_index) {
        float distance = nested_nav_distance(
            next_x,
            next_y,
            scene->obstacle_centers[obstacle_index][0],
            scene->obstacle_centers[obstacle_index][1]
        ) - g_nested_nav_obstacle_radii[obstacle_index];
        if (distance < best) {
            best = distance;
        }
    }

    return best;
}

/**
 * @brief Normalize a direction vector only when it has usable length.
 */
static void normalize_if_possible(float* x, float* y) {
    float length = sqrtf((*x * *x) + (*y * *y));

    if (length > 0.00001f) {
        *x /= length;
        *y /= length;
    }
}

/**
 * @brief Pick a safe movement direction close to the model proposal.
 */
static int choose_safe_move(
    const NestedNavScene* scene,
    const float* raw_output,
    float* applied_output
) {
    static const float g_candidate_directions[16][2] = {
        { 1.000000f,  0.000000f},
        { 0.923880f,  0.382683f},
        { 0.707107f,  0.707107f},
        { 0.382683f,  0.923880f},
        { 0.000000f,  1.000000f},
        {-0.382683f,  0.923880f},
        {-0.707107f,  0.707107f},
        {-0.923880f,  0.382683f},
        {-1.000000f,  0.000000f},
        {-0.923880f, -0.382683f},
        {-0.707107f, -0.707107f},
        {-0.382683f, -0.923880f},
        { 0.000000f, -1.000000f},
        { 0.382683f, -0.923880f},
        { 0.707107f, -0.707107f},
        { 0.923880f, -0.382683f}
    };
    float raw_dir[2] = {raw_output[0], raw_output[1]};
    float target_dir[2] = {
        scene->target_x - scene->current_x,
        scene->target_y - scene->current_y
    };
    float best_score = -1000000.0f;
    int raw_dir_valid = 0;
    size_t candidate_index;
    int found = 0;

    if (sqrtf((raw_dir[0] * raw_dir[0]) + (raw_dir[1] * raw_dir[1])) > 0.00001f) {
        normalize_if_possible(&raw_dir[0], &raw_dir[1]);
        raw_dir_valid = 1;
    }
    normalize_if_possible(&target_dir[0], &target_dir[1]);

    applied_output[0] = 0.0f;
    applied_output[1] = 0.0f;

    if (raw_dir_valid) {
        float score;

        if (!predicted_step_collides(scene, raw_dir)) {
            score =
                1.25f * ((raw_dir[0] * target_dir[0]) + (raw_dir[1] * target_dir[1])) +
                0.90f +
                0.15f * candidate_clearance(scene, raw_dir);
            best_score = score;
            applied_output[0] = raw_dir[0];
            applied_output[1] = raw_dir[1];
            found = 1;
        }
    }

    for (candidate_index = 0U; candidate_index < 16U; ++candidate_index) {
        float candidate[2] = {
            g_candidate_directions[candidate_index][0],
            g_candidate_directions[candidate_index][1]
        };
        float score;

        if (predicted_step_collides(scene, candidate)) {
            continue;
        }

        score =
            1.25f * ((candidate[0] * target_dir[0]) + (candidate[1] * target_dir[1])) +
            (raw_dir_valid ? (0.75f * ((candidate[0] * raw_dir[0]) + (candidate[1] * raw_dir[1]))) : 0.0f) +
            0.15f * candidate_clearance(scene, candidate);

        if (!found || score > best_score) {
            best_score = score;
            applied_output[0] = candidate[0];
            applied_output[1] = candidate[1];
            found = 1;
        }
    }

    if (!found) {
        return -1;
    }

    if (!raw_dir_valid) {
        return 1;
    }

    return (fabsf(applied_output[0] - raw_dir[0]) > 0.001f || fabsf(applied_output[1] - raw_dir[1]) > 0.001f) ? 1 : 0;
}

int main(void) {
    void* infer_ctx;
    const char* weights_file = "../../data/weights.bin";
    NestedNavScene scene;
    unsigned int rng_state = ((unsigned int)time(NULL)) ^ ((unsigned int)clock()) ^ 0x4E41564EU;
    float input[NESTED_NAV_INPUT_SIZE];
    float output[NESTED_NAV_OUTPUT_SIZE];
    float trail[NESTED_NAV_MAX_FRAMES + 1U][2];
    size_t trail_count = 0U;
    size_t frame_index;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Nested Navigation Inference ===\n");
    printf("Scenario: lower-left start, upper-right target, random diagonal obstacles sampled for this run.\n");
    printf("Random seed: %u\n", rng_state);

    nested_nav_generate_random_scene(&scene, &rng_state);
    nested_nav_fill_input(input, &scene);

    printf("Sampled obstacle layout:\n");
    printf("Start=(%.2f, %.2f)  Target=(%.2f, %.2f)\n",
        scene.current_x,
        scene.current_y,
        scene.target_x,
        scene.target_y);
    printf("Obstacle 1=(%.2f, %.2f) r=%.2f\n",
        scene.obstacle_centers[0][0],
        scene.obstacle_centers[0][1],
        g_nested_nav_obstacle_radii[0]);
    printf("Obstacle 2=(%.2f, %.2f) r=%.2f\n",
        scene.obstacle_centers[1][0],
        scene.obstacle_centers[1][1],
        g_nested_nav_obstacle_radii[1]);
    printf("Obstacle 3=(%.2f, %.2f) r=%.2f\n\n",
        scene.obstacle_centers[2][0],
        scene.obstacle_centers[2][1],
        g_nested_nav_obstacle_radii[2]);

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

    trail[trail_count][0] = scene.current_x;
    trail[trail_count][1] = scene.current_y;
    trail_count += 1U;

    printf("=== Navigation rollout ===\n");
    for (frame_index = 0U; frame_index < NESTED_NAV_MAX_FRAMES; ++frame_index) {
        int safety_status;
        float applied_output[NESTED_NAV_OUTPUT_SIZE];

        nested_nav_fill_input(input, &scene);
        if (infer_auto_run(infer_ctx, input, output) != 0) {
            fprintf(stderr, "Inference failed during rollout\n");
            infer_destroy(infer_ctx);
            return 1;
        }

        normalize_vector(&output[0], &output[1]);
        safety_status = choose_safe_move(&scene, output, applied_output);
        if (safety_status < 0) {
            printf("No safe step is available from the current position.\n");
            break;
        }
        printf("Raw predicted move: dx=%.3f dy=%.3f\n", output[0], output[1]);
        if (safety_status > 0) {
            printf(
                "Safety gate adjusted move to: dx=%.3f dy=%.3f\n",
                applied_output[0],
                applied_output[1]
            );
        }
        render_navigation_map(&scene, applied_output, trail, trail_count, frame_index, 0);

        scene.current_x += applied_output[0] * NESTED_NAV_STEP_SCALE;
        scene.current_y += applied_output[1] * NESTED_NAV_STEP_SCALE;
        trail[trail_count][0] = scene.current_x;
        trail[trail_count][1] = scene.current_y;
        trail_count += 1U;

        if (nested_nav_distance(
                scene.current_x,
                scene.current_y,
                scene.target_x,
                scene.target_y) < 0.16f) {
            printf("Reached target neighborhood after %zu frame(s).\n", frame_index + 1U);
            break;
        }
    }

    infer_destroy(infer_ctx);
    return 0;
}
