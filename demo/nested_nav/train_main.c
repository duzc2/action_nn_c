/**
 * @file train_main.c
 * @brief Nested navigation demo training entry
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "../demo_runtime_paths.h"
#include "nested_nav_scene.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define NESTED_NAV_EPOCHS 60
#define NESTED_NAV_SCENES_PER_EPOCH 4U
#define NESTED_NAV_GRID_SIZE 21U
#define NESTED_NAV_ASTAR_NODE_COUNT (NESTED_NAV_GRID_SIZE * NESTED_NAV_GRID_SIZE)
#define NESTED_NAV_PLANNER_CLEARANCE 0.08f

/**
 * @brief One node in the coarse A* grid.
 */
typedef struct {
    float g_score;
    float f_score;
    int parent;
    unsigned char open;
    unsigned char closed;
    unsigned char blocked;
} AStarNode;

/**
 * @brief Clamp a normalized movement component into the model output range.
 */
static float clamp_magnitude(float value) {
    if (value > 1.0f) {
        return 1.0f;
    }
    if (value < -1.0f) {
        return -1.0f;
    }
    return value;
}

/**
 * @brief Normalize a 2D vector in-place.
 */
static void normalize_vector(float* x, float* y) {
    float length = sqrtf((*x * *x) + (*y * *y));

    if (length > 0.00001f) {
        *x /= length;
        *y /= length;
    }
}

/**
 * @brief Return the world-space distance represented by one grid cell.
 */
static float grid_step_size(void) {
    return (NESTED_NAV_WORLD_MAX - NESTED_NAV_WORLD_MIN) /
        (float)(NESTED_NAV_GRID_SIZE - 1U);
}

/**
 * @brief Map a world coordinate to the nearest grid coordinate.
 */
static int world_to_grid_index(float value) {
    float normalized = (value - NESTED_NAV_WORLD_MIN) / grid_step_size();
    int index = (int)lroundf(normalized);

    if (index < 0) {
        return 0;
    }
    if (index >= (int)NESTED_NAV_GRID_SIZE) {
        return (int)NESTED_NAV_GRID_SIZE - 1;
    }
    return index;
}

/**
 * @brief Map a grid coordinate back to world space.
 */
static float grid_to_world_value(int index) {
    return NESTED_NAV_WORLD_MIN + ((float)index * grid_step_size());
}

/**
 * @brief Flatten a grid coordinate into the node array.
 */
static int grid_cell_index(int x, int y) {
    return y * (int)NESTED_NAV_GRID_SIZE + x;
}

/**
 * @brief Expand a flattened node index back into grid coordinates.
 */
static void index_to_grid_cell(int index, int* x, int* y) {
    *x = index % (int)NESTED_NAV_GRID_SIZE;
    *y = index / (int)NESTED_NAV_GRID_SIZE;
}

/**
 * @brief Check whether a world-space point is inside an obstacle buffer.
 */
static int is_blocked_world(
    const NestedNavScene* scene,
    float world_x,
    float world_y,
    float extra_clearance
) {
    size_t obstacle_index;

    for (obstacle_index = 0U; obstacle_index < NESTED_NAV_OBSTACLE_COUNT; ++obstacle_index) {
        float radius = g_nested_nav_obstacle_radii[obstacle_index] + extra_clearance;
        if (nested_nav_distance(
                world_x,
                world_y,
                scene->obstacle_centers[obstacle_index][0],
                scene->obstacle_centers[obstacle_index][1]) <= radius) {
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Check whether a segment would clip any buffered obstacle.
 */
static int segment_blocked_world(
    const NestedNavScene* scene,
    float start_x,
    float start_y,
    float end_x,
    float end_y,
    float extra_clearance
) {
    size_t obstacle_index;

    for (obstacle_index = 0U; obstacle_index < NESTED_NAV_OBSTACLE_COUNT; ++obstacle_index) {
        if (nested_nav_segment_hits_obstacle(
                start_x,
                start_y,
                end_x,
                end_y,
                scene->obstacle_centers[obstacle_index][0],
                scene->obstacle_centers[obstacle_index][1],
                g_nested_nav_obstacle_radii[obstacle_index] + extra_clearance)) {
            return 1;
        }
    }

    return 0;
}

/**
 * @brief Standard Euclidean heuristic for A* on the coarse grid.
 */
static float heuristic_cost(int node_index, int goal_index) {
    int node_x;
    int node_y;
    int goal_x;
    int goal_y;
    float dx;
    float dy;

    index_to_grid_cell(node_index, &node_x, &node_y);
    index_to_grid_cell(goal_index, &goal_x, &goal_y);
    dx = (float)(goal_x - node_x);
    dy = (float)(goal_y - node_y);
    return sqrtf((dx * dx) + (dy * dy));
}

/**
 * @brief Initialize the coarse occupancy grid for the current scene.
 */
static void initialize_astar_nodes(const NestedNavScene* scene, AStarNode* nodes) {
    size_t index;

    for (index = 0U; index < NESTED_NAV_ASTAR_NODE_COUNT; ++index) {
        int grid_x;
        int grid_y;
        float world_x;
        float world_y;

        index_to_grid_cell((int)index, &grid_x, &grid_y);
        world_x = grid_to_world_value(grid_x);
        world_y = grid_to_world_value(grid_y);

        nodes[index].g_score = FLT_MAX;
        nodes[index].f_score = FLT_MAX;
        nodes[index].parent = -1;
        nodes[index].open = 0U;
        nodes[index].closed = 0U;
        nodes[index].blocked = (unsigned char)(is_blocked_world(
            scene,
            world_x,
            world_y,
            NESTED_NAV_PLANNER_CLEARANCE
        ) ? 1 : 0);
    }
}

/**
 * @brief Pick the currently best open node.
 */
static int find_lowest_open_node(const AStarNode* nodes) {
    int best_index = -1;
    float best_score = FLT_MAX;
    size_t index;

    for (index = 0U; index < NESTED_NAV_ASTAR_NODE_COUNT; ++index) {
        if (!nodes[index].open) {
            continue;
        }
        if (nodes[index].f_score < best_score) {
            best_score = nodes[index].f_score;
            best_index = (int)index;
        }
    }

    return best_index;
}

/**
 * @brief Run A* and return the first step on the shortest safe route.
 */
static int astar_find_next_node(
    const NestedNavScene* scene,
    int start_index,
    int goal_index
) {
    AStarNode nodes[NESTED_NAV_ASTAR_NODE_COUNT];
    static const int g_neighbor_offsets[8][2] = {
        {-1, -1}, {0, -1}, {1, -1},
        {-1,  0},           {1,  0},
        {-1,  1}, {0,  1}, {1,  1}
    };

    initialize_astar_nodes(scene, nodes);
    if (nodes[start_index].blocked || nodes[goal_index].blocked) {
        return -1;
    }

    nodes[start_index].g_score = 0.0f;
    nodes[start_index].f_score = heuristic_cost(start_index, goal_index);
    nodes[start_index].open = 1U;

    for (;;) {
        int current_index = find_lowest_open_node(nodes);
        int current_x;
        int current_y;
        int neighbor_index;

        if (current_index < 0) {
            return -1;
        }
        if (current_index == goal_index) {
            int step_index = goal_index;
            while (nodes[step_index].parent >= 0 && nodes[step_index].parent != start_index) {
                step_index = nodes[step_index].parent;
            }
            return step_index == goal_index && start_index == goal_index ? start_index : step_index;
        }

        nodes[current_index].open = 0U;
        nodes[current_index].closed = 1U;
        index_to_grid_cell(current_index, &current_x, &current_y);

        for (neighbor_index = 0; neighbor_index < 8; ++neighbor_index) {
            int neighbor_x = current_x + g_neighbor_offsets[neighbor_index][0];
            int neighbor_y = current_y + g_neighbor_offsets[neighbor_index][1];
            int next_index;
            float current_world_x;
            float current_world_y;
            float next_world_x;
            float next_world_y;
            float tentative_g;

            if (neighbor_x < 0 || neighbor_y < 0 ||
                neighbor_x >= (int)NESTED_NAV_GRID_SIZE || neighbor_y >= (int)NESTED_NAV_GRID_SIZE) {
                continue;
            }

            next_index = grid_cell_index(neighbor_x, neighbor_y);
            if (nodes[next_index].blocked || nodes[next_index].closed) {
                continue;
            }

            current_world_x = grid_to_world_value(current_x);
            current_world_y = grid_to_world_value(current_y);
            next_world_x = grid_to_world_value(neighbor_x);
            next_world_y = grid_to_world_value(neighbor_y);

            if (segment_blocked_world(
                    scene,
                    current_world_x,
                    current_world_y,
                    next_world_x,
                    next_world_y,
                    NESTED_NAV_PLANNER_CLEARANCE)) {
                continue;
            }

            tentative_g = nodes[current_index].g_score +
                (((g_neighbor_offsets[neighbor_index][0] != 0) &&
                  (g_neighbor_offsets[neighbor_index][1] != 0)) ? 1.41421356f : 1.0f);

            if (!nodes[next_index].open || tentative_g < nodes[next_index].g_score) {
                nodes[next_index].parent = current_index;
                nodes[next_index].g_score = tentative_g;
                nodes[next_index].f_score = tentative_g + heuristic_cost(next_index, goal_index);
                nodes[next_index].open = 1U;
            }
        }
    }
}

/**
 * @brief Check that the fixed start and fixed target are connected in this scene.
 */
static int scene_has_reference_path(const NestedNavScene* scene) {
    int start_x = world_to_grid_index(g_nested_nav_start_position[0]);
    int start_y = world_to_grid_index(g_nested_nav_start_position[1]);
    int goal_x = world_to_grid_index(g_nested_nav_target_position[0]);
    int goal_y = world_to_grid_index(g_nested_nav_target_position[1]);

    return astar_find_next_node(
        scene,
        grid_cell_index(start_x, start_y),
        grid_cell_index(goal_x, goal_y)
    ) >= 0 ? 1 : 0;
}

/**
 * @brief Generate a random training scene and reject scenes without a safe route.
 */
static int generate_training_scene(NestedNavScene* scene, unsigned int* state) {
    size_t attempt;

    for (attempt = 0U; attempt < 128U; ++attempt) {
        nested_nav_generate_random_scene(scene, state);
        if (scene_has_reference_path(scene)) {
            return 0;
        }
    }

    return -1;
}

/**
 * @brief Build the teacher action for the current scene position.
 */
static int build_expected_action(const NestedNavScene* scene, float* expected) {
    int start_x = world_to_grid_index(scene->current_x);
    int start_y = world_to_grid_index(scene->current_y);
    int goal_x = world_to_grid_index(scene->target_x);
    int goal_y = world_to_grid_index(scene->target_y);
    int start_index = grid_cell_index(start_x, start_y);
    int goal_index = grid_cell_index(goal_x, goal_y);
    int next_index;
    int next_x;
    int next_y;
    float next_world_x;
    float next_world_y;

    if (nested_nav_distance(
            scene->current_x,
            scene->current_y,
            scene->target_x,
            scene->target_y) < 0.12f) {
        expected[0] = 0.0f;
        expected[1] = 0.0f;
        return 0;
    }

    next_index = astar_find_next_node(scene, start_index, goal_index);
    if (next_index < 0) {
        return -1;
    }

    if (next_index == start_index) {
        expected[0] = 0.0f;
        expected[1] = 0.0f;
        return 0;
    }

    index_to_grid_cell(next_index, &next_x, &next_y);
    next_world_x = grid_to_world_value(next_x);
    next_world_y = grid_to_world_value(next_y);

    if (segment_blocked_world(
            scene,
            scene->current_x,
            scene->current_y,
            next_world_x,
            next_world_y,
            NESTED_NAV_PLANNER_CLEARANCE)) {
        return -1;
    }

    expected[0] = next_world_x - scene->current_x;
    expected[1] = next_world_y - scene->current_y;
    normalize_vector(&expected[0], &expected[1]);
    expected[0] = clamp_magnitude(expected[0]);
    expected[1] = clamp_magnitude(expected[1]);
    return 0;
}

/**
 * @brief Mean squared error for log output.
 */
static float compute_loss(const float* output, const float* expected, size_t size) {
    float loss = 0.0f;
    size_t index;

    for (index = 0U; index < size; ++index) {
        float diff = output[index] - expected[index];
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
    unsigned int rng_state = ((unsigned int)time(NULL)) ^ ((unsigned int)clock()) ^ 0x6E61764EU;
    float input[NESTED_NAV_INPUT_SIZE];
    float expected[NESTED_NAV_OUTPUT_SIZE];
    float output[NESTED_NAV_OUTPUT_SIZE];
    size_t sample_count_per_epoch = 0U;
    size_t skipped_points_per_epoch = 0U;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Nested Navigation Training ===\n");
    printf("Scenario: fixed start in lower-left, fixed target in upper-right, random obstacles sampled on the diagonal corridor.\n");
    printf("Label generator: A* grid planner with buffered obstacle occupancy and segment-level collision rejection.\n");
    printf("Hierarchy: agent -> perception(target_encoder, obstacle_encoder) -> planner(fusion_head)\n");
    printf("Leaf graph: [target_encoder, obstacle_encoder] -> fusion_head\n");
    printf("Random seed: %u\n\n", rng_state);

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

    printf("Training for %d epochs with %u random scenes per epoch...\n\n",
        NESTED_NAV_EPOCHS,
        (unsigned int)NESTED_NAV_SCENES_PER_EPOCH);

    for (epoch = 0; epoch < NESTED_NAV_EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        size_t sample_count = 0U;
        size_t skipped_points = 0U;
        size_t scene_index;

        for (scene_index = 0U; scene_index < NESTED_NAV_SCENES_PER_EPOCH; ++scene_index) {
            NestedNavScene scene;
            int grid_y;

            if (generate_training_scene(&scene, &rng_state) != 0) {
                fprintf(stderr, "Failed to generate a reachable random training scene\n");
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                return 1;
            }

            for (grid_y = 0; grid_y < (int)NESTED_NAV_GRID_SIZE; ++grid_y) {
                int grid_x;
                float current_y = grid_to_world_value(grid_y);

                for (grid_x = 0; grid_x < (int)NESTED_NAV_GRID_SIZE; ++grid_x) {
                    float current_x = grid_to_world_value(grid_x);

                    if (is_blocked_world(&scene, current_x, current_y, 0.0f)) {
                        continue;
                    }

                    scene.current_x = current_x;
                    scene.current_y = current_y;
                    nested_nav_fill_input(input, &scene);

                    if (build_expected_action(&scene, expected) != 0) {
                        skipped_points += 1U;
                        continue;
                    }

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

                    epoch_loss += compute_loss(output, expected, NESTED_NAV_OUTPUT_SIZE);
                    sample_count += 1U;
                }
            }
        }

        sample_count_per_epoch = sample_count;
        skipped_points_per_epoch = skipped_points;

        if (sample_count == 0U) {
            fprintf(stderr, "No valid training samples were produced at epoch %d\n", epoch + 1);
            train_destroy(train_ctx);
            infer_destroy(infer_ctx);
            return 1;
        }

        if (((epoch + 1) % 5) == 0 || epoch == 0 || epoch == (NESTED_NAV_EPOCHS - 1)) {
            printf(
                "Epoch %d/%d - dataset loss: %.4f - trainer loss: %.4f - samples: %zu - skipped: %zu\n",
                epoch + 1,
                NESTED_NAV_EPOCHS,
                epoch_loss / (float)sample_count,
                train_get_loss(train_ctx),
                sample_count,
                skipped_points
            );
        }
    }

    printf("\n=== Training Complete ===\n");
    printf("Average loss: %.4f\n", train_get_loss(train_ctx));
    printf("Samples per epoch: %zu\n", sample_count_per_epoch);
    printf("Skipped unreachable points per epoch: %zu\n", skipped_points_per_epoch);

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
