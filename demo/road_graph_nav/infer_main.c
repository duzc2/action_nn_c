/**
 * @file infer_main.c
 * @brief road_graph_nav demo inference entry.
 */

#include "infer.h"
#include "weights_load.h"
#include "../demo_runtime_paths.h"
#include "road_graph_nav_scene.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#define ROAD_GRAPH_NAV_INPUT_SIZE (ROAD_GRAPH_NAV_NODE_COUNT * ROAD_GRAPH_NAV_NODE_FEATURE_SIZE)
#define ROAD_GRAPH_NAV_MAP_RASTER (ROAD_GRAPH_NAV_GRID_SIZE * 2U - 1U)

/**
 * @brief Pick the highest-scoring valid move from the four action scores.
 */
static int road_graph_nav_choose_best_action(
    const RoadGraphNavScene* scene,
    const float* output,
    size_t* out_action
) {
    float best_score = -1000000.0f;
    int found = 0;
    size_t action_index;

    for (action_index = 0U; action_index < ROAD_GRAPH_NAV_ACTION_COUNT; ++action_index) {
        if (road_graph_nav_next_node(scene, scene->current_node, action_index) >= 0 &&
            (!found || output[action_index] > best_score)) {
            best_score = output[action_index];
            *out_action = action_index;
            found = 1;
        }
    }

    return found ? 0 : -1;
}

/**
 * @brief Prefer the policy action, but fall back to the teacher action when the rollout stalls.
 */
static size_t road_graph_nav_choose_rollout_action(
    const RoadGraphNavScene* scene,
    const float* output,
    size_t teacher_action,
    const unsigned int* visit_counts,
    int* out_used_teacher
) {
    size_t chosen_action = teacher_action;
    size_t current_distance = 0U;
    int policy_next_node;
    int teacher_next_node;

    if (out_used_teacher != NULL) {
        *out_used_teacher = 0;
    }

    if (road_graph_nav_choose_best_action(scene, output, &chosen_action) != 0) {
        if (out_used_teacher != NULL) {
            *out_used_teacher = 1;
        }
        return teacher_action;
    }

    policy_next_node = road_graph_nav_next_node(scene, scene->current_node, chosen_action);
    teacher_next_node = road_graph_nav_next_node(scene, scene->current_node, teacher_action);
    if (policy_next_node < 0) {
        if (out_used_teacher != NULL) {
            *out_used_teacher = 1;
        }
        return teacher_action;
    }
    if (road_graph_nav_shortest_distance(scene, &current_distance) == 0) {
        RoadGraphNavScene policy_scene = *scene;
        size_t policy_distance = current_distance;

        policy_scene.current_node = (size_t)policy_next_node;
        if (road_graph_nav_shortest_distance(&policy_scene, &policy_distance) == 0) {
            int policy_stalls =
                (policy_distance >= current_distance) ||
                (visit_counts[(size_t)policy_next_node] >= 2U);

            if (policy_stalls && teacher_next_node >= 0) {
                if (out_used_teacher != NULL) {
                    *out_used_teacher = 1;
                }
                return teacher_action;
            }
        }
    }

    return chosen_action;
}

/**
 * @brief Rasterize the current road graph into a tiny ASCII map.
 */
static void road_graph_nav_render_map(
    const RoadGraphNavScene* scene,
    const unsigned int* visit_counts,
    size_t reached_goals,
    size_t step_index,
    const float* output,
    size_t chosen_action,
    size_t teacher_action
) {
    char map[ROAD_GRAPH_NAV_MAP_RASTER][ROAD_GRAPH_NAV_MAP_RASTER + 1U];
    size_t row;
    size_t column;
    size_t node_index;

    for (row = 0U; row < ROAD_GRAPH_NAV_MAP_RASTER; ++row) {
        for (column = 0U; column < ROAD_GRAPH_NAV_MAP_RASTER; ++column) {
            map[row][column] = ' ';
        }
        map[row][ROAD_GRAPH_NAV_MAP_RASTER] = '\0';
    }

    for (node_index = 0U; node_index < ROAD_GRAPH_NAV_NODE_COUNT; ++node_index) {
        size_t raster_row = road_graph_nav_row(node_index) * 2U;
        size_t raster_col = road_graph_nav_col(node_index) * 2U;
        char marker;

        if (!road_graph_nav_node_is_open(scene, node_index)) {
            marker = 'X';
        } else if (scene->current_node == node_index) {
            marker = 'C';
        } else if (scene->target_node == node_index) {
            marker = 'T';
        } else if (visit_counts[node_index] > 0U) {
            marker = 'o';
        } else {
            marker = '.';
        }
        map[raster_row][raster_col] = marker;

        if (road_graph_nav_node_is_open(scene, node_index)) {
            int right_neighbor = road_graph_nav_neighbor(node_index, ROAD_GRAPH_NAV_RIGHT);
            int down_neighbor = road_graph_nav_neighbor(node_index, ROAD_GRAPH_NAV_DOWN);

            if (right_neighbor >= 0 && road_graph_nav_node_is_open(scene, (size_t)right_neighbor)) {
                map[raster_row][raster_col + 1U] = '-';
            }
            if (down_neighbor >= 0 && road_graph_nav_node_is_open(scene, (size_t)down_neighbor)) {
                map[raster_row + 1U][raster_col] = '|';
            }
        }
    }

    printf(
        "\nStep %zu  Goals %zu/%u\n"
        "Map (.=open, X=blocked, C=current, T=target, o=trail)\n",
        step_index + 1U,
        reached_goals,
        ROAD_GRAPH_NAV_GOAL_COUNT
    );
    for (row = 0U; row < ROAD_GRAPH_NAV_MAP_RASTER; ++row) {
        printf("%s\n", map[row]);
    }
    printf(
        "Scores -> up=% .3f right=% .3f down=% .3f left=% .3f\n",
        output[ROAD_GRAPH_NAV_UP],
        output[ROAD_GRAPH_NAV_RIGHT],
        output[ROAD_GRAPH_NAV_DOWN],
        output[ROAD_GRAPH_NAV_LEFT]
    );
    printf(
        "Chosen action -> %s, teacher reference -> %s\n",
        g_road_graph_nav_action_names[chosen_action],
        g_road_graph_nav_action_names[teacher_action]
    );
    printf(
        "Current node=%zu target node=%zu blocked_mask=0x%016llx\n",
        scene->current_node,
        scene->target_node,
        (unsigned long long)scene->blocked_mask
    );
}

int main(void) {
    void* infer_ctx;
    const char* weights_file = "../data/weights.bin";
    RoadGraphNavScene scene;
    unsigned int rng_state = ((unsigned int)time(NULL)) ^ ((unsigned int)clock()) ^ 0x52474E56U;
    unsigned int visit_counts[ROAD_GRAPH_NAV_NODE_COUNT];
    float input[ROAD_GRAPH_NAV_INPUT_SIZE];
    float output[ROAD_GRAPH_NAV_ACTION_COUNT];
    size_t reached_goals = 0U;
    size_t step_index;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Road Graph Navigation Inference ===\n");
    printf("Scenario: a learned graph encoder proposes a direction on a blocked 8x8 road graph.\n");
    printf("Stop rule: reach %u goals or stop after %u steps.\n", ROAD_GRAPH_NAV_GOAL_COUNT, ROAD_GRAPH_NAV_STEP_CAP);
    printf("Random seed: %u\n", rng_state);

    road_graph_nav_reset_scene(&scene, &rng_state);
    (void)memset(visit_counts, 0, sizeof(visit_counts));
    visit_counts[scene.current_node] = 1U;

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

    for (step_index = 0U; step_index < ROAD_GRAPH_NAV_STEP_CAP && reached_goals < ROAD_GRAPH_NAV_GOAL_COUNT; ++step_index) {
        size_t chosen_action;
        size_t teacher_action = ROAD_GRAPH_NAV_UP;
        int used_teacher = 0;
        int next_node;

        road_graph_nav_fill_input(input, &scene);
        if (infer_auto_run(infer_ctx, input, output) != 0) {
            fprintf(stderr, "Inference failed during rollout\n");
            infer_destroy(infer_ctx);
            return 1;
        }
        if (road_graph_nav_teacher_action(&scene, &teacher_action) != 0) {
            fprintf(stderr, "Scene became unreachable during rollout\n");
            break;
        }
        chosen_action = road_graph_nav_choose_rollout_action(
            &scene,
            output,
            teacher_action,
            visit_counts,
            &used_teacher
        );

        road_graph_nav_render_map(
            &scene,
            visit_counts,
            reached_goals,
            step_index,
            output,
            chosen_action,
            teacher_action
        );
        if (used_teacher) {
            printf("Rollout controller: teacher action used to break a stall or loop.\n");
        }

        next_node = road_graph_nav_next_node(&scene, scene.current_node, chosen_action);
        if (next_node < 0) {
            fprintf(stderr, "Chosen action unexpectedly became invalid\n");
            break;
        }

        scene.current_node = (size_t)next_node;
        visit_counts[scene.current_node] += 1U;

        if (scene.current_node == scene.target_node) {
            reached_goals += 1U;
            printf("Reached goal %zu/%u at node %zu\n", reached_goals, ROAD_GRAPH_NAV_GOAL_COUNT, scene.current_node);
            if (reached_goals < ROAD_GRAPH_NAV_GOAL_COUNT) {
                road_graph_nav_choose_random_goal(&scene, &rng_state);
                (void)memset(visit_counts, 0, sizeof(visit_counts));
                visit_counts[scene.current_node] = 1U;
                printf("Sampling next target from current node %zu ...\n", scene.current_node);
            }
        }
    }

    if (reached_goals >= ROAD_GRAPH_NAV_GOAL_COUNT) {
        printf("Demo finished successfully after reaching %zu goals.\n", reached_goals);
    } else {
        printf(
            "Demo stopped after %zu step(s) with %zu/%u goals reached.\n",
            step_index,
            reached_goals,
            ROAD_GRAPH_NAV_GOAL_COUNT
        );
    }

    infer_destroy(infer_ctx);
    return 0;
}
