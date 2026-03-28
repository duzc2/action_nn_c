/**
 * @file road_graph_nav_scene.h
 * @brief Shared scene helpers for the road-graph navigation example.
 *
 * The example uses an 8x8 road graph so the GNN backend can reason over
 * explicit graph topology while an MLP head turns the graph embedding into
 * four movement scores.
 */

#ifndef ROAD_GRAPH_NAV_SCENE_H
#define ROAD_GRAPH_NAV_SCENE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define ROAD_GRAPH_NAV_GRID_SIZE 8U
#define ROAD_GRAPH_NAV_NODE_COUNT (ROAD_GRAPH_NAV_GRID_SIZE * ROAD_GRAPH_NAV_GRID_SIZE)
#define ROAD_GRAPH_NAV_NODE_FEATURE_SIZE 5U
#define ROAD_GRAPH_NAV_GNN_OUTPUT_SIZE 8U
#define ROAD_GRAPH_NAV_ACTION_COUNT 4U
#define ROAD_GRAPH_NAV_SLOT_COUNT 4U
#define ROAD_GRAPH_NAV_PATTERN_COUNT 12U
#define ROAD_GRAPH_NAV_START_NODE (((ROAD_GRAPH_NAV_GRID_SIZE / 2U) - 1U) * ROAD_GRAPH_NAV_GRID_SIZE + ((ROAD_GRAPH_NAV_GRID_SIZE / 2U) - 1U))
#define ROAD_GRAPH_NAV_GOAL_COUNT 8U
#define ROAD_GRAPH_NAV_STEP_CAP 720U
#define ROAD_GRAPH_NAV_MIN_GOAL_MANHATTAN 5U
#define ROAD_GRAPH_NAV_MIN_GOAL_PATH 8U
#define ROAD_GRAPH_NAV_MIN_GOAL_DETOUR 2U
#define ROAD_GRAPH_NAV_CANDIDATE_CAP (ROAD_GRAPH_NAV_PATTERN_COUNT * ROAD_GRAPH_NAV_NODE_COUNT)
#define ROAD_GRAPH_NAV_SCORE_BAND 64U
#define ROAD_GRAPH_NAV_FEATURE_OPEN 0U
#define ROAD_GRAPH_NAV_FEATURE_CURRENT 1U
#define ROAD_GRAPH_NAV_FEATURE_TARGET 2U
#define ROAD_GRAPH_NAV_FEATURE_X 3U
#define ROAD_GRAPH_NAV_FEATURE_Y 4U

typedef enum {
    ROAD_GRAPH_NAV_UP = 0,
    ROAD_GRAPH_NAV_RIGHT = 1,
    ROAD_GRAPH_NAV_DOWN = 2,
    ROAD_GRAPH_NAV_LEFT = 3
} RoadGraphNavAction;

typedef struct {
    uint64_t blocked_mask;
    size_t current_node;
    size_t target_node;
} RoadGraphNavScene;

typedef struct {
    RoadGraphNavScene scene;
    size_t score;
} RoadGraphNavCandidate;

static const char* g_road_graph_nav_action_names[ROAD_GRAPH_NAV_ACTION_COUNT] = {
    "up",
    "right",
    "down",
    "left"
};

/**
 * @brief Advance the shared deterministic RNG used by the example scene sampler.
 */
static inline unsigned int road_graph_nav_next_random(unsigned int* state) {
    *state = (*state * 1664525U) + 1013904223U;
    return *state;
}

/**
 * @brief Return the graph row for one compact node index.
 */
static inline size_t road_graph_nav_row(size_t node_index) {
    return node_index / ROAD_GRAPH_NAV_GRID_SIZE;
}

/**
 * @brief Return the graph column for one compact node index.
 */
static inline size_t road_graph_nav_col(size_t node_index) {
    return node_index % ROAD_GRAPH_NAV_GRID_SIZE;
}

/**
 * @brief Convert one grid coordinate back into the compact node index.
 */
static inline size_t road_graph_nav_node_index(size_t row, size_t col) {
    return (row * ROAD_GRAPH_NAV_GRID_SIZE) + col;
}

/**
 * @brief Convert a node coordinate into a normalized horizontal feature.
 */
static inline float road_graph_nav_x_norm(size_t node_index) {
    return ((float)road_graph_nav_col(node_index) / (float)(ROAD_GRAPH_NAV_GRID_SIZE - 1U)) * 2.0f - 1.0f;
}

/**
 * @brief Convert a node coordinate into a normalized vertical feature.
 */
static inline float road_graph_nav_y_norm(size_t node_index) {
    return 1.0f - (((float)road_graph_nav_row(node_index) / (float)(ROAD_GRAPH_NAV_GRID_SIZE - 1U)) * 2.0f);
}

/**
 * @brief Compute the Manhattan distance between two grid nodes.
 */
static inline size_t road_graph_nav_manhattan_distance(size_t from_node, size_t to_node) {
    size_t from_row = road_graph_nav_row(from_node);
    size_t from_col = road_graph_nav_col(from_node);
    size_t to_row = road_graph_nav_row(to_node);
    size_t to_col = road_graph_nav_col(to_node);
    size_t row_delta = from_row > to_row ? (from_row - to_row) : (to_row - from_row);
    size_t col_delta = from_col > to_col ? (from_col - to_col) : (to_col - from_col);

    return row_delta + col_delta;
}

/**
 * @brief Set one blocked bit in a 64-bit obstacle mask.
 */
static inline void road_graph_nav_block_cell(uint64_t* mask, size_t row, size_t col) {
    if (mask == NULL || row >= ROAD_GRAPH_NAV_GRID_SIZE || col >= ROAD_GRAPH_NAV_GRID_SIZE) {
        return;
    }
    *mask |= (uint64_t)1ULL << road_graph_nav_node_index(row, col);
}

/**
 * @brief Return the neighbor node reached by one directional action.
 */
static inline int road_graph_nav_neighbor(size_t node_index, size_t action_index) {
    size_t row = road_graph_nav_row(node_index);
    size_t col = road_graph_nav_col(node_index);

    if (node_index >= ROAD_GRAPH_NAV_NODE_COUNT || action_index >= ROAD_GRAPH_NAV_ACTION_COUNT) {
        return -1;
    }

    switch (action_index) {
        case ROAD_GRAPH_NAV_UP:
            return row > 0U ? (int)road_graph_nav_node_index(row - 1U, col) : -1;
        case ROAD_GRAPH_NAV_RIGHT:
            return col + 1U < ROAD_GRAPH_NAV_GRID_SIZE ? (int)road_graph_nav_node_index(row, col + 1U) : -1;
        case ROAD_GRAPH_NAV_DOWN:
            return row + 1U < ROAD_GRAPH_NAV_GRID_SIZE ? (int)road_graph_nav_node_index(row + 1U, col) : -1;
        case ROAD_GRAPH_NAV_LEFT:
            return col > 0U ? (int)road_graph_nav_node_index(row, col - 1U) : -1;
        default:
            return -1;
    }
}

/**
 * @brief Fill the fixed four-slot neighbor table consumed by the GNN backend.
 */
static inline void road_graph_nav_fill_neighbors(int neighbors[ROAD_GRAPH_NAV_NODE_COUNT][ROAD_GRAPH_NAV_SLOT_COUNT]) {
    size_t node_index;
    size_t action_index;

    for (node_index = 0U; node_index < ROAD_GRAPH_NAV_NODE_COUNT; ++node_index) {
        for (action_index = 0U; action_index < ROAD_GRAPH_NAV_SLOT_COUNT; ++action_index) {
            neighbors[node_index][action_index] = road_graph_nav_neighbor(node_index, action_index);
        }
    }
}

/**
 * @brief Build one deterministic obstacle template used by training and rollout.
 */
static inline uint64_t road_graph_nav_pattern_mask(size_t pattern_index) {
    uint64_t mask = 0ULL;
    size_t row;
    size_t col;

    switch (pattern_index % ROAD_GRAPH_NAV_PATTERN_COUNT) {
        case 0U:
            break;
        case 1U:
            for (row = 0U; row < ROAD_GRAPH_NAV_GRID_SIZE; ++row) {
                if (row != 1U) {
                    road_graph_nav_block_cell(&mask, row, 2U);
                }
            }
            break;
        case 2U:
            for (row = 0U; row < ROAD_GRAPH_NAV_GRID_SIZE; ++row) {
                if (row != ROAD_GRAPH_NAV_GRID_SIZE - 2U) {
                    road_graph_nav_block_cell(&mask, row, ROAD_GRAPH_NAV_GRID_SIZE - 3U);
                }
            }
            break;
        case 3U:
            for (col = 0U; col < ROAD_GRAPH_NAV_GRID_SIZE; ++col) {
                if (col != ROAD_GRAPH_NAV_GRID_SIZE - 3U) {
                    road_graph_nav_block_cell(&mask, 2U, col);
                }
            }
            break;
        case 4U:
            for (col = 0U; col < ROAD_GRAPH_NAV_GRID_SIZE; ++col) {
                if (col != 2U) {
                    road_graph_nav_block_cell(&mask, ROAD_GRAPH_NAV_GRID_SIZE - 3U, col);
                }
            }
            break;
        case 5U:
            for (row = 1U; row + 1U < ROAD_GRAPH_NAV_GRID_SIZE; ++row) {
                road_graph_nav_block_cell(&mask, row, (row + 1U) % ROAD_GRAPH_NAV_GRID_SIZE);
            }
            break;
        case 6U:
            for (row = 1U; row + 1U < ROAD_GRAPH_NAV_GRID_SIZE; ++row) {
                road_graph_nav_block_cell(&mask, row, (ROAD_GRAPH_NAV_GRID_SIZE - 2U - row));
            }
            break;
        case 7U:
            for (row = 1U; row <= 2U; ++row) {
                for (col = 1U; col <= 2U; ++col) {
                    road_graph_nav_block_cell(&mask, row, col);
                }
            }
            break;
        case 8U:
            for (row = ROAD_GRAPH_NAV_GRID_SIZE - 3U; row < ROAD_GRAPH_NAV_GRID_SIZE - 1U; ++row) {
                for (col = ROAD_GRAPH_NAV_GRID_SIZE - 3U; col < ROAD_GRAPH_NAV_GRID_SIZE - 1U; ++col) {
                    road_graph_nav_block_cell(&mask, row, col);
                }
            }
            break;
        case 9U:
            for (col = 1U; col + 1U < ROAD_GRAPH_NAV_GRID_SIZE; ++col) {
                if (col != 3U && col != 4U) {
                    road_graph_nav_block_cell(&mask, 3U, col);
                }
            }
            for (row = 1U; row + 1U < ROAD_GRAPH_NAV_GRID_SIZE; ++row) {
                if (row != 3U && row != 4U) {
                    road_graph_nav_block_cell(&mask, row, 4U);
                }
            }
            break;
        case 10U:
            for (col = 1U; col + 1U < ROAD_GRAPH_NAV_GRID_SIZE; ++col) {
                if (col != 3U) {
                    road_graph_nav_block_cell(&mask, 1U, col);
                }
                if (col != 4U) {
                    road_graph_nav_block_cell(&mask, ROAD_GRAPH_NAV_GRID_SIZE - 2U, col);
                }
            }
            break;
        case 11U:
            for (row = 2U; row + 2U < ROAD_GRAPH_NAV_GRID_SIZE; ++row) {
                if (row != 4U) {
                    road_graph_nav_block_cell(&mask, row, 1U);
                }
                if (row != 3U) {
                    road_graph_nav_block_cell(&mask, row, ROAD_GRAPH_NAV_GRID_SIZE - 2U);
                }
            }
            break;
        default:
            break;
    }

    mask &= ~((uint64_t)1ULL << ROAD_GRAPH_NAV_START_NODE);
    return mask;
}

/**
 * @brief Check whether one node is blocked by the current scenario mask.
 */
static inline int road_graph_nav_node_is_open(const RoadGraphNavScene* scene, size_t node_index) {
    return ((scene->blocked_mask >> node_index) & 1ULL) == 0ULL;
}

/**
 * @brief Fill the flattened graph feature tensor consumed by the GNN leaf.
 */
static inline void road_graph_nav_fill_input(float* input, const RoadGraphNavScene* scene) {
    size_t node_index;

    for (node_index = 0U; node_index < ROAD_GRAPH_NAV_NODE_COUNT; ++node_index) {
        size_t offset = node_index * ROAD_GRAPH_NAV_NODE_FEATURE_SIZE;

        input[offset + ROAD_GRAPH_NAV_FEATURE_OPEN] =
            road_graph_nav_node_is_open(scene, node_index) ? 1.0f : 0.0f;
        input[offset + ROAD_GRAPH_NAV_FEATURE_CURRENT] =
            scene->current_node == node_index ? 1.0f : 0.0f;
        input[offset + ROAD_GRAPH_NAV_FEATURE_TARGET] =
            scene->target_node == node_index ? 1.0f : 0.0f;
        input[offset + ROAD_GRAPH_NAV_FEATURE_X] = road_graph_nav_x_norm(node_index);
        input[offset + ROAD_GRAPH_NAV_FEATURE_Y] = road_graph_nav_y_norm(node_index);
    }
}

/**
 * @brief Return the neighbor node reached by one directional action.
 */
static inline int road_graph_nav_next_node(const RoadGraphNavScene* scene, size_t node_index, size_t action_index) {
    int neighbor;

    if (action_index >= ROAD_GRAPH_NAV_ACTION_COUNT) {
        return -1;
    }
    neighbor = road_graph_nav_neighbor(node_index, action_index);
    if (neighbor < 0) {
        return -1;
    }
    if (!road_graph_nav_node_is_open(scene, (size_t)neighbor)) {
        return -1;
    }
    return neighbor;
}

/**
 * @brief Solve the shortest path teacher action with a tiny BFS.
 */
static inline int road_graph_nav_teacher_action(const RoadGraphNavScene* scene, size_t* out_action) {
    size_t queue[ROAD_GRAPH_NAV_NODE_COUNT];
    size_t parent[ROAD_GRAPH_NAV_NODE_COUNT];
    size_t parent_action[ROAD_GRAPH_NAV_NODE_COUNT];
    unsigned char visited[ROAD_GRAPH_NAV_NODE_COUNT];
    size_t head = 0U;
    size_t tail = 0U;
    size_t node_index;

    if (!road_graph_nav_node_is_open(scene, scene->current_node) ||
        !road_graph_nav_node_is_open(scene, scene->target_node)) {
        return -1;
    }
    if (scene->current_node == scene->target_node) {
        if (out_action != NULL) {
            *out_action = ROAD_GRAPH_NAV_UP;
        }
        return 0;
    }

    (void)memset(visited, 0, sizeof(visited));
    for (node_index = 0U; node_index < ROAD_GRAPH_NAV_NODE_COUNT; ++node_index) {
        parent[node_index] = ROAD_GRAPH_NAV_NODE_COUNT;
        parent_action[node_index] = ROAD_GRAPH_NAV_ACTION_COUNT;
    }

    visited[scene->current_node] = 1U;
    queue[tail] = scene->current_node;
    tail += 1U;

    while (head < tail) {
        size_t current = queue[head];
        size_t action_index;

        head += 1U;
        if (current == scene->target_node) {
            break;
        }

        for (action_index = 0U; action_index < ROAD_GRAPH_NAV_ACTION_COUNT; ++action_index) {
            int neighbor = road_graph_nav_next_node(scene, current, action_index);

            if (neighbor >= 0 && !visited[(size_t)neighbor]) {
                visited[(size_t)neighbor] = 1U;
                parent[(size_t)neighbor] = current;
                parent_action[(size_t)neighbor] = action_index;
                queue[tail] = (size_t)neighbor;
                tail += 1U;
            }
        }
    }

    if (!visited[scene->target_node]) {
        return -1;
    }

    node_index = scene->target_node;
    while (parent[node_index] != scene->current_node) {
        if (parent[node_index] >= ROAD_GRAPH_NAV_NODE_COUNT) {
            return -1;
        }
        node_index = parent[node_index];
    }

    if (out_action != NULL) {
        *out_action = parent_action[node_index];
    }
    return 0;
}

/**
 * @brief Compute the shortest reachable path length between current and target.
 */
static inline int road_graph_nav_shortest_distance(const RoadGraphNavScene* scene, size_t* out_distance) {
    size_t queue[ROAD_GRAPH_NAV_NODE_COUNT];
    size_t distance[ROAD_GRAPH_NAV_NODE_COUNT];
    unsigned char visited[ROAD_GRAPH_NAV_NODE_COUNT];
    size_t head = 0U;
    size_t tail = 0U;
    size_t node_index;

    if (!road_graph_nav_node_is_open(scene, scene->current_node) ||
        !road_graph_nav_node_is_open(scene, scene->target_node)) {
        return -1;
    }
    if (scene->current_node == scene->target_node) {
        if (out_distance != NULL) {
            *out_distance = 0U;
        }
        return 0;
    }

    (void)memset(visited, 0, sizeof(visited));
    for (node_index = 0U; node_index < ROAD_GRAPH_NAV_NODE_COUNT; ++node_index) {
        distance[node_index] = 0U;
    }

    visited[scene->current_node] = 1U;
    queue[tail] = scene->current_node;
    tail += 1U;

    while (head < tail) {
        size_t current = queue[head];
        size_t action_index;

        head += 1U;
        if (current == scene->target_node) {
            if (out_distance != NULL) {
                *out_distance = distance[current];
            }
            return 0;
        }

        for (action_index = 0U; action_index < ROAD_GRAPH_NAV_ACTION_COUNT; ++action_index) {
            int neighbor = road_graph_nav_next_node(scene, current, action_index);

            if (neighbor >= 0 && !visited[(size_t)neighbor]) {
                visited[(size_t)neighbor] = 1U;
                distance[(size_t)neighbor] = distance[current] + 1U;
                queue[tail] = (size_t)neighbor;
                tail += 1U;
            }
        }
    }

    return -1;
}

/**
 * @brief Choose a reachable goal that strongly prefers long detours over direct paths.
 */
static inline void road_graph_nav_choose_random_goal(RoadGraphNavScene* scene, unsigned int* state) {
    RoadGraphNavCandidate detour_candidates[ROAD_GRAPH_NAV_CANDIDATE_CAP];
    RoadGraphNavCandidate fallback_candidates[ROAD_GRAPH_NAV_CANDIDATE_CAP];
    size_t detour_count = 0U;
    size_t fallback_count = 0U;
    size_t best_detour_score = 0U;
    size_t best_fallback_score = 0U;
    size_t pattern_start;
    size_t pattern_offset;

    pattern_start = (size_t)(road_graph_nav_next_random(state) % ROAD_GRAPH_NAV_PATTERN_COUNT);

    for (pattern_offset = 0U; pattern_offset < ROAD_GRAPH_NAV_PATTERN_COUNT; ++pattern_offset) {
        size_t pattern_index = (pattern_start + pattern_offset) % ROAD_GRAPH_NAV_PATTERN_COUNT;
        RoadGraphNavScene candidate = *scene;
        size_t target_node;

        candidate.blocked_mask = road_graph_nav_pattern_mask(pattern_index);
        if (!road_graph_nav_node_is_open(&candidate, candidate.current_node)) {
            continue;
        }

        for (target_node = 0U; target_node < ROAD_GRAPH_NAV_NODE_COUNT; ++target_node) {
            size_t manhattan;
            size_t shortest_path;
            size_t detour;
            size_t score;

            if (target_node == candidate.current_node) {
                continue;
            }
            if (!road_graph_nav_node_is_open(&candidate, target_node)) {
                continue;
            }

            candidate.target_node = target_node;
            if (road_graph_nav_shortest_distance(&candidate, &shortest_path) != 0) {
                continue;
            }

            manhattan = road_graph_nav_manhattan_distance(candidate.current_node, target_node);
            detour = shortest_path > manhattan ? (shortest_path - manhattan) : 0U;
            score = (detour * 256U) + (shortest_path * 16U) + manhattan;

            if (detour >= ROAD_GRAPH_NAV_MIN_GOAL_DETOUR &&
                shortest_path >= ROAD_GRAPH_NAV_MIN_GOAL_PATH &&
                manhattan >= ROAD_GRAPH_NAV_MIN_GOAL_MANHATTAN) {
                if (detour_count < ROAD_GRAPH_NAV_CANDIDATE_CAP) {
                    detour_candidates[detour_count].scene = candidate;
                    detour_candidates[detour_count].score = score;
                    detour_count += 1U;
                }
                if (score > best_detour_score) {
                    best_detour_score = score;
                }
            } else {
                if (fallback_count < ROAD_GRAPH_NAV_CANDIDATE_CAP) {
                    fallback_candidates[fallback_count].scene = candidate;
                    fallback_candidates[fallback_count].score = score;
                    fallback_count += 1U;
                }
                if (score > best_fallback_score) {
                    best_fallback_score = score;
                }
            }
        }
    }

    if (detour_count > 0U) {
        size_t eligible_count = 0U;
        size_t candidate_index;

        for (candidate_index = 0U; candidate_index < detour_count; ++candidate_index) {
            if (detour_candidates[candidate_index].score + ROAD_GRAPH_NAV_SCORE_BAND >= best_detour_score) {
                detour_candidates[eligible_count] = detour_candidates[candidate_index];
                eligible_count += 1U;
            }
        }

        *scene = detour_candidates[
            (size_t)(road_graph_nav_next_random(state) % (unsigned int)eligible_count)
        ].scene;
        return;
    }
    if (fallback_count > 0U) {
        size_t eligible_count = 0U;
        size_t candidate_index;

        for (candidate_index = 0U; candidate_index < fallback_count; ++candidate_index) {
            if (fallback_candidates[candidate_index].score + ROAD_GRAPH_NAV_SCORE_BAND >= best_fallback_score) {
                fallback_candidates[eligible_count] = fallback_candidates[candidate_index];
                eligible_count += 1U;
            }
        }

        *scene = fallback_candidates[
            (size_t)(road_graph_nav_next_random(state) % (unsigned int)eligible_count)
        ].scene;
        return;
    }

    scene->blocked_mask = 0ULL;
    scene->target_node = (scene->current_node + ROAD_GRAPH_NAV_GRID_SIZE) % ROAD_GRAPH_NAV_NODE_COUNT;
    if (scene->target_node == scene->current_node) {
        scene->target_node = (scene->current_node + 1U) % ROAD_GRAPH_NAV_NODE_COUNT;
    }
}

/**
 * @brief Reset the example scene to the standard near-center start node and sample a goal.
 */
static inline void road_graph_nav_reset_scene(RoadGraphNavScene* scene, unsigned int* state) {
    scene->blocked_mask = 0ULL;
    scene->current_node = ROAD_GRAPH_NAV_START_NODE;
    scene->target_node = ROAD_GRAPH_NAV_START_NODE;
    road_graph_nav_choose_random_goal(scene, state);
}

#endif
