/**
 * @file nested_nav_scene.h
 * @brief Shared scene helpers for nested navigation demo
 */

#ifndef NESTED_NAV_SCENE_H
#define NESTED_NAV_SCENE_H

#include <math.h>
#include <stddef.h>

#define NESTED_NAV_INPUT_SIZE 10U
#define NESTED_NAV_OUTPUT_SIZE 2U
#define NESTED_NAV_OBSTACLE_COUNT 3U
#define NESTED_NAV_WORLD_MIN (-1.0f)
#define NESTED_NAV_WORLD_MAX (1.0f)

typedef struct {
    float current_x;
    float current_y;
    float target_x;
    float target_y;
    float obstacle_centers[NESTED_NAV_OBSTACLE_COUNT][2];
} NestedNavScene;

static const float g_nested_nav_start_position[2] = {-0.85f, -0.85f};
static const float g_nested_nav_target_position[2] = {0.85f, 0.85f};
static const float g_nested_nav_obstacle_radii[NESTED_NAV_OBSTACLE_COUNT] = {0.20f, 0.18f, 0.18f};
static const float g_nested_nav_obstacle_anchors[NESTED_NAV_OBSTACLE_COUNT][2] = {
    {-0.20f, -0.05f},
    {0.15f, 0.20f},
    {0.48f, 0.52f}
};

static float nested_nav_distance(float ax, float ay, float bx, float by) {
    float dx = ax - bx;
    float dy = ay - by;
    return sqrtf((dx * dx) + (dy * dy));
}

static float nested_nav_rand01(unsigned int* state) {
    *state = (*state * 1664525U) + 1013904223U;
    return (float)((*state >> 8) & 0x00FFFFFFU) / 16777215.0f;
}

static float nested_nav_rand_symmetric(unsigned int* state, float amplitude) {
    return (nested_nav_rand01(state) * 2.0f * amplitude) - amplitude;
}

static void nested_nav_fill_input(float* input, const NestedNavScene* scene) {
    size_t obstacle_index;

    input[0] = scene->current_x;
    input[1] = scene->current_y;
    input[2] = scene->target_x;
    input[3] = scene->target_y;
    for (obstacle_index = 0U; obstacle_index < NESTED_NAV_OBSTACLE_COUNT; ++obstacle_index) {
        input[4U + obstacle_index * 2U] = scene->obstacle_centers[obstacle_index][0];
        input[5U + obstacle_index * 2U] = scene->obstacle_centers[obstacle_index][1];
    }
}

static int nested_nav_segment_hits_obstacle(
    float start_x,
    float start_y,
    float end_x,
    float end_y,
    float obstacle_x,
    float obstacle_y,
    float obstacle_radius
) {
    float seg_x = end_x - start_x;
    float seg_y = end_y - start_y;
    float seg_len_sq = (seg_x * seg_x) + (seg_y * seg_y);
    float t;
    float closest_x;
    float closest_y;

    if (seg_len_sq < 0.0000001f) {
        return nested_nav_distance(start_x, start_y, obstacle_x, obstacle_y) <= obstacle_radius;
    }

    t = (((obstacle_x - start_x) * seg_x) + ((obstacle_y - start_y) * seg_y)) / seg_len_sq;
    if (t < 0.0f) {
        t = 0.0f;
    }
    if (t > 1.0f) {
        t = 1.0f;
    }

    closest_x = start_x + (seg_x * t);
    closest_y = start_y + (seg_y * t);
    return nested_nav_distance(closest_x, closest_y, obstacle_x, obstacle_y) <= obstacle_radius;
}

static int nested_nav_scene_valid(const NestedNavScene* scene, float extra_clearance) {
    size_t left_index;

    for (left_index = 0U; left_index < NESTED_NAV_OBSTACLE_COUNT; ++left_index) {
        size_t right_index;
        float left_x = scene->obstacle_centers[left_index][0];
        float left_y = scene->obstacle_centers[left_index][1];
        float left_radius = g_nested_nav_obstacle_radii[left_index] + extra_clearance;

        if (nested_nav_distance(
                left_x,
                left_y,
                g_nested_nav_start_position[0],
                g_nested_nav_start_position[1]) <= (left_radius + 0.18f)) {
            return 0;
        }
        if (nested_nav_distance(
                left_x,
                left_y,
                g_nested_nav_target_position[0],
                g_nested_nav_target_position[1]) <= (left_radius + 0.18f)) {
            return 0;
        }

        for (right_index = left_index + 1U; right_index < NESTED_NAV_OBSTACLE_COUNT; ++right_index) {
            float right_x = scene->obstacle_centers[right_index][0];
            float right_y = scene->obstacle_centers[right_index][1];
            float min_distance = left_radius + g_nested_nav_obstacle_radii[right_index] + extra_clearance + 0.06f;

            if (nested_nav_distance(left_x, left_y, right_x, right_y) <= min_distance) {
                return 0;
            }
        }
    }

    return 1;
}

static void nested_nav_generate_random_scene(NestedNavScene* scene, unsigned int* state) {
    size_t attempt;

    scene->current_x = g_nested_nav_start_position[0];
    scene->current_y = g_nested_nav_start_position[1];
    scene->target_x = g_nested_nav_target_position[0];
    scene->target_y = g_nested_nav_target_position[1];

    for (attempt = 0U; attempt < 256U; ++attempt) {
        size_t obstacle_index;
        int diagonal_hits = 0;

        for (obstacle_index = 0U; obstacle_index < NESTED_NAV_OBSTACLE_COUNT; ++obstacle_index) {
            scene->obstacle_centers[obstacle_index][0] =
                g_nested_nav_obstacle_anchors[obstacle_index][0] +
                nested_nav_rand_symmetric(state, 0.10f);
            scene->obstacle_centers[obstacle_index][1] =
                g_nested_nav_obstacle_anchors[obstacle_index][1] +
                nested_nav_rand_symmetric(state, 0.10f);
        }

        if (!nested_nav_scene_valid(scene, 0.0f)) {
            continue;
        }

        for (obstacle_index = 0U; obstacle_index < NESTED_NAV_OBSTACLE_COUNT; ++obstacle_index) {
            if (nested_nav_segment_hits_obstacle(
                    g_nested_nav_start_position[0],
                    g_nested_nav_start_position[1],
                    g_nested_nav_target_position[0],
                    g_nested_nav_target_position[1],
                    scene->obstacle_centers[obstacle_index][0],
                    scene->obstacle_centers[obstacle_index][1],
                    g_nested_nav_obstacle_radii[obstacle_index] + 0.05f)) {
                diagonal_hits += 1;
            }
        }

        if (diagonal_hits >= 2) {
            return;
        }
    }

    for (attempt = 0U; attempt < NESTED_NAV_OBSTACLE_COUNT; ++attempt) {
        scene->obstacle_centers[attempt][0] = g_nested_nav_obstacle_anchors[attempt][0];
        scene->obstacle_centers[attempt][1] = g_nested_nav_obstacle_anchors[attempt][1];
    }
}

#endif
