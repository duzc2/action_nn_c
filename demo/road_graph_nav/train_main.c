/**
 * @file train_main.c
 * @brief road_graph_nav demo training entry.
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "../demo_runtime_paths.h"
#include "road_graph_nav_scene.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define ROAD_GRAPH_NAV_EPOCHS 48U
#define ROAD_GRAPH_NAV_SAMPLES_PER_EPOCH 4096U
#define ROAD_GRAPH_NAV_EVAL_SAMPLES 256U
#define ROAD_GRAPH_NAV_INPUT_SIZE (ROAD_GRAPH_NAV_NODE_COUNT * ROAD_GRAPH_NAV_NODE_FEATURE_SIZE)

typedef struct {
    uint64_t blocked_mask;
    unsigned char current_node;
    unsigned char target_node;
    unsigned char action;
    unsigned char detour_priority;
} RoadGraphNavSample;

/**
 * @brief Mean squared error for logging and quick sanity checks.
 */
static float road_graph_nav_compute_loss(const float* output, const float* expected) {
    float loss = 0.0f;
    size_t index;

    for (index = 0U; index < ROAD_GRAPH_NAV_ACTION_COUNT; ++index) {
        float diff = output[index] - expected[index];
        loss += diff * diff;
    }
    return loss / (float)ROAD_GRAPH_NAV_ACTION_COUNT;
}

/**
 * @brief Build one compact supervised sample table once, then reuse it across epochs.
 */
static int road_graph_nav_build_dataset(
    RoadGraphNavSample* samples,
    size_t capacity,
    size_t* out_count,
    size_t* out_detour_count
) {
    size_t sample_count = 0U;
    size_t detour_count = 0U;
    size_t pattern_index;

    if (samples == NULL || out_count == NULL || out_detour_count == NULL) {
        return -1;
    }

    for (pattern_index = 0U; pattern_index < ROAD_GRAPH_NAV_PATTERN_COUNT; ++pattern_index) {
        size_t current_node;

        for (current_node = 0U; current_node < ROAD_GRAPH_NAV_NODE_COUNT; ++current_node) {
            size_t target_node;

            for (target_node = 0U; target_node < ROAD_GRAPH_NAV_NODE_COUNT; ++target_node) {
                RoadGraphNavScene scene;
                size_t action_index = 0U;
                size_t shortest_path = 0U;
                size_t manhattan = 0U;
                size_t detour = 0U;

                if (current_node == target_node) {
                    continue;
                }

                scene.blocked_mask = road_graph_nav_pattern_mask(pattern_index);
                scene.current_node = current_node;
                scene.target_node = target_node;

                if (!road_graph_nav_node_is_open(&scene, current_node) ||
                    !road_graph_nav_node_is_open(&scene, target_node)) {
                    continue;
                }
                if (road_graph_nav_shortest_distance(&scene, &shortest_path) != 0) {
                    continue;
                }
                if (road_graph_nav_teacher_action(&scene, &action_index) != 0) {
                    continue;
                }
                if (sample_count >= capacity) {
                    return -1;
                }

                manhattan = road_graph_nav_manhattan_distance(current_node, target_node);
                detour = shortest_path > manhattan ? (shortest_path - manhattan) : 0U;
                samples[sample_count].blocked_mask = scene.blocked_mask;
                samples[sample_count].current_node = (unsigned char)current_node;
                samples[sample_count].target_node = (unsigned char)target_node;
                samples[sample_count].action = (unsigned char)action_index;
                samples[sample_count].detour_priority =
                    (unsigned char)((detour >= ROAD_GRAPH_NAV_MIN_GOAL_DETOUR &&
                                     shortest_path >= ROAD_GRAPH_NAV_MIN_GOAL_PATH &&
                                     manhattan >= ROAD_GRAPH_NAV_MIN_GOAL_MANHATTAN) ? 1U : 0U);
                if (samples[sample_count].detour_priority != 0U) {
                    detour_count += 1U;
                }
                sample_count += 1U;
            }
        }
    }

    *out_count = sample_count;
    *out_detour_count = detour_count;
    return 0;
}

/**
 * @brief Rehydrate one compact dataset sample into the scene/input/target tensors.
 */
static void road_graph_nav_materialize_sample(
    const RoadGraphNavSample* sample,
    RoadGraphNavScene* scene,
    float* input,
    float* expected
) {
    size_t action_index;

    scene->blocked_mask = sample->blocked_mask;
    scene->current_node = (size_t)sample->current_node;
    scene->target_node = (size_t)sample->target_node;

    road_graph_nav_fill_input(input, scene);
    (void)memset(expected, 0, ROAD_GRAPH_NAV_ACTION_COUNT * sizeof(float));
    action_index = (size_t)sample->action;
    if (action_index < ROAD_GRAPH_NAV_ACTION_COUNT) {
        expected[action_index] = 1.0f;
    }
}

/**
 * @brief Evaluate a small random subset so logging stays representative but cheap.
 */
static float road_graph_nav_evaluate_subset(
    void* infer_ctx,
    const RoadGraphNavSample* samples,
    size_t sample_count,
    unsigned int* rng_state,
    size_t eval_count
) {
    RoadGraphNavScene scene;
    float input[ROAD_GRAPH_NAV_INPUT_SIZE];
    float expected[ROAD_GRAPH_NAV_ACTION_COUNT];
    float output[ROAD_GRAPH_NAV_ACTION_COUNT];
    float total_loss = 0.0f;
    size_t eval_index;

    if (infer_ctx == NULL || samples == NULL || sample_count == 0U || rng_state == NULL || eval_count == 0U) {
        return -1.0f;
    }

    for (eval_index = 0U; eval_index < eval_count; ++eval_index) {
        size_t sample_index = (size_t)(road_graph_nav_next_random(rng_state) % (unsigned int)sample_count);

        road_graph_nav_materialize_sample(&samples[sample_index], &scene, input, expected);
        if (infer_auto_run(infer_ctx, input, output) != 0) {
            return -1.0f;
        }
        total_loss += road_graph_nav_compute_loss(output, expected);
    }

    return total_loss / (float)eval_count;
}

int main(void) {
    const char* output_file = "../data/weights.bin";
    void* infer_ctx;
    void* train_ctx;
    RoadGraphNavSample* dataset = NULL;
    size_t* detour_indices = NULL;
    size_t dataset_capacity;
    size_t dataset_count = 0U;
    size_t detour_count = 0U;
    unsigned int rng_state = 0x52474E54U;
    RoadGraphNavScene scene;
    float input[ROAD_GRAPH_NAV_INPUT_SIZE];
    float expected[ROAD_GRAPH_NAV_ACTION_COUNT];
    size_t epoch_index;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Road Graph Navigation Training ===\n");
    printf("Scene: 8x8 road graph, deterministic obstacle templates, shortest-path teacher.\n");
    printf("Leaf graph: gnn(graph_encoder) -> mlp(decision_head)\n\n");

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

    dataset_capacity =
        ROAD_GRAPH_NAV_PATTERN_COUNT * ROAD_GRAPH_NAV_NODE_COUNT * (ROAD_GRAPH_NAV_NODE_COUNT - 1U);
    dataset = (RoadGraphNavSample*)calloc(dataset_capacity, sizeof(RoadGraphNavSample));
    if (dataset == NULL) {
        fprintf(stderr, "Failed to allocate compact training dataset\n");
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        return 1;
    }
    if (road_graph_nav_build_dataset(dataset, dataset_capacity, &dataset_count, &detour_count) != 0 ||
        dataset_count == 0U) {
        fprintf(stderr, "Failed to build compact training dataset\n");
        free(dataset);
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        return 1;
    }
    detour_indices = (size_t*)calloc(dataset_count, sizeof(size_t));
    if (detour_indices == NULL) {
        fprintf(stderr, "Failed to allocate detour sample index buffer\n");
        free(dataset);
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        return 1;
    }
    if (detour_count > 0U) {
        size_t source_index;
        size_t detour_index = 0U;

        for (source_index = 0U; source_index < dataset_count; ++source_index) {
            if (dataset[source_index].detour_priority != 0U) {
                detour_indices[detour_index] = source_index;
                detour_index += 1U;
            }
        }
    }

    printf("Reachable samples: %zu\n", dataset_count);
    printf("Detour-priority samples: %zu\n", detour_count);
    printf("Updates per epoch: %u\n", ROAD_GRAPH_NAV_SAMPLES_PER_EPOCH);
    printf("Eval samples per report: %u\n\n", ROAD_GRAPH_NAV_EVAL_SAMPLES);

    for (epoch_index = 0U; epoch_index < ROAD_GRAPH_NAV_EPOCHS; ++epoch_index) {
        size_t sample_index;

        for (sample_index = 0U; sample_index < ROAD_GRAPH_NAV_SAMPLES_PER_EPOCH; ++sample_index) {
            size_t picked_index;
            unsigned int random_value = road_graph_nav_next_random(&rng_state);

            if (detour_count > 0U && (random_value % 100U) < 70U) {
                size_t detour_pick = (size_t)(road_graph_nav_next_random(&rng_state) % (unsigned int)detour_count);
                picked_index = detour_indices[detour_pick];
            } else {
                picked_index = (size_t)(road_graph_nav_next_random(&rng_state) % (unsigned int)dataset_count);
            }

            road_graph_nav_materialize_sample(&dataset[picked_index], &scene, input, expected);
            if (train_step(train_ctx, input, expected) != 0) {
                fprintf(stderr, "Training step failed at epoch %zu\n", epoch_index + 1U);
                free(detour_indices);
                free(dataset);
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                return 1;
            }
        }

        if (epoch_index == 0U ||
            ((epoch_index + 1U) % 8U) == 0U ||
            epoch_index + 1U == ROAD_GRAPH_NAV_EPOCHS) {
            float eval_loss = road_graph_nav_evaluate_subset(
                infer_ctx,
                dataset,
                dataset_count,
                &rng_state,
                ROAD_GRAPH_NAV_EVAL_SAMPLES
            );

            if (eval_loss < 0.0f) {
                fprintf(stderr, "Inference check failed at epoch %zu\n", epoch_index + 1U);
                free(detour_indices);
                free(dataset);
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                return 1;
            }
            printf(
                "Epoch %zu/%u - eval loss: %.4f - trainer loss: %.4f\n",
                epoch_index + 1U,
                ROAD_GRAPH_NAV_EPOCHS,
                eval_loss,
                train_get_loss(train_ctx)
            );
        }
    }

    printf("\nTraining completed.\n");
    printf("Average loss: %.4f\n", train_get_loss(train_ctx));
    if (weights_save_to_file(infer_ctx, output_file) != 0) {
        fprintf(stderr, "Failed to save weights to %s\n", output_file);
        free(detour_indices);
        free(dataset);
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        return 1;
    }
    printf("Weights saved to: %s\n", output_file);

    free(detour_indices);
    free(dataset);
    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    return 0;
}
