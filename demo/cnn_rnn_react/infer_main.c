/**
 * @file infer_main.c
 * @brief Inference entry for the CNN + RNN road-crossing controller demo.
 *
 * This version renders the same 30x20 full-map input used by training. The
 * rollout therefore shows exactly the world size the network is consuming, with
 * 4 fixed-direction lanes and visible car heads that make traffic direction
 * obvious to a human reader.
 */

#include "infer.h"
#include "weights_load.h"
#include "../demo_runtime_paths.h"
#include "cnn_rnn_react_scene.h"

#include <stdio.h>
#include <time.h>

/**
 * @brief Clamp an integer into a closed interval.
 */
static int clamp_int(int value, int lower, int upper) {
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

/**
 * @brief Shift the history window left and append the newest world frame.
 */
static void push_world_history(
    CnnRnnReactWorldState history[CNN_RNN_REACT_SEQUENCE_LENGTH],
    const CnnRnnReactWorldState* newest
) {
    size_t frame_index;

    for (frame_index = 1U; frame_index < CNN_RNN_REACT_SEQUENCE_LENGTH; ++frame_index) {
        history[frame_index - 1U] = history[frame_index];
    }
    history[CNN_RNN_REACT_SEQUENCE_LENGTH - 1U] = *newest;
}

/**
 * @brief Print one short traffic summary so readers know how each lane moves.
 */
static void print_lane_summary(const CnnRnnReactWorldState* world) {
    size_t lane_index;

    printf("Lane setup for this run:\n");
    for (lane_index = 0U; lane_index < CNN_RNN_REACT_LANE_COUNT; ++lane_index) {
        const CnnRnnReactVehicle* lane_vehicle =
            &world->vehicles[lane_index * CNN_RNN_REACT_MAX_CARS_PER_LANE];
        const char* direction = lane_vehicle->direction > 0 ? "left -> right" : "right -> left";

        printf(
            "  lane %zu rows %d-%d : %s, speed=%d cells/step, cars=%u\n",
            lane_index + 1U,
            cnn_rnn_react_lane_top_row(lane_index),
            cnn_rnn_react_lane_top_row(lane_index) + 1,
            direction,
            lane_vehicle->speed,
            (unsigned int)CNN_RNN_REACT_MAX_CARS_PER_LANE
        );
    }
    printf("\n");
}

/**
 * @brief Print the full ASCII world for the current rollout step.
 */
static void print_world(
    const CnnRnnReactWorldState* world,
    const int trail[][2],
    size_t trail_count,
    int predicted_row,
    int predicted_column,
    char predicted_marker,
    size_t step_index,
    const float output[CNN_RNN_REACT_OUTPUT_SIZE],
    int row_delta,
    int column_delta
) {
    char rows[CNN_RNN_REACT_WORLD_HEIGHT][CNN_RNN_REACT_WORLD_WIDTH + 1U];
    size_t row;

    cnn_rnn_react_render_world_ascii(
        world,
        trail,
        trail_count,
        predicted_row,
        predicted_column,
        predicted_marker,
        rows
    );

    printf("Step %zu\n", step_index + 1U);
    printf(
        "Legend: C=self  G=goal  #=car body  </>=car head  o=trail  ^=forward  L/R=predicted left/right  !=wait  ==lane divider\n"
    );
    for (row = 0U; row < CNN_RNN_REACT_WORLD_HEIGHT; ++row) {
        printf("%s\n", rows[row]);
    }
    printf(
        "Goal=(row=%d,col=%d)  Current=(row=%d,col=%d)  Predicted=(row=%d,col=%d)\n",
        world->goal_row,
        world->goal_column,
        world->ego_row,
        world->ego_column,
        predicted_row,
        predicted_column
    );
    printf(
        "Output: turn_axis=%.3f (%s), move_axis=%.3f (%s)\n",
        output[CNN_RNN_REACT_TURN_INDEX],
        cnn_rnn_react_describe_turn(output[CNN_RNN_REACT_TURN_INDEX]),
        output[CNN_RNN_REACT_MOVE_INDEX],
        cnn_rnn_react_describe_move(output[CNN_RNN_REACT_MOVE_INDEX])
    );
    printf(
        "Applied step: row_delta=%d, column_delta=%d\n\n",
        row_delta,
        column_delta
    );
}

int main(void) {
    const char* weights_file = "../../data/weights.bin";
    void* infer_ctx;
    unsigned int rng_state = ((unsigned int)time(NULL)) ^ 0x43524E4EU;
    CnnRnnReactWorldState history[CNN_RNN_REACT_SEQUENCE_LENGTH];
    float input[CNN_RNN_REACT_INPUT_SIZE];
    float output[CNN_RNN_REACT_OUTPUT_SIZE];
    int trail[CNN_RNN_REACT_MAX_ROLLOUT_STEPS + 1U][2];
    size_t trail_count = 0U;
    size_t frame_index;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== CNN + RNN Road-Crossing Inference ===\n");
    printf("Scenario: C starts on the lower sidewalk and tries to reach G on the upper sidewalk.\n");
    printf("Input and display both use the same 30x20 full map over 4 recent frames.\n");
    printf("Lane 1-2 move left to right, lane 3-4 move right to left, and car heads show direction.\n");
    printf("The network output is a reaction: left / center / right bias + wait / creep / go.\n\n");

    cnn_rnn_react_world_init_random(&history[0], &rng_state);
    for (frame_index = 1U; frame_index < CNN_RNN_REACT_SEQUENCE_LENGTH; ++frame_index) {
        history[frame_index] = history[frame_index - 1U];
        cnn_rnn_react_world_step(&history[frame_index], &rng_state);
    }

    printf("Random seed: %u\n", rng_state);
    print_lane_summary(&history[CNN_RNN_REACT_SEQUENCE_LENGTH - 1U]);

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

    trail[trail_count][0] = history[CNN_RNN_REACT_SEQUENCE_LENGTH - 1U].ego_row;
    trail[trail_count][1] = history[CNN_RNN_REACT_SEQUENCE_LENGTH - 1U].ego_column;
    trail_count += 1U;

    for (frame_index = 0U; frame_index < CNN_RNN_REACT_MAX_ROLLOUT_STEPS; ++frame_index) {
        CnnRnnReactWorldState current = history[CNN_RNN_REACT_SEQUENCE_LENGTH - 1U];
        int row_delta;
        int column_delta;
        int predicted_row;
        int predicted_column;
        char predicted_marker;

        cnn_rnn_react_build_input_from_history(history, input);
        if (infer_auto_run(infer_ctx, input, output) != 0) {
            fprintf(stderr, "inference failed during rollout\n");
            infer_destroy(infer_ctx);
            return 1;
        }

        cnn_rnn_react_choose_motion(output, frame_index, &row_delta, &column_delta, &predicted_marker);
        predicted_row = clamp_int(
            current.ego_row + row_delta,
            0,
            (int)CNN_RNN_REACT_WORLD_HEIGHT - 1
        );
        predicted_column = clamp_int(
            current.ego_column + column_delta,
            0,
            (int)CNN_RNN_REACT_WORLD_WIDTH - 1
        );

        print_world(
            &current,
            trail,
            trail_count,
            predicted_row,
            predicted_column,
            predicted_marker,
            frame_index,
            output,
            row_delta,
            column_delta
        );

        current.ego_row = predicted_row;
        current.ego_column = predicted_column;
        if (cnn_rnn_react_world_has_collision(&current)) {
            printf("Result: C stepped directly into a car cell. Rollout stops here.\n");
            break;
        }
        if (cnn_rnn_react_world_reached_goal(&current)) {
            printf("Result: C reached the opposite sidewalk near G. Rollout succeeded.\n");
            break;
        }

        cnn_rnn_react_world_step(&current, &rng_state);
        if (cnn_rnn_react_world_has_collision(&current)) {
            printf("Result: a moving car reached C after the step. Rollout stops here.\n");
            break;
        }

        if (trail_count < (CNN_RNN_REACT_MAX_ROLLOUT_STEPS + 1U)) {
            trail[trail_count][0] = current.ego_row;
            trail[trail_count][1] = current.ego_column;
            trail_count += 1U;
        }

        push_world_history(history, &current);

        if (cnn_rnn_react_world_reached_goal(&current)) {
            printf("Result: C reached the opposite sidewalk near G after traffic moved.\n");
            break;
        }
    }

    infer_destroy(infer_ctx);
    return 0;
}
