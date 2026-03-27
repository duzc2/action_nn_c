/**
 * @file cnn_rnn_react_scene.h
 * @brief Shared world-model helpers for the CNN + RNN road-crossing demo.
 *
 * The demo is intentionally concrete:
 * - C is a pedestrian / small robot that wants to cross upward;
 * - G is the single goal point on the opposite sidewalk;
 * - # are horizontally moving cars that stay inside their own lanes.
 *
 * The neural-network input is still a compact 8x8x4 observation history, but the
 * inference demo now renders a full 30x20 world so first-time users can see what
 * the controller is reacting to in plain scene terms instead of abstract tensors.
 */

#ifndef CNN_RNN_REACT_SCENE_H
#define CNN_RNN_REACT_SCENE_H

#include <stddef.h>

#define CNN_RNN_REACT_SEQUENCE_LENGTH 4U
#define CNN_RNN_REACT_FRAME_WIDTH 30U
#define CNN_RNN_REACT_FRAME_HEIGHT 20U
#define CNN_RNN_REACT_CHANNEL_COUNT 1U
#define CNN_RNN_REACT_CNN_FEATURE_SIZE 12U
#define CNN_RNN_REACT_CNN_OUTPUT_SIZE \
    (CNN_RNN_REACT_SEQUENCE_LENGTH * CNN_RNN_REACT_CNN_FEATURE_SIZE)
#define CNN_RNN_REACT_RNN_HIDDEN_SIZE 20U
#define CNN_RNN_REACT_INPUT_SIZE \
    (CNN_RNN_REACT_SEQUENCE_LENGTH * CNN_RNN_REACT_FRAME_WIDTH * CNN_RNN_REACT_FRAME_HEIGHT)
#define CNN_RNN_REACT_EGO_ROW (CNN_RNN_REACT_FRAME_HEIGHT - 2U)
#define CNN_RNN_REACT_EGO_COLUMN (CNN_RNN_REACT_FRAME_WIDTH / 2U)
#define CNN_RNN_REACT_OUTPUT_SIZE 2U
#define CNN_RNN_REACT_TURN_INDEX 0U
#define CNN_RNN_REACT_MOVE_INDEX 1U

#define CNN_RNN_REACT_WORLD_WIDTH 30U
#define CNN_RNN_REACT_WORLD_HEIGHT 20U
#define CNN_RNN_REACT_LANE_COUNT 4U
#define CNN_RNN_REACT_MAX_CARS_PER_LANE 2U
#define CNN_RNN_REACT_MAX_VEHICLES \
    (CNN_RNN_REACT_LANE_COUNT * CNN_RNN_REACT_MAX_CARS_PER_LANE)
#define CNN_RNN_REACT_MAX_ROLLOUT_STEPS 40U

/**
 * @brief One car moving horizontally inside a fixed lane.
 *
 * `left` uses world columns and may temporarily live outside the visible map so
 * wrap-around respawn can happen cleanly at lane edges.
 */
typedef struct {
    int lane_index;
    int left;
    int width;
    int speed;
    int direction;
} CnnRnnReactVehicle;

/**
 * @brief One full world snapshot used by both training and inference.
 *
 * The ego and goal are single cells. Cars are rectangles that span the two rows
 * of their lane and move only horizontally.
 */
typedef struct {
    int ego_row;
    int ego_column;
    int goal_row;
    int goal_column;
    CnnRnnReactVehicle vehicles[CNN_RNN_REACT_MAX_VEHICLES];
} CnnRnnReactWorldState;

/**
 * @brief One training sample made from four recent world snapshots.
 *
 * `history[0]` is the oldest frame and `history[3]` is the newest/current frame.
 * `input` is the full 30x20 world sequence consumed by the generated network,
 * and `target` is the teacher controller reaction for the newest frame.
 */
typedef struct {
    float input[CNN_RNN_REACT_INPUT_SIZE];
    float target[CNN_RNN_REACT_OUTPUT_SIZE];
    CnnRnnReactWorldState history[CNN_RNN_REACT_SEQUENCE_LENGTH];
} CnnRnnReactSample;

void cnn_rnn_react_build_sample(
    CnnRnnReactSample* sample,
    int target_column,
    int traffic_seed,
    int speed_bias,
    int density_bias
);
void cnn_rnn_react_build_random_sample(
    CnnRnnReactSample* sample,
    unsigned int* state
);
const char* cnn_rnn_react_describe_turn(float value);
const char* cnn_rnn_react_describe_move(float value);

void cnn_rnn_react_world_init_random(
    CnnRnnReactWorldState* world,
    unsigned int* state
);
void cnn_rnn_react_world_step(
    CnnRnnReactWorldState* world,
    unsigned int* state
);
void cnn_rnn_react_build_input_from_history(
    const CnnRnnReactWorldState history[CNN_RNN_REACT_SEQUENCE_LENGTH],
    float input[CNN_RNN_REACT_INPUT_SIZE]
);
void cnn_rnn_react_render_world_ascii(
    const CnnRnnReactWorldState* world,
    const int trail[][2],
    size_t trail_count,
    int predicted_row,
    int predicted_column,
    char predicted_marker,
    char rows[CNN_RNN_REACT_WORLD_HEIGHT][CNN_RNN_REACT_WORLD_WIDTH + 1U]
);
void cnn_rnn_react_choose_motion(
    const float output[CNN_RNN_REACT_OUTPUT_SIZE],
    size_t step_index,
    int* row_delta,
    int* column_delta,
    char* predicted_marker
);
int cnn_rnn_react_world_has_collision(const CnnRnnReactWorldState* world);
int cnn_rnn_react_world_reached_goal(const CnnRnnReactWorldState* world);
int cnn_rnn_react_lane_top_row(size_t lane_index);

#endif
