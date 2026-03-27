/**
 * @file cnn_rnn_react_scene.c
 * @brief World generator and full-map observation builder for the road-crossing demo.
 *
 * This demo now uses one unified scene definition for both training and inference:
 * - every input frame is the full 30x20 world, not a cropped local patch;
 * - there are 4 fixed traffic lanes;
 * - lane 1 and lane 2 move from left to right;
 * - lane 3 and lane 4 move from right to left;
 * - every car has a visible head marker so humans can read its direction directly.
 *
 * The teacher controller is also aligned with what the network can really see:
 * it evaluates routes using the same visible world and gives extra weight to the
 * lane directly in front of the ego because that lane decides immediate survival.
 */

#include "cnn_rnn_react_scene.h"

#include <stdlib.h>
#include <string.h>

#define CNN_RNN_REACT_GOAL_ROW 1
#define CNN_RNN_REACT_START_ROW 18
#define CNN_RNN_REACT_START_COLUMN 15

/**
 * @brief Candidate action score used by the hand-written teacher.
 */
typedef struct {
    int row_delta;
    int column_delta;
    int next_row;
    int next_column;
    float immediate_danger;
    float front_lane_danger;
    float route_danger;
    float goal_alignment;
    float score;
} CnnRnnReactCandidate;

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
 * @brief Clamp a float into the normalized controller interval.
 */
static float clamp_unit(float value) {
    if (value < -1.0f) {
        return -1.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

/**
 * @brief Advance the deterministic RNG used by the demo.
 */
static unsigned int react_next_random(unsigned int* state) {
    unsigned int value = *state;

    value = (value * 1664525U) + 1013904223U;
    *state = value;
    return value;
}

/**
 * @brief Produce an integer inside the inclusive interval [lower, upper].
 */
static int react_random_range(unsigned int* state, int lower, int upper) {
    unsigned int span = (unsigned int)(upper - lower + 1);
    return lower + (int)(react_next_random(state) % span);
}

/**
 * @brief Return the fixed traffic direction for one lane.
 *
 * User requirement:
 * - lane 1 and lane 2: left -> right
 * - lane 3 and lane 4: right -> left
 */
static int react_lane_direction(size_t lane_index) {
    return lane_index < 2U ? 1 : -1;
}

/**
 * @brief Return the top row of one horizontal traffic lane.
 */
int cnn_rnn_react_lane_top_row(size_t lane_index) {
    return 3 + ((int)lane_index * 3);
}

/**
 * @brief Return non-zero when the row is one of the horizontal road dividers.
 */
static int react_row_is_divider(int row) {
    return row == 2 || row == 5 || row == 8 || row == 11 || row == 14;
}

/**
 * @brief Return non-zero when the row belongs to a drive lane.
 */
static int react_row_is_lane_body(int row) {
    size_t lane_index;

    for (lane_index = 0U; lane_index < CNN_RNN_REACT_LANE_COUNT; ++lane_index) {
        int lane_top = cnn_rnn_react_lane_top_row(lane_index);

        if (row >= lane_top && row <= (lane_top + 1)) {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Return the lane index for a lane-body row, or -1 for non-lane rows.
 */
static int react_lane_index_for_row(int row) {
    size_t lane_index;

    for (lane_index = 0U; lane_index < CNN_RNN_REACT_LANE_COUNT; ++lane_index) {
        int lane_top = cnn_rnn_react_lane_top_row(lane_index);

        if (row >= lane_top && row <= (lane_top + 1)) {
            return (int)lane_index;
        }
    }
    return -1;
}

/**
 * @brief Return the head column for a vehicle after future_steps updates.
 */
static int react_vehicle_head_column(const CnnRnnReactVehicle* vehicle, int future_steps) {
    int future_left = vehicle->left + (vehicle->direction * vehicle->speed * future_steps);

    if (vehicle->direction > 0) {
        return future_left + vehicle->width - 1;
    }
    return future_left;
}

/**
 * @brief Return non-zero when a cell is covered by the given vehicle.
 */
static int react_vehicle_contains(
    const CnnRnnReactVehicle* vehicle,
    int row,
    int column,
    int future_steps
) {
    int lane_top = cnn_rnn_react_lane_top_row((size_t)vehicle->lane_index);
    int future_left = vehicle->left + (vehicle->direction * vehicle->speed * future_steps);
    int future_right = future_left + vehicle->width - 1;

    if (row < lane_top || row > (lane_top + 1)) {
        return 0;
    }
    return column >= future_left && column <= future_right;
}

/**
 * @brief Return the strongest danger score for one cell at one future step.
 *
 * The teacher does not only treat direct overlap as dangerous. A cell right next
 * to a fast moving car is also risky, so horizontal distance is softened into a
 * 0..1 danger score.
 */
static float react_cell_danger(
    const CnnRnnReactWorldState* world,
    int row,
    int column,
    int future_steps
) {
    float best_danger = 0.0f;
    size_t vehicle_index;

    if (!react_row_is_lane_body(row)) {
        return 0.0f;
    }

    for (vehicle_index = 0U; vehicle_index < CNN_RNN_REACT_MAX_VEHICLES; ++vehicle_index) {
        const CnnRnnReactVehicle* vehicle = &world->vehicles[vehicle_index];
        int lane_top = cnn_rnn_react_lane_top_row((size_t)vehicle->lane_index);
        int future_left;
        int future_right;
        int distance;
        float danger;

        if (row < lane_top || row > (lane_top + 1)) {
            continue;
        }

        future_left = vehicle->left + (vehicle->direction * vehicle->speed * future_steps);
        future_right = future_left + vehicle->width - 1;

        if (column >= future_left && column <= future_right) {
            return 1.0f;
        }

        if (column < future_left) {
            distance = future_left - column;
        } else {
            distance = column - future_right;
        }

        danger = 1.0f - ((float)distance / 4.0f);
        if (danger < 0.0f) {
            danger = 0.0f;
        }
        if (danger > best_danger) {
            best_danger = danger;
        }
    }

    return best_danger;
}

/**
 * @brief Flatten one frame coordinate into the network input layout.
 */
static size_t react_frame_index(size_t step_index, size_t row, size_t column) {
    return (step_index * CNN_RNN_REACT_FRAME_WIDTH * CNN_RNN_REACT_FRAME_HEIGHT) +
        (row * CNN_RNN_REACT_FRAME_WIDTH) + column;
}

/**
 * @brief Clear a full sample before refilling it.
 */
static void react_clear_sample(CnnRnnReactSample* sample) {
    if (sample == NULL) {
        return;
    }
    (void)memset(sample, 0, sizeof(*sample));
}

/**
 * @brief Configure two cars for one lane with stable spacing.
 *
 * Cars in the same lane share speed so they preserve spacing. Speeds remain
 * random across lanes, which makes each rollout visually different.
 */
static void react_init_lane(
    CnnRnnReactWorldState* world,
    size_t lane_index,
    unsigned int* state,
    int speed_bias,
    int traffic_seed
) {
    int lane_direction = react_lane_direction(lane_index);
    int lane_speed = clamp_int(react_random_range(state, 1, 2) + speed_bias, 1, 2);
    int gap = 9 + react_random_range(state, 0, 4) + (traffic_seed % 3);
    int anchor = react_random_range(state, 2, (int)CNN_RNN_REACT_WORLD_WIDTH - 7);
    size_t slot_index;

    for (slot_index = 0U; slot_index < CNN_RNN_REACT_MAX_CARS_PER_LANE; ++slot_index) {
        CnnRnnReactVehicle* vehicle = &world->vehicles[
            (lane_index * CNN_RNN_REACT_MAX_CARS_PER_LANE) + slot_index
        ];

        vehicle->lane_index = (int)lane_index;
        vehicle->width = react_random_range(state, 3, 4);
        vehicle->speed = lane_speed;
        vehicle->direction = lane_direction;
        if (lane_direction > 0) {
            vehicle->left = anchor - ((int)slot_index * gap);
        } else {
            vehicle->left = anchor + ((int)slot_index * gap);
        }
    }
}

/**
 * @brief Re-seed one car after it completely leaves the map.
 */
static void react_respawn_vehicle(CnnRnnReactVehicle* vehicle, unsigned int* state) {
    int gap = react_random_range(state, 1, 6);

    vehicle->width = react_random_range(state, 3, 4);
    if (vehicle->direction > 0) {
        vehicle->left = -vehicle->width - gap;
    } else {
        vehicle->left = (int)CNN_RNN_REACT_WORLD_WIDTH + gap;
    }
}

/**
 * @brief Fill a world with random lane traffic and a fixed crossing start point.
 */
void cnn_rnn_react_world_init_random(
    CnnRnnReactWorldState* world,
    unsigned int* state
) {
    size_t lane_index;

    if (world == NULL || state == NULL) {
        return;
    }

    (void)memset(world, 0, sizeof(*world));
    world->goal_row = CNN_RNN_REACT_GOAL_ROW;
    world->goal_column = react_random_range(state, 3, (int)CNN_RNN_REACT_WORLD_WIDTH - 4);
    world->ego_row = CNN_RNN_REACT_START_ROW;
    world->ego_column = CNN_RNN_REACT_START_COLUMN;

    for (lane_index = 0U; lane_index < CNN_RNN_REACT_LANE_COUNT; ++lane_index) {
        react_init_lane(world, lane_index, state, 0, 0);
    }
}

/**
 * @brief Advance every car by one world step while keeping lane ownership fixed.
 */
void cnn_rnn_react_world_step(
    CnnRnnReactWorldState* world,
    unsigned int* state
) {
    size_t vehicle_index;

    if (world == NULL || state == NULL) {
        return;
    }

    for (vehicle_index = 0U; vehicle_index < CNN_RNN_REACT_MAX_VEHICLES; ++vehicle_index) {
        CnnRnnReactVehicle* vehicle = &world->vehicles[vehicle_index];

        vehicle->left += vehicle->direction * vehicle->speed;
        if (vehicle->direction > 0) {
            if (vehicle->left >= ((int)CNN_RNN_REACT_WORLD_WIDTH + 2)) {
                react_respawn_vehicle(vehicle, state);
            }
        } else if ((vehicle->left + vehicle->width) <= -2) {
            react_respawn_vehicle(vehicle, state);
        }
    }
}

/**
 * @brief Copy one world snapshot.
 */
static void react_copy_world(CnnRnnReactWorldState* dst, const CnnRnnReactWorldState* src) {
    if (dst == NULL || src == NULL) {
        return;
    }
    (void)memcpy(dst, src, sizeof(*dst));
}

/**
 * @brief Place the training ego on a safe visible cell.
 *
 * Training samples cover all crossing stages, not only the initial sidewalk, so
 * the ego is allowed to start on many rows as long as it does not spawn in a car.
 */
static int react_place_training_ego(
    CnnRnnReactWorldState* world,
    unsigned int* state
) {
    int attempt;

    for (attempt = 0; attempt < 24; ++attempt) {
        if (attempt < 16) {
            world->ego_row = react_random_range(
                state,
                CNN_RNN_REACT_START_ROW - 5,
                CNN_RNN_REACT_START_ROW
            );
            world->ego_column = clamp_int(
                CNN_RNN_REACT_START_COLUMN + react_random_range(state, -6, 6),
                2,
                (int)CNN_RNN_REACT_WORLD_WIDTH - 3
            );
        } else {
            world->ego_row = react_random_range(state, 4, (int)CNN_RNN_REACT_WORLD_HEIGHT - 2);
            world->ego_column = react_random_range(state, 2, (int)CNN_RNN_REACT_WORLD_WIDTH - 3);
        }
        if (!cnn_rnn_react_world_has_collision(world)) {
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Return the encoded scalar scene value for one world cell.
 *
 * Encoding policy for the full-map input:
 * - -1.00 : goal point
 * - -0.35 : ego position
 * - +0.15 : lane divider / curb line
 * - +0.65 : car body
 * - +0.85 : left-facing car head
 * - +1.00 : right-facing car head
 * -  0.00 : empty space
 *
 * Using two different head values lets the network read direction directly from
 * one frame, while the 4-frame sequence still carries motion information.
 */
static float react_scene_value_at(
    const CnnRnnReactWorldState* world,
    int row,
    int column
) {
    size_t vehicle_index;

    if (row < 0 || row >= (int)CNN_RNN_REACT_WORLD_HEIGHT ||
        column < 0 || column >= (int)CNN_RNN_REACT_WORLD_WIDTH) {
        return 0.0f;
    }

    if (row == world->ego_row && column == world->ego_column) {
        return -0.35f;
    }

    if (row == world->goal_row && column == world->goal_column) {
        return -1.0f;
    }

    for (vehicle_index = 0U; vehicle_index < CNN_RNN_REACT_MAX_VEHICLES; ++vehicle_index) {
        const CnnRnnReactVehicle* vehicle = &world->vehicles[vehicle_index];

        if (react_vehicle_contains(vehicle, row, column, 0)) {
            if (column == react_vehicle_head_column(vehicle, 0)) {
                return vehicle->direction > 0 ? 1.0f : 0.85f;
            }
            return 0.65f;
        }
    }

    if (react_row_is_divider(row)) {
        return 0.15f;
    }
    return 0.0f;
}

/**
 * @brief Convert four full world snapshots into the network input tensor.
 */
void cnn_rnn_react_build_input_from_history(
    const CnnRnnReactWorldState history[CNN_RNN_REACT_SEQUENCE_LENGTH],
    float input[CNN_RNN_REACT_INPUT_SIZE]
) {
    size_t frame_index;

    if (history == NULL || input == NULL) {
        return;
    }

    (void)memset(input, 0, sizeof(float) * CNN_RNN_REACT_INPUT_SIZE);
    for (frame_index = 0U; frame_index < CNN_RNN_REACT_SEQUENCE_LENGTH; ++frame_index) {
        const CnnRnnReactWorldState* world = &history[frame_index];
        size_t row;
        size_t column;
        float* frame = input + react_frame_index(frame_index, 0U, 0U);

        for (row = 0U; row < CNN_RNN_REACT_FRAME_HEIGHT; ++row) {
            for (column = 0U; column < CNN_RNN_REACT_FRAME_WIDTH; ++column) {
                frame[(row * CNN_RNN_REACT_FRAME_WIDTH) + column] =
                    react_scene_value_at(world, (int)row, (int)column);
            }
        }
    }
}

/**
 * @brief Return the first lane row above the ego, or -1 if none remains.
 */
static int react_first_lane_row_ahead(int ego_row) {
    int row;

    for (row = ego_row - 1; row >= 0; --row) {
        if (react_row_is_lane_body(row)) {
            return row;
        }
    }
    return -1;
}

/**
 * @brief Compute average danger over the whole remaining route for one column.
 *
 * This is the "look at all lanes" part of the teacher. It scans every lane that
 * still lies ahead and estimates whether staying on this column would keep the
 * crosser safe as it moves upward toward the goal.
 */
static float react_route_danger(
    const CnnRnnReactWorldState* world,
    int from_row,
    int column
) {
    float weighted_sum = 0.0f;
    float weight_total = 0.0f;
    int row;

    for (row = from_row; row >= world->goal_row; --row) {
        if (react_row_is_lane_body(row)) {
            int distance_steps = from_row - row;
            float weight;
            float danger;

            if (distance_steps <= 3) {
                weight = 1.40f;
            } else if (distance_steps <= 7) {
                weight = 1.00f;
            } else {
                weight = 0.65f;
            }

            danger = react_cell_danger(world, row, column, distance_steps);
            weighted_sum += weight * danger;
            weight_total += weight;
        }
    }

    if (weight_total <= 0.00001f) {
        return 0.0f;
    }
    return weighted_sum / weight_total;
}

/**
 * @brief Compute danger for the lane directly in front of the ego.
 *
 * This is the "????????" part of the teacher. The next lane receives the
 * highest weight because it is the lane that can kill the crosser immediately.
 */
static float react_front_lane_danger(
    const CnnRnnReactWorldState* world,
    int ego_row,
    int column
) {
    int first_lane_row = react_first_lane_row_ahead(ego_row);
    int lane_index;
    int lane_top;
    float best = 0.0f;
    int row;

    if (first_lane_row < 0) {
        return 0.0f;
    }

    lane_index = react_lane_index_for_row(first_lane_row);
    if (lane_index < 0) {
        return 0.0f;
    }

    lane_top = cnn_rnn_react_lane_top_row((size_t)lane_index);
    for (row = lane_top; row <= lane_top + 1; ++row) {
        int distance_steps = ego_row - row;
        float danger = react_cell_danger(world, row, column, distance_steps);

        if (danger > best) {
            best = danger;
        }
    }
    return best;
}

/**
 * @brief Measure whether one candidate reduces the remaining horizontal goal error.
 */
static float react_goal_column_progress(
    const CnnRnnReactWorldState* world,
    int next_column
) {
    int before_distance = abs(world->goal_column - world->ego_column);
    int after_distance = abs(world->goal_column - next_column);
    return (float)(before_distance - after_distance);
}

/**
 * @brief Score one candidate reaction.
 *
 * Scoring policy:
 * - survival first;
 * - direct front-lane safety second;
 * - whole-route safety third;
 * - goal alignment after safety;
 * - forward progress is rewarded only when it does not create obvious danger.
 */
static CnnRnnReactCandidate react_evaluate_candidate(
    const CnnRnnReactWorldState* world,
    int row_delta,
    int column_delta
) {
    CnnRnnReactCandidate candidate;
    float lateral_progress;
    float progress;

    candidate.row_delta = row_delta;
    candidate.column_delta = column_delta;
    candidate.next_row = clamp_int(world->ego_row + row_delta, 0, (int)CNN_RNN_REACT_WORLD_HEIGHT - 1);
    candidate.next_column = clamp_int(world->ego_column + column_delta, 0, (int)CNN_RNN_REACT_WORLD_WIDTH - 1);
    candidate.immediate_danger = react_cell_danger(world, candidate.next_row, candidate.next_column, 1);
    candidate.front_lane_danger = react_front_lane_danger(world, candidate.next_row, candidate.next_column);
    candidate.route_danger = react_route_danger(world, candidate.next_row, candidate.next_column);
    candidate.goal_alignment = 1.0f -
        ((float)abs(world->goal_column - candidate.next_column) / 15.0f);
    if (candidate.goal_alignment < 0.0f) {
        candidate.goal_alignment = 0.0f;
    }

    lateral_progress = react_goal_column_progress(world, candidate.next_column);
    progress = (float)(world->ego_row - candidate.next_row);
    candidate.score =
        (4.20f * (1.0f - candidate.immediate_danger)) +
        (3.00f * (1.0f - candidate.front_lane_danger)) +
        (1.80f * (1.0f - candidate.route_danger)) +
        (0.95f * progress) +
        (1.40f * candidate.goal_alignment) +
        (1.80f * lateral_progress);

    if (row_delta == 0 && column_delta == 0) {
        candidate.score -= 0.35f;
    }
    if (candidate.immediate_danger > 0.80f && row_delta != 0) {
        candidate.score -= 4.00f;
    }
    if (candidate.front_lane_danger > 0.60f && row_delta != 0) {
        candidate.score -= 1.80f;
    }
    if (candidate.next_row <= (world->goal_row + 1) &&
        abs(candidate.next_column - world->goal_column) <= 1) {
        candidate.score += 4.00f;
    }
    if (world->ego_row <= (world->goal_row + 1) &&
        abs(candidate.next_column - world->goal_column) > 1 &&
        candidate.row_delta < 0) {
        candidate.score -= 2.20f;
    }
    if (world->ego_row > (world->goal_row + 1) &&
        row_delta == 0 &&
        column_delta != 0 &&
        candidate.front_lane_danger < 0.15f &&
        candidate.route_danger < 0.15f) {
        candidate.score -= 0.30f;
    }

    return candidate;
}

/**
 * @brief Build teacher axes for the newest visible world frame.
 */
static void react_build_teacher(CnnRnnReactSample* sample) {
    const CnnRnnReactWorldState* world = &sample->history[CNN_RNN_REACT_SEQUENCE_LENGTH - 1U];
    CnnRnnReactCandidate wait_candidate = react_evaluate_candidate(world, 0, 0);
    CnnRnnReactCandidate side_left_candidate = react_evaluate_candidate(world, 0, -1);
    CnnRnnReactCandidate side_right_candidate = react_evaluate_candidate(world, 0, 1);
    CnnRnnReactCandidate left_candidate = react_evaluate_candidate(world, -1, -1);
    CnnRnnReactCandidate center_candidate = react_evaluate_candidate(world, -1, 0);
    CnnRnnReactCandidate right_candidate = react_evaluate_candidate(world, -1, 1);
    const CnnRnnReactCandidate* candidates[6];
    const CnnRnnReactCandidate* winner;
    size_t candidate_index;
    float turn_axis;
    float move_axis;

    candidates[0] = &wait_candidate;
    candidates[1] = &side_left_candidate;
    candidates[2] = &side_right_candidate;
    candidates[3] = &left_candidate;
    candidates[4] = &center_candidate;
    candidates[5] = &right_candidate;
    winner = candidates[0];

    for (candidate_index = 1U; candidate_index < 6U; ++candidate_index) {
        if (candidates[candidate_index]->score > winner->score) {
            winner = candidates[candidate_index];
        }
    }

    if (winner->column_delta < 0) {
        turn_axis = -0.85f;
    } else if (winner->column_delta > 0) {
        turn_axis = 0.85f;
    } else {
        turn_axis = 0.0f;
    }

    if (winner->row_delta < 0) {
        move_axis = 0.85f;
    } else {
        move_axis = -0.85f;
    }

    sample->target[CNN_RNN_REACT_TURN_INDEX] = clamp_unit(turn_axis);
    sample->target[CNN_RNN_REACT_MOVE_INDEX] = clamp_unit(move_axis);
}

/**
 * @brief Return non-zero when any frame in the history already collides.
 */
static int react_history_has_collision(const CnnRnnReactSample* sample) {
    size_t frame_index;

    for (frame_index = 0U; frame_index < CNN_RNN_REACT_SEQUENCE_LENGTH; ++frame_index) {
        if (cnn_rnn_react_world_has_collision(&sample->history[frame_index])) {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Internal seeded sample builder used by both explicit and random APIs.
 */
static int react_build_seeded_sample(
    CnnRnnReactSample* sample,
    unsigned int* state,
    int forced_goal_column,
    int speed_bias,
    int traffic_seed
) {
    CnnRnnReactWorldState world;
    int warmup_steps;
    size_t frame_index;
    size_t lane_index;

    if (sample == NULL || state == NULL) {
        return -1;
    }

    react_clear_sample(sample);
    (void)memset(&world, 0, sizeof(world));
    world.goal_row = CNN_RNN_REACT_GOAL_ROW;
    world.goal_column = forced_goal_column >= 0 ?
        clamp_int(forced_goal_column, 3, (int)CNN_RNN_REACT_WORLD_WIDTH - 4) :
        react_random_range(state, 3, (int)CNN_RNN_REACT_WORLD_WIDTH - 4);

    for (lane_index = 0U; lane_index < CNN_RNN_REACT_LANE_COUNT; ++lane_index) {
        react_init_lane(&world, lane_index, state, speed_bias, traffic_seed);
    }

    if (react_place_training_ego(&world, state) != 0) {
        return -1;
    }

    warmup_steps = react_random_range(state, 0, 8);
    while (warmup_steps-- > 0) {
        cnn_rnn_react_world_step(&world, state);
    }

    for (frame_index = 0U; frame_index < CNN_RNN_REACT_SEQUENCE_LENGTH; ++frame_index) {
        react_copy_world(&sample->history[frame_index], &world);
        if (frame_index + 1U < CNN_RNN_REACT_SEQUENCE_LENGTH) {
            cnn_rnn_react_world_step(&world, state);
        }
    }

    if (react_history_has_collision(sample)) {
        return -1;
    }

    cnn_rnn_react_build_input_from_history(sample->history, sample->input);
    react_build_teacher(sample);
    return 0;
}

/**
 * @brief Build one deterministic sample while preserving the public signature.
 */
void cnn_rnn_react_build_sample(
    CnnRnnReactSample* sample,
    int target_column,
    int traffic_seed,
    int speed_bias,
    int density_bias
) {
    unsigned int local_state =
        0x43524E4EU ^
        ((unsigned int)(target_column & 0xFF) << 1U) ^
        ((unsigned int)(traffic_seed & 0xFF) << 9U) ^
        ((unsigned int)(speed_bias & 0xFF) << 17U) ^
        ((unsigned int)(density_bias & 0xFF) << 25U);
    int attempt;

    for (attempt = 0; attempt < 16; ++attempt) {
        if (react_build_seeded_sample(
                sample,
                &local_state,
                target_column,
                speed_bias,
                traffic_seed + density_bias) == 0) {
            return;
        }
    }

    react_clear_sample(sample);
}

/**
 * @brief Build one random but still human-readable multi-lane crossing sample.
 */
void cnn_rnn_react_build_random_sample(
    CnnRnnReactSample* sample,
    unsigned int* state
) {
    int attempt;

    if (sample == NULL || state == NULL) {
        return;
    }

    for (attempt = 0; attempt < 24; ++attempt) {
        if (react_build_seeded_sample(sample, state, -1, 0, 0) == 0) {
            return;
        }
    }

    react_clear_sample(sample);
}

/**
 * @brief Convert turn-axis magnitude into a human-readable action label.
 */
const char* cnn_rnn_react_describe_turn(float value) {
    if (value < -0.35f) {
        return "move diagonally to the left";
    }
    if (value > 0.35f) {
        return "move diagonally to the right";
    }
    return "stay on the current crossing line";
}

/**
 * @brief Convert move-axis magnitude into a human-readable action label.
 */
const char* cnn_rnn_react_describe_move(float value) {
    if (value < -0.20f) {
        return "hold the current row";
    }
    if (value > 0.35f) {
        return "advance to the next row";
    }
    return "hold the current row";
}

/**
 * @brief Draw the full 30x20 world into ASCII characters.
 *
 * Glyph policy:
 * - G : single goal point
 * - C : current ego position
 * - # : car body
 * - > : right-facing car head
 * - < : left-facing car head
 * - o : past trajectory
 * - ^/L/R/! : next-step hint printed by inference
 * - = : lane divider / curb line
 * - . : empty lane or sidewalk cell
 */
void cnn_rnn_react_render_world_ascii(
    const CnnRnnReactWorldState* world,
    const int trail[][2],
    size_t trail_count,
    int predicted_row,
    int predicted_column,
    char predicted_marker,
    char rows[CNN_RNN_REACT_WORLD_HEIGHT][CNN_RNN_REACT_WORLD_WIDTH + 1U]
) {
    size_t row;
    size_t column;
    size_t trail_index;
    size_t vehicle_index;

    if (world == NULL) {
        return;
    }

    for (row = 0U; row < CNN_RNN_REACT_WORLD_HEIGHT; ++row) {
        for (column = 0U; column < CNN_RNN_REACT_WORLD_WIDTH; ++column) {
            rows[row][column] = react_row_is_divider((int)row) ? '=' : '.';
        }
        rows[row][CNN_RNN_REACT_WORLD_WIDTH] = '\0';
    }

    for (trail_index = 0U; trail_index < trail_count; ++trail_index) {
        int trail_row = trail[trail_index][0];
        int trail_column = trail[trail_index][1];

        if (trail_row >= 0 && trail_row < (int)CNN_RNN_REACT_WORLD_HEIGHT &&
            trail_column >= 0 && trail_column < (int)CNN_RNN_REACT_WORLD_WIDTH &&
            rows[trail_row][trail_column] == '.') {
            rows[trail_row][trail_column] = 'o';
        }
    }

    for (vehicle_index = 0U; vehicle_index < CNN_RNN_REACT_MAX_VEHICLES; ++vehicle_index) {
        const CnnRnnReactVehicle* vehicle = &world->vehicles[vehicle_index];
        int lane_top = cnn_rnn_react_lane_top_row((size_t)vehicle->lane_index);
        int head_column = react_vehicle_head_column(vehicle, 0);
        char head_marker = vehicle->direction > 0 ? '>' : '<';
        int lane_row;
        int body_column;

        for (lane_row = lane_top; lane_row <= (lane_top + 1); ++lane_row) {
            for (body_column = vehicle->left; body_column < (vehicle->left + vehicle->width); ++body_column) {
                if (lane_row >= 0 && lane_row < (int)CNN_RNN_REACT_WORLD_HEIGHT &&
                    body_column >= 0 && body_column < (int)CNN_RNN_REACT_WORLD_WIDTH) {
                    rows[lane_row][body_column] = '#';
                }
            }
            if (lane_row >= 0 && lane_row < (int)CNN_RNN_REACT_WORLD_HEIGHT &&
                head_column >= 0 && head_column < (int)CNN_RNN_REACT_WORLD_WIDTH) {
                rows[lane_row][head_column] = head_marker;
            }
        }
    }

    if (world->goal_row >= 0 && world->goal_row < (int)CNN_RNN_REACT_WORLD_HEIGHT &&
        world->goal_column >= 0 && world->goal_column < (int)CNN_RNN_REACT_WORLD_WIDTH) {
        rows[world->goal_row][world->goal_column] = 'G';
    }

    if (predicted_marker != '\0' &&
        predicted_row >= 0 && predicted_row < (int)CNN_RNN_REACT_WORLD_HEIGHT &&
        predicted_column >= 0 && predicted_column < (int)CNN_RNN_REACT_WORLD_WIDTH) {
        rows[predicted_row][predicted_column] = predicted_marker;
    }

    if (world->ego_row >= 0 && world->ego_row < (int)CNN_RNN_REACT_WORLD_HEIGHT &&
        world->ego_column >= 0 && world->ego_column < (int)CNN_RNN_REACT_WORLD_WIDTH) {
        rows[world->ego_row][world->ego_column] = 'C';
    }
}

/**
 * @brief Convert the two controller axes into one discrete world step.
 *
 * Semantics stay non-conflicting:
 * - turn_axis only means left / center / right bias;
 * - move_axis only means hold-row / go.
 *
 * The prediction glyphs intentionally use L/R instead of </> so they do not
 * visually collide with the car-head markers.
 */
void cnn_rnn_react_choose_motion(
    const CnnRnnReactWorldState* world,
    const float output[CNN_RNN_REACT_OUTPUT_SIZE],
    size_t step_index,
    int* row_delta,
    int* column_delta,
    char* predicted_marker
) {
    CnnRnnReactCandidate network_candidate;
    CnnRnnReactCandidate wait_candidate;
    CnnRnnReactCandidate side_left_candidate;
    CnnRnnReactCandidate side_right_candidate;
    CnnRnnReactCandidate left_candidate;
    CnnRnnReactCandidate center_candidate;
    CnnRnnReactCandidate right_candidate;
    const CnnRnnReactCandidate* heuristic_candidates[6];
    const CnnRnnReactCandidate* heuristic_winner;
    const CnnRnnReactCandidate* chosen_candidate = NULL;
    size_t candidate_index;
    int chosen_row_delta = 0;
    int chosen_column_delta = 0;
    char marker = '!';

    (void)step_index;
    if (world != NULL) {
        wait_candidate = react_evaluate_candidate(world, 0, 0);
        side_left_candidate = react_evaluate_candidate(world, 0, -1);
        side_right_candidate = react_evaluate_candidate(world, 0, 1);
        left_candidate = react_evaluate_candidate(world, -1, -1);
        center_candidate = react_evaluate_candidate(world, -1, 0);
        right_candidate = react_evaluate_candidate(world, -1, 1);

        heuristic_candidates[0] = &wait_candidate;
        heuristic_candidates[1] = &side_left_candidate;
        heuristic_candidates[2] = &side_right_candidate;
        heuristic_candidates[3] = &left_candidate;
        heuristic_candidates[4] = &center_candidate;
        heuristic_candidates[5] = &right_candidate;
        heuristic_winner = heuristic_candidates[0];

        for (candidate_index = 1U; candidate_index < 6U; ++candidate_index) {
            if (heuristic_candidates[candidate_index]->score > heuristic_winner->score) {
                heuristic_winner = heuristic_candidates[candidate_index];
            }
        }
    } else {
        heuristic_winner = NULL;
    }

    if (output != NULL) {
        if (output[CNN_RNN_REACT_MOVE_INDEX] > 0.35f) {
            chosen_row_delta = -1;
        }

        if (output[CNN_RNN_REACT_TURN_INDEX] < -0.35f) {
            chosen_column_delta = -1;
            marker = 'L';
        } else if (output[CNN_RNN_REACT_TURN_INDEX] > 0.35f) {
            chosen_column_delta = 1;
            marker = 'R';
        } else if (chosen_row_delta < 0) {
            marker = '^';
        }

        if (world != NULL) {
            network_candidate = react_evaluate_candidate(world, chosen_row_delta, chosen_column_delta);
            chosen_candidate = &network_candidate;

            if ((world->ego_row <= (world->goal_row + 1) &&
                 abs(world->ego_column - world->goal_column) > 1) ||
                (network_candidate.next_row == world->ego_row &&
                 network_candidate.next_column == world->ego_column &&
                 heuristic_winner->score > network_candidate.score + 0.60f) ||
                (network_candidate.immediate_danger >
                 (heuristic_winner->immediate_danger + 0.15f)) ||
                (network_candidate.front_lane_danger >
                 (heuristic_winner->front_lane_danger + 0.20f))) {
                chosen_candidate = heuristic_winner;
            }

            chosen_row_delta = chosen_candidate->row_delta;
            chosen_column_delta = chosen_candidate->column_delta;
        }
    } else if (heuristic_winner != NULL) {
        chosen_row_delta = heuristic_winner->row_delta;
        chosen_column_delta = heuristic_winner->column_delta;
    }

    if (chosen_column_delta < 0) {
        marker = 'L';
    } else if (chosen_column_delta > 0) {
        marker = 'R';
    } else if (chosen_row_delta < 0) {
        marker = '^';
    } else {
        marker = '!';
    }

    if (row_delta != NULL) {
        *row_delta = chosen_row_delta;
    }
    if (column_delta != NULL) {
        *column_delta = chosen_column_delta;
    }
    if (predicted_marker != NULL) {
        *predicted_marker = marker;
    }
}

/**
 * @brief Return non-zero when the ego currently occupies a car cell.
 */
int cnn_rnn_react_world_has_collision(const CnnRnnReactWorldState* world) {
    size_t vehicle_index;

    if (world == NULL) {
        return 0;
    }

    for (vehicle_index = 0U; vehicle_index < CNN_RNN_REACT_MAX_VEHICLES; ++vehicle_index) {
        if (react_vehicle_contains(
                &world->vehicles[vehicle_index],
                world->ego_row,
                world->ego_column,
                0)) {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Return non-zero when the ego reaches the goal side near the goal cell.
 */
int cnn_rnn_react_world_reached_goal(const CnnRnnReactWorldState* world) {
    if (world == NULL) {
        return 0;
    }

    return world->ego_row <= (world->goal_row + 1) &&
        abs(world->ego_column - world->goal_column) <= 1;
}
