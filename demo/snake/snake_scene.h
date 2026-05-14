/**
 * @file snake_scene.h
 * @brief Snake game scene for TUI demo with neural network
 *
 * Header-only (static inline), follows road_graph_nav_scene.h pattern.
 * Defines game state, encoding, teacher heuristic, and rendering.
 */

#ifndef SNAKE_SCENE_H
#define SNAKE_SCENE_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

/* --- Constants --- */
#define SNAKE_GRID_W      20
#define SNAKE_GRID_H      12
#define SNAKE_INPUT_SIZE  8
#define SNAKE_OUTPUT_SIZE 4
#define SNAKE_MAX_LENGTH  256
#define SNAKE_MAX_STEPS   500
#define SNAKE_INIT_LENGTH 3
#define SNAKE_SEED        42U

/* --- Direction enum --- */
typedef enum {
    SNAKE_UP    = 0,
    SNAKE_RIGHT = 1,
    SNAKE_DOWN  = 2,
    SNAKE_LEFT  = 3
} SnakeDirection;

/* --- Relative turn enum (teacher output) --- */
typedef enum {
    SNAKE_TURN_STRAIGHT = 0,
    SNAKE_TURN_LEFT     = 1,
    SNAKE_TURN_RIGHT    = 2
} SnakeTurn;

/* --- Game state --- */
typedef struct {
    /* Grid: 0=empty, 1=wall, 2=body, 3=food, 4=head */
    int grid[SNAKE_GRID_H][SNAKE_GRID_W];
    int body_x[SNAKE_MAX_LENGTH];
    int body_y[SNAKE_MAX_LENGTH];
    int length;
    SnakeDirection direction;
    int score;
    int food_x;
    int food_y;
    int game_over;
    int steps;
} SnakeState;

/* --- LCG RNG (matches nested_nav pattern) --- */
static inline unsigned int snake_next_random(unsigned int* state) {
    *state = (*state) * 1664525U + 1013904223U;
    return *state;
}

static inline int snake_rand_int(unsigned int* state, int max) {
    return (int)(snake_next_random(state) % (unsigned int)max);
}

/* --- Direction helpers --- */
static inline int snake_dir_dx(SnakeDirection dir) {
    switch (dir) {
        case SNAKE_RIGHT: return 1;
        case SNAKE_LEFT:  return -1;
        default:          return 0;
    }
}

static inline int snake_dir_dy(SnakeDirection dir) {
    switch (dir) {
        case SNAKE_DOWN: return 1;
        case SNAKE_UP:   return -1;
        default:         return 0;
    }
}

/* --- Check if a cell is dangerous (wall or body) --- */
static inline int snake_cell_dangerous(const SnakeState* state, int x, int y) {
    if (x < 1 || x >= SNAKE_GRID_W - 1 || y < 1 || y >= SNAKE_GRID_H - 1) {
        return 1; /* wall */
    }
    if (state->grid[y][x] == 2) {
        return 1; /* body */
    }
    return 0;
}

/* --- Convert relative turn to absolute direction --- */
static inline SnakeDirection snake_turn_to_absolute(SnakeDirection current, SnakeTurn turn) {
    switch (turn) {
        case SNAKE_TURN_STRAIGHT:
            return current;
        case SNAKE_TURN_LEFT:
            switch (current) {
                case SNAKE_UP:    return SNAKE_LEFT;
                case SNAKE_RIGHT: return SNAKE_UP;
                case SNAKE_DOWN:  return SNAKE_RIGHT;
                case SNAKE_LEFT:  return SNAKE_DOWN;
                default:          return current;
            }
        case SNAKE_TURN_RIGHT:
            switch (current) {
                case SNAKE_UP:    return SNAKE_RIGHT;
                case SNAKE_RIGHT: return SNAKE_DOWN;
                case SNAKE_DOWN:  return SNAKE_LEFT;
                case SNAKE_LEFT:  return SNAKE_UP;
                default:          return current;
            }
        default:
            return current;
    }
}

/* --- Encode game state into NN input features (8 floats, [-1, 1]) --- */
static inline void snake_encode_state(const SnakeState* state, float* input) {
    int hx = state->body_x[0];
    int hy = state->body_y[0];
    int dx = snake_dir_dx(state->direction);
    int dy = snake_dir_dy(state->direction);

    /* Look-ahead positions for danger detection */
    int fwd_x = hx + dx;
    int fwd_y = hy + dy;

    /* Left of heading */
    int left_x, left_y;
    /* Right of heading */
    int right_x, right_y;

    switch (state->direction) {
        case SNAKE_UP:    left_x = hx - 1; left_y = hy;     right_x = hx + 1; right_y = hy;     break;
        case SNAKE_DOWN:  left_x = hx + 1; left_y = hy;     right_x = hx - 1; right_y = hy;     break;
        case SNAKE_LEFT:  left_x = hx;     left_y = hy + 1; right_x = hx;     right_y = hy - 1; break;
        case SNAKE_RIGHT: left_x = hx;     left_y = hy - 1; right_x = hx;     right_y = hy + 1; break;
        default:          left_x = hx;     left_y = hy;     right_x = hx;     right_y = hy;     break;
    }

    /* Features */
    input[0] = snake_cell_dangerous(state, fwd_x, fwd_y)   ? 1.0f : 0.0f;
    input[1] = snake_cell_dangerous(state, left_x, left_y)   ? 1.0f : 0.0f;
    input[2] = snake_cell_dangerous(state, right_x, right_y) ? 1.0f : 0.0f;

    input[3] = (float)(state->food_x - hx) / (float)(SNAKE_GRID_W - 1);
    input[4] = (float)(state->food_y - hy) / (float)(SNAKE_GRID_H - 1);

    input[5] = (state->direction == SNAKE_UP)    ? 1.0f : 0.0f;
    input[6] = (state->direction == SNAKE_DOWN)  ? 1.0f : 0.0f;
    input[7] = (state->direction == SNAKE_LEFT)  ? 1.0f : 0.0f;
}

/* --- Teacher heuristic: priority-ordered deterministic rule --- */
static inline SnakeTurn snake_teacher_turn(const SnakeState* state) {
    int can_straight, can_left, can_right;
    int hx = state->body_x[0];
    int hy = state->body_y[0];
    int dx = snake_dir_dx(state->direction);
    int dy = snake_dir_dy(state->direction);

    /* Check safety of relative turns */
    can_straight = !snake_cell_dangerous(state, hx + dx, hy + dy);

    switch (state->direction) {
        case SNAKE_UP:    can_left  = !snake_cell_dangerous(state, hx - 1, hy);     can_right = !snake_cell_dangerous(state, hx + 1, hy);     break;
        case SNAKE_DOWN:  can_left  = !snake_cell_dangerous(state, hx + 1, hy);     can_right = !snake_cell_dangerous(state, hx - 1, hy);     break;
        case SNAKE_LEFT:  can_left  = !snake_cell_dangerous(state, hx, hy + 1);     can_right = !snake_cell_dangerous(state, hx, hy - 1);     break;
        case SNAKE_RIGHT: can_left  = !snake_cell_dangerous(state, hx, hy - 1);     can_right = !snake_cell_dangerous(state, hx, hy + 1);     break;
        default:          can_left  = 0;                       can_right = 0;                       break;
    }

    /* Compute alignment scores for each safe turn */
    int straight_score = -1, left_score = -1, right_score = -1;

    if (can_straight) {
        int nx = hx + dx;
        int ny = hy + dy;
        straight_score = (nx - state->food_x) * (nx - state->food_x)
                       + (ny - state->food_y) * (ny - state->food_y);
    }
    if (can_left) {
        SnakeDirection ld = snake_turn_to_absolute(state->direction, SNAKE_TURN_LEFT);
        int nx = hx + snake_dir_dx(ld);
        int ny = hy + snake_dir_dy(ld);
        left_score = (nx - state->food_x) * (nx - state->food_x)
                   + (ny - state->food_y) * (ny - state->food_y);
    }
    if (can_right) {
        SnakeDirection rd = snake_turn_to_absolute(state->direction, SNAKE_TURN_RIGHT);
        int nx = hx + snake_dir_dx(rd);
        int ny = hy + snake_dir_dy(rd);
        right_score = (nx - state->food_x) * (nx - state->food_x)
                    + (ny - state->food_y) * (ny - state->food_y);
    }

    /* Priority 1: avoid danger */
    if (!can_straight && !can_left && !can_right) {
        return SNAKE_TURN_STRAIGHT; /* doomed, no safe moves */
    }

    /* Priority 2: among safe moves, prefer the one closest to food */
    if (can_straight) {
        if (can_left && left_score <= straight_score) {
            if (can_right && right_score <= straight_score && right_score <= left_score) {
                /* all safe, pick best */
                if (right_score <= left_score && right_score <= straight_score) {
                    return SNAKE_TURN_RIGHT;
                }
                if (left_score <= right_score && left_score <= straight_score) {
                    return SNAKE_TURN_LEFT;
                }
                return SNAKE_TURN_STRAIGHT;
            } else {
                if (left_score <= straight_score) return SNAKE_TURN_LEFT;
                return SNAKE_TURN_STRAIGHT;
            }
        } else if (can_right && right_score <= straight_score) {
            return SNAKE_TURN_RIGHT;
        } else {
            return SNAKE_TURN_STRAIGHT;
        }
    }
    if (can_left) {
        if (can_right && right_score < left_score) return SNAKE_TURN_RIGHT;
        return SNAKE_TURN_LEFT;
    }
    if (can_right) {
        return SNAKE_TURN_RIGHT;
    }

    return SNAKE_TURN_STRAIGHT; /* fallback */
}

/* --- Generate one-hot target for teacher action --- */
static inline void snake_teacher_target(const SnakeState* state, float* target) {
    int i;
    for (i = 0; i < SNAKE_OUTPUT_SIZE; ++i) target[i] = 0.0f;

    SnakeTurn turn = snake_teacher_turn(state);
    SnakeDirection absolute = snake_turn_to_absolute(state->direction, turn);
    target[(int)absolute] = 1.0f;
}

/* --- Place food on a random empty cell --- */
static inline void snake_place_food(SnakeState* state, unsigned int* rng) {
    int attempts = 0;

    /* Try random placement if rng is provided */
    if (rng != NULL) {
        while (attempts < 1000) {
            int fx = 1 + snake_rand_int(rng, SNAKE_GRID_W - 2);
            int fy = 1 + snake_rand_int(rng, SNAKE_GRID_H - 2);
            if (state->grid[fy][fx] == 0) {
                state->food_x = fx;
                state->food_y = fy;
                state->grid[fy][fx] = 3;
                return;
            }
            attempts++;
        }
    }

    /* Fallback: scan for any empty cell */
    {
        int y, x;
        for (y = 1; y < SNAKE_GRID_H - 1; y++) {
            for (x = 1; x < SNAKE_GRID_W - 1; x++) {
                if (state->grid[y][x] == 0) {
                    state->food_x = x;
                    state->food_y = y;
                    state->grid[y][x] = 3;
                    return;
                }
            }
        }
    }
}

/* --- Initialize a random snake game --- */
static inline void snake_init_random(SnakeState* state, unsigned int* rng) {
    int i;

    memset(state, 0, sizeof(SnakeState));

    /* Draw walls */
    {
        int y, x;
        for (y = 0; y < SNAKE_GRID_H; y++) {
            for (x = 0; x < SNAKE_GRID_W; x++) {
                if (y == 0 || y == SNAKE_GRID_H - 1 || x == 0 || x == SNAKE_GRID_W - 1) {
                    state->grid[y][x] = 1;
                }
            }
        }
    }

    /* Place snake head in random open area */
    {
        int hx = 2 + snake_rand_int(rng, SNAKE_GRID_W - 4);
        int hy = 2 + snake_rand_int(rng, SNAKE_GRID_H - 4);
        state->body_x[0] = hx;
        state->body_y[0] = hy;
    }

    /* Pick random initial direction */
    state->direction = (SnakeDirection)(snake_rand_int(rng, 4));

    /* Grow body opposite to direction */
    {
        int dx = -snake_dir_dx(state->direction);
        int dy = -snake_dir_dy(state->direction);
        for (i = 1; i < SNAKE_INIT_LENGTH; i++) {
            state->body_x[i] = state->body_x[i - 1] + dx;
            state->body_y[i] = state->body_y[i - 1] + dy;
        }
    }
    state->length = SNAKE_INIT_LENGTH;

    /* Mark grid */
    state->grid[state->body_y[0]][state->body_x[0]] = 4; /* head */
    for (i = 1; i < state->length; i++) {
        state->grid[state->body_y[i]][state->body_x[i]] = 2; /* body */
    }

    snake_place_food(state, rng);

    state->game_over = 0;
    state->score = 0;
    state->steps = 0;
}

/* --- Advance game by one step using a given absolute direction --- */
static inline void snake_step(SnakeState* state, SnakeDirection dir) {
    int new_x, new_y, i;

    if (state->game_over) return;

    state->steps++;

    new_x = state->body_x[0] + snake_dir_dx(dir);
    new_y = state->body_y[0] + snake_dir_dy(dir);

    /* Check wall collision */
    if (new_x < 1 || new_x >= SNAKE_GRID_W - 1 || new_y < 1 || new_y >= SNAKE_GRID_H - 1) {
        state->game_over = 1;
        return;
    }

    /* Check self collision (except tail which will move) */
    if (state->grid[new_y][new_x] == 2) {
        /* Check if it's the tail (which will move away) */
        int eating_food = (new_x == state->food_x && new_y == state->food_y);
        if (!eating_food) {
            /* Only check against body segments that won't move */
            /* Tail is at body_x[length-1], body_y[length-1] */
            if (!(new_x == state->body_x[state->length - 1] &&
                  new_y == state->body_y[state->length - 1])) {
                state->game_over = 1;
                return;
            }
        }
    }

    /* Check max steps */
    if (state->steps >= SNAKE_MAX_STEPS) {
        state->game_over = 1;
        return;
    }

    /* Move head */
    state->grid[state->body_y[0]][state->body_x[0]] = 2; /* old head -> body */
    state->body_x[0] = new_x;
    state->body_y[0] = new_y;
    state->grid[new_y][new_x] = 4; /* new head */

    state->direction = dir;

    /* Check food */
    if (new_x == state->food_x && new_y == state->food_y) {
        state->score++;
        /* Grow: shift all body positions down */
        for (i = state->length; i > 0; i--) {
            state->body_x[i] = state->body_x[i - 1];
            state->body_y[i] = state->body_y[i - 1];
        }
        state->length++;
        snake_place_food(state, NULL);
    } else {
        /* Move tail */
        int tail_x = state->body_x[state->length - 1];
        int tail_y = state->body_y[state->length - 1];
        state->grid[tail_y][tail_x] = 0; /* clear tail */

        /* Shift body positions (head already updated) */
        for (i = state->length - 1; i > 0; i--) {
            state->body_x[i] = state->body_x[i - 1];
            state->body_y[i] = state->body_y[i - 1];
        }
    }
}

/* --- Render game to terminal --- */
static inline void snake_render(const SnakeState* state, const float* nn_output) {
    int y, x;

    /* Clear screen and move cursor home */
    printf("\033[2J\033[H");

    /* Grid */
    for (y = 0; y < SNAKE_GRID_H; y++) {
        for (x = 0; x < SNAKE_GRID_W; x++) {
            switch (state->grid[y][x]) {
                case 1: printf("#"); break;
                case 2: printf("o"); break;
                case 3: printf("*"); break;
                case 4: printf("O"); break;
                default: printf("."); break;
            }
        }
        printf("\n");
    }

    printf("Score: %d  Steps: %d", state->score, state->steps);
    if (state->game_over) {
        printf("  GAME OVER");
    }
    printf("\n");

    /* NN output */
    printf("NN: Up=%.3f Right=%.3f Down=%.3f Left=%.3f\n",
           nn_output[0], nn_output[1], nn_output[2], nn_output[3]);
}

/* --- Sleep milliseconds (cross-platform) --- */
static inline void snake_sleep_ms(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    usleep(ms * 1000);
#endif
}

/* --- Collect one training sample (features + one-hot target) --- */
static inline int snake_collect_sample(
    SnakeState* state,
    float* features,
    float* target)
{
    if (state->game_over) return 0;

    snake_encode_state(state, features);
    snake_teacher_target(state, target);

    /* Advance game by teacher action */
    SnakeTurn turn = snake_teacher_turn(state);
    SnakeDirection dir = snake_turn_to_absolute(state->direction, turn);
    snake_step(state, dir);

    return 1;
}

#endif /* SNAKE_SCENE_H */
