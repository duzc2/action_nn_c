# Snake TUI Demo

A classic Snake game with a minimal neural network (MLP) that learns to play
via imitation of a teacher heuristic.

---

## How the Network Is Applied

The NN takes the current game state (8 features) and outputs 4 softmax
probabilities—one per move direction. The game picks `argmax(output)` as the
action. This happens **every game step** in a tight loop:

```
encode_state() → infer_auto_run() → argmax() → step() → render() → sleep
```

The network itself is tiny: 8→6→4 MLP, ~82 parameters. It never sees reward
signals directly. Instead it is pre-trained on **teacher demonstrations**
offline, then deployed as a frozen model during gameplay.

Code: `demo/snake/infer_main.c:72-93`

---

## Teacher Game — What It Is and Where It Comes From

The "teacher" is **not a separate program or dataset file**. It is a hand-written
deterministic heuristic in `snake_scene.h`. It knows the full game state
(wall locations, body positions, food position) and plays the game perfectly.

### Teacher algorithm (`snake_teacher_turn`, line 164)

Priority-ordered, no randomness:

1. **Safety check** — compute which of the three relative moves
   (STRAIGHT, LEFT, RIGHT) are safe (not a wall or body cell).
2. **Food alignment** — among safe moves, calculate squared distance to food.
   Pick the safe move that minimizes distance.
3. **Fallback** — if all moves are unsafe, go straight (doomed case).

The relative turn is converted to an absolute direction by
`snake_turn_to_absolute()`, then executed by `snake_step()`.

### How training data is generated

`train_main.c` runs the teacher **from scratch** for 1000 games:

```
for game in 0..999:
    snake_init_random()          // random board, random snake
    while not game_over:
        encode_state()           // extract 8 features → sample_input
        teacher_target()         // teacher action → one-hot target
        teacher_turn() → step()  // advance game with teacher's move
        collect sample
```

This produces ~250k (state, action) pairs from 1000 teacher-played games.
The teacher **never plays during inference**—it only exists to generate
training data.

Code: `demo/snake/train_main.c:83-101`

---

## File Map

| File | Role | ~Lines |
|---|---|---|
| `snake_scene.h` | Header-only game engine, teacher, encoding, render, RNG | 474 |
| `generate_main.c` | Define 8→[6]→4 MLP, invoke profiler codegen | ~130 |
| `train_main.c` | Play 1000 teacher games, collect samples, train MLP | ~177 |
| `infer_main.c` | Load weights, run NN inference loop with TUI | ~104 |
| `generate/CMakeLists.txt` | Link `profiler_core`, build `snake_generate` | ~15 |
| `train/CMakeLists.txt` | Link `nn_train_core`, build `snake_train` | ~42 |
| `infer/CMakeLists.txt` | Link `infer_core`, build `snake_infer` | ~42 |
| `run_demo.bat` | Windows 6-step pipeline (gen/train/infer) | ~70 |
| `run_demo.sh` | POSIX 6-step pipeline | ~40 |

All under `demo/snake/`.

---

## Demo Pipeline (3 Stages)

The project follows a strict 3-phase convention shared by all demos:

### Stage 1 — Generate (`snake_generate`)

`generate_main.c` describes the MLP architecture to the profiler framework.
The profiler validates the topology, computes hashes, and emits C source
files containing the full network implementation:

```
build/demo/snake/data/
  infer.c / infer.h             ← inference runtime
  train.c / train.h             ← training runtime
  weights_load.c / weights_load.h
  weights_save.c / weights_save.h
  tokenizer.c / tokenizer.h
  network_init.c / network_init.h
  network_metadata.h
```

These generated files are **not committed**—they are always built from the
network definition in `generate_main.c`.

Code: `demo/snake/generate_main.c:35-99`

### Stage 2 — Train (`snake_train`)

Compiles the generated code together with `train_main.c`. The training loop:

1. Play 1000 teacher games with seeded RNG, collecting `(state_features, one_hot_action)` pairs
2. Fisher-Yates shuffle all samples (`train_main.c:32-40`)
3. Run 80 full passes over the shuffled dataset
4. Report loss every 10 epochs
5. Write `build/demo/snake/data/weights.bin`

Training config: ADAM optimizer, MSE loss, lr=0.003, momentum=0.9,
weight_decay=0.0001, batch_size=1, seed=42.

Code: `demo/snake/train_main.c`

### Stage 3 — Infer (`snake_infer`)

Compiles the generated code with `infer_main.c`. The runtime loop:

1. `weights_load_from_file()` → restore trained parameters from `weights.bin`
2. `snake_init_random()` → random board and snake
3. **Game loop** (every 150ms):
   - `snake_encode_state()` → extract 8 floats from grid, positions, direction
   - `infer_auto_run()` → forward pass through MLP, get 4 softmax outputs
   - `snake_argmax()` → pick the direction with highest probability
   - `snake_step()` → advance game by one cell
   - `snake_render()` → ANSI clear + redraw grid, score, NN activations
4. Loop exits on `game_over` (wall/self collision or 500-step limit)

Code: `demo/snake/infer_main.c`

---

## Network Architecture

```
Input (8) ──→ Hidden (6, TANH) ──→ Output (4, SOFTMAX)
```

| Layer | Size | Activation | Parameters |
|-------|------|------------|------------|
| Input | 8 | — | — |
| Weight + Bias | 8→6 | TANH | 8×6 + 6 = 54 |
| Weight + Bias | 6→4 | SOFTMAX | 6×4 + 4 = 28 |
| **Total** | | | **82** |

**Training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimizer | ADAM (MLP_OPT_ADAM = 0) |
| Loss | MSE (MLP_LOSS_MSE = 0) |
| Learning rate | 0.003 |
| Momentum | 0.9 |
| Weight decay | 0.0001 |
| Batch size | 1 |
| Seed | 42 |

Code: `demo/snake/generate_main.c:41-73`

---

## Input Features (8 floats, range [-1, 1])

Extracted by `snake_encode_state()` in `snake_scene.h:127-161`.

| Idx | Feature | Formula |
|-----|---------|---------|
| 0 | danger_forward | 1.0 if cell ahead is wall/body, else 0.0 |
| 1 | danger_left | 1.0 if cell to left of heading is wall/body |
| 2 | danger_right | 1.0 if cell to right of heading is wall/body |
| 3 | food_dx | `(food_x - head_x) / (GRID_W - 1)` |
| 4 | food_dy | `(food_y - head_y) / (GRID_H - 1)` |
| 5 | dir_up | 1.0 if facing up, else 0.0 |
| 6 | dir_down | 1.0 if facing down, else 0.0 |
| 7 | dir_left | 1.0 if facing left, else 0.0 |

RIGHT direction is implicit: all three direction bits = 0.

**Output:** 4 softmax neurons → `argmax` picks UP(0), RIGHT(1), DOWN(2), LEFT(3).

---

## Game Mechanics

| Property | Value | Code |
|----------|-------|------|
| Grid size | 20×12 with border walls | `SNAKE_GRID_W/H` |
| Initial length | 3 | `SNAKE_INIT_LENGTH` |
| Max body length | 256 | `SNAKE_MAX_LENGTH` |
| Max steps per game | 500 | `SNAKE_MAX_STEPS` |
| Sleep per frame | 150ms | `snake_sleep_ms(150)` |

### Rendering Glyphs

| Char | Meaning |
|------|---------|
| `#` | Wall (border) |
| `O` | Snake head |
| `o` | Snake body |
| `*` | Food |
| `.` | Empty cell |

Rendering uses ANSI escape `\033[2J\033[H` to clear the terminal each frame.

Code: `snake_render()` in `snake_scene.h:415-442`

### Game State (`SnakeState`)

```c
typedef struct {
    int grid[12][20];          // cell values: 0=empty, 1=wall, 2=body, 3=food, 4=head
    int body_x[256];           // x positions of all segments (0 = head)
    int body_y[256];           // y positions of all segments
    int length;                // current snake length
    SnakeDirection direction;  // current heading
    int score;                 // food eaten
    int food_x, food_y;        // current food position
    int game_over;             // collision flag
    int steps;                 // step counter
    unsigned int rng;          // RNG state for food placement
} SnakeState;
```

Code: `snake_scene.h:48-60`

### Random Number Generator

Uses a simple LCG (same pattern as `nested_nav`):

```c
*state = (*state) * 1664525U + 1013904223U;
```

Code: `snake_next_random()` in `snake_scene.h:63-66`

---

## Training Data Statistics

| Metric | Value |
|--------|-------|
| Teacher games played | 1000 |
| Samples collected | ~250,000 |
| Shuffle method | Fisher-Yates |
| Epochs | 80 |
| Typical loss range | 0.033 → 0.015 |
| weights.bin size | ~387 bytes |

Code: `demo/snake/train_main.c`

---

## Build & Run

```bash
# Step 1: Generate network code
cmake -S demo/snake/generate -B build/demo/snake/generate
cmake --build build/demo/snake/generate --config Debug
./build/demo/snake/generate/Debug/snake_generate

# Step 2: Train (produces weights.bin)
cmake -S demo/snake/train -B build/demo/snake/train
cmake --build build/demo/snake/train --config Debug
./build/demo/snake/train/Debug/snake_train

# Step 3: Play (autonomous NN, TUI animation)
cmake -S demo/snake/infer -B build/demo/snake/infer
cmake --build build/demo/snake/infer --config Debug
./build/demo/snake/infer/Debug/snake_infer
```

Or use the batch/shell script:

```
demo\snake\run_demo.bat   (Windows)
demo/snake/run_demo.sh    (Linux/macOS)
```

---

## Typical Performance

| Run | Score |
|-----|-------|
| Best | 20+ |
| Average | 12-16 |
| Worst | 4-5 |

The NN faithfully imitates the teacher heuristic at ~80-90% fidelity.
Performance varies with random seed (starting position, initial direction,
food placement). Loss after 80 epochs is typically 0.014-0.017.
