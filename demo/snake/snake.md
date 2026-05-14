# Snake TUI Demo

A classic Snake game with a minimal neural network (MLP) that learns to play
via imitation of a teacher heuristic.

## Game Description

- **Grid:** 20x12 with border walls (`#`)
- **Snake:** initial length 3, grows by 1 when eating food (`*`)
- **Scoring:** +1 per food eaten
- **Game over:** wall collision, self collision, or max 500 steps
- **Glyphs:** `#` = wall, `O` = head, `o` = body, `*` = food, `.` = empty

## Network Architecture

| Layer     | Size | Activation |
|-----------|------|------------|
| Input     | 8    | -          |
| Hidden 0  | 6    | TANH       |
| Output    | 4    | SOFTMAX    |

- **Parameters:** ~82 total
- **Training:** ADAM optimizer, MSE loss, lr=0.003, momentum=0.9, weight_decay=0.0001
- **Batch size:** 1, seed: 42

## Input Features (8 floats, [-1, 1])

| Index | Feature         | Description                          |
|-------|-----------------|--------------------------------------|
| 0     | danger_forward  | 1.0 if wall/body ahead, else 0.0    |
| 1     | danger_left     | 1.0 if wall/body to left, else 0.0  |
| 2     | danger_right    | 1.0 if wall/body to right, else 0.0 |
| 3     | food_dx         | Normalized x distance to food        |
| 4     | food_dy         | Normalized y distance to food        |
| 5     | dir_up          | 1.0 if facing up                     |
| 6     | dir_down        | 1.0 if facing down                   |
| 7     | dir_left        | 1.0 if facing left                   |

RIGHT direction is implicit (all direction bits = 0).

**Output:** 4 neurons (softmax) → UP(0), RIGHT(1), DOWN(2), LEFT(3). Use argmax.

## Teacher Heuristic

Priority-ordered deterministic rule:
1. Avoid danger (wall/body collisions) with highest priority
2. Among safe moves, prefer the one that minimizes distance to food
3. Fallback: go straight (even if dangerous, when no safe moves exist)

The heuristic outputs relative turns (STRAIGHT/LEFT/RIGHT) which are converted
to absolute directions based on the current heading.

## Dataset & Training

- 1000 teacher games collected into feature/target samples
- Fisher-Yates shuffle applied
- 80 epochs over all samples
- Loss typically decreases from ~0.25 to <0.10

## Build & Run

```bash
# Step 1: Generate network code
cmake -S demo/snake/generate -B build/demo/snake/generate
cmake --build build/demo/snake/generate --config Debug
./build/demo/snake/generate/Debug/snake_generate

# Step 2: Train
cmake -S demo/snake/train -B build/demo/snake/train
cmake --build build/demo/snake/train --config Debug
./build/demo/snake/train/Debug/snake_train

# Step 3: Play (autonomous NN)
cmake -S demo/snake/infer -B build/demo/snake/infer
cmake --build build/demo/snake/infer --config Debug
./build/demo/snake/infer/Debug/snake_infer
```

Or use the batch/shell script:
```
demo\snake\run_demo.bat   (Windows)
demo/snake/run_demo.sh    (Linux/macOS)
```
