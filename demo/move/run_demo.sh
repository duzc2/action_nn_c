#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ACTION_C_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
BUILD_ROOT="$ACTION_C_ROOT/build/demo/move"

echo "[move] step 1/6 configure + build generate"
cmake -S "$SCRIPT_DIR/generate" -B "$BUILD_ROOT/generate"
cmake --build "$BUILD_ROOT/generate" --config Debug

echo "[move] step 2/6 run generate"
"$BUILD_ROOT/generate/Debug/move_generate.exe"

echo "[move] step 3/6 configure + build train"
cmake -S "$SCRIPT_DIR/train" -B "$BUILD_ROOT/train"
cmake --build "$BUILD_ROOT/train" --config Debug

echo "[move] step 4/6 run train"
"$BUILD_ROOT/train/Debug/move_train.exe"

echo "[move] step 5/6 configure + build infer"
cmake -S "$SCRIPT_DIR/infer" -B "$BUILD_ROOT/infer"
cmake --build "$BUILD_ROOT/infer" --config Debug

echo "[move] step 6/6 run infer"
printf '5 5\n0\n0\n0\n4\n' | "$BUILD_ROOT/infer/Debug/move_infer.exe"
