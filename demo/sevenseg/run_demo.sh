#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ACTION_C_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
BUILD_ROOT="$ACTION_C_ROOT/build/demo/sevenseg"

echo "[sevenseg] step 1/6 configure + build generate"
cmake -S "$SCRIPT_DIR/generate" -B "$BUILD_ROOT/generate"
cmake --build "$BUILD_ROOT/generate" --config Debug

echo "[sevenseg] step 2/6 run generate"
"$BUILD_ROOT/generate/Debug/sevenseg_generate.exe"

echo "[sevenseg] step 3/6 configure + build train"
cmake -S "$SCRIPT_DIR/train" -B "$BUILD_ROOT/train"
cmake --build "$BUILD_ROOT/train" --config Debug

echo "[sevenseg] step 4/6 run train"
"$BUILD_ROOT/train/Debug/sevenseg_train.exe"

echo "[sevenseg] step 5/6 configure + build infer"
cmake -S "$SCRIPT_DIR/infer" -B "$BUILD_ROOT/infer"
cmake --build "$BUILD_ROOT/infer" --config Debug

echo "[sevenseg] step 6/6 run infer"
printf '3\nq\n' | "$BUILD_ROOT/infer/Debug/sevenseg_infer.exe"
