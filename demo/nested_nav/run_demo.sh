#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ACTION_C_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
BUILD_ROOT="$ACTION_C_ROOT/build/demo/nested_nav"

echo "[nested_nav] step 1/6 configure + build generate"
cmake -S "$SCRIPT_DIR/generate" -B "$BUILD_ROOT/generate"
cmake --build "$BUILD_ROOT/generate" --config Debug

echo "[nested_nav] step 2/6 run generate"
"$BUILD_ROOT/generate/Debug/nested_nav_generate.exe"

echo "[nested_nav] step 3/6 configure + build train"
cmake -S "$SCRIPT_DIR/train" -B "$BUILD_ROOT/train"
cmake --build "$BUILD_ROOT/train" --config Debug

echo "[nested_nav] step 4/6 run train"
"$BUILD_ROOT/train/Debug/nested_nav_train.exe"

echo "[nested_nav] step 5/6 configure + build infer"
cmake -S "$SCRIPT_DIR/infer" -B "$BUILD_ROOT/infer"
cmake --build "$BUILD_ROOT/infer" --config Debug

echo "[nested_nav] step 6/6 run infer"
"$BUILD_ROOT/infer/Debug/nested_nav_infer.exe"
