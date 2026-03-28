#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ACTION_C_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
BUILD_ROOT="$ACTION_C_ROOT/build/demo/road_graph_nav"

COMMON_ARGS='-G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang'

echo "[road_graph_nav] step 1/6 configure + build generate"
cmake -S "$SCRIPT_DIR/generate" -B "$BUILD_ROOT/generate" $COMMON_ARGS
cmake --build "$BUILD_ROOT/generate"

echo "[road_graph_nav] step 2/6 run generate"
"$BUILD_ROOT/generate/road_graph_nav_generate"

echo "[road_graph_nav] step 3/6 configure + build train"
cmake -S "$SCRIPT_DIR/train" -B "$BUILD_ROOT/train" $COMMON_ARGS
cmake --build "$BUILD_ROOT/train"

echo "[road_graph_nav] step 4/6 run train"
"$BUILD_ROOT/train/road_graph_nav_train"

echo "[road_graph_nav] step 5/6 configure + build infer"
cmake -S "$SCRIPT_DIR/infer" -B "$BUILD_ROOT/infer" $COMMON_ARGS
cmake --build "$BUILD_ROOT/infer"

echo "[road_graph_nav] step 6/6 run infer"
"$BUILD_ROOT/infer/road_graph_nav_infer"
