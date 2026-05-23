#!/usr/bin/env bash
set -euo pipefail

ACTION_C_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_ROOT="${ACTION_C_ROOT}/build/demo/edge_video_preprocess"
GENERATED_DIR="${BUILD_ROOT}/data"

echo "============================================================"
echo "Edge Video Preprocessing Demo - Full Pipeline"
echo "============================================================"
echo ""

# Step 1: Data preparation
echo "[edge_video_preprocess] step 1/7 run data preparation"
echo "------------------------------------------------------------"
python3 "$(dirname "${BASH_SOURCE[0]}")/data_prep.py" "$@" || {
    echo "Data preparation failed"
    exit 1
}
echo ""

# Step 2: Configure + build generate
echo "[edge_video_preprocess] step 2/7 configure + build generate"
echo "------------------------------------------------------------"
cmake -S "${ACTION_C_ROOT}/demo/edge_video_preprocess/generate" \
      -B "${BUILD_ROOT}/generate" \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
    && cmake --build "${BUILD_ROOT}/generate" \
    || { echo "Generate build failed"; exit 1; }
echo ""

# Step 3: Run generate
echo "[edge_video_preprocess] step 3/7 run generate"
echo "------------------------------------------------------------"
"${BUILD_ROOT}/generate/edge_video_preprocess_generate" || {
    echo "Code generation failed"
    exit 1
}
echo ""

# Step 4: Configure + build train
echo "[edge_video_preprocess] step 4/7 configure + build train"
echo "------------------------------------------------------------"
cmake -S "${ACTION_C_ROOT}/demo/edge_video_preprocess/train" \
      -B "${BUILD_ROOT}/train" \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DACTION_C_GENERATED_DIR="${GENERATED_DIR}" \
    && cmake --build "${BUILD_ROOT}/train" \
    || { echo "Train build failed"; exit 1; }
echo ""

# Step 5: Run train
echo "[edge_video_preprocess] step 5/7 run train"
echo "------------------------------------------------------------"
"${BUILD_ROOT}/train/edge_video_preprocess_train" || {
    echo "Training failed"
    exit 1
}
echo ""

# Step 6: Configure + build infer
echo "[edge_video_preprocess] step 6/7 configure + build infer"
echo "------------------------------------------------------------"
cmake -S "${ACTION_C_ROOT}/demo/edge_video_preprocess/infer" \
      -B "${BUILD_ROOT}/infer" \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DACTION_C_GENERATED_DIR="${GENERATED_DIR}" \
    && cmake --build "${BUILD_ROOT}/infer" \
    || { echo "Infer build failed"; exit 1; }
echo ""

# Step 7: Run infer
echo "[edge_video_preprocess] step 7/7 run infer"
echo "------------------------------------------------------------"
"${BUILD_ROOT}/infer/edge_video_preprocess_infer" || {
    echo "Inference failed"
    exit 1
}
echo ""

echo "============================================================"
echo "[edge_video_preprocess] demo completed successfully"
echo "============================================================"
