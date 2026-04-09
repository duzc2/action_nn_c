#!/usr/bin/env bash
set -euo pipefail

ACTION_C_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_ROOT="$ACTION_C_ROOT/build/demo/mnist_cnn"
VCVARS="C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"

echo "[mnist_cnn] step 1/6 configure + build generate"
cmd /c "call \"$VCVARS\" && cmake -S \"$ACTION_C_ROOT/demo/mnist_cnn/generate\" -B \"$BUILD_ROOT/generate\" -G Ninja -DCMAKE_C_COMPILER=clang && cmake --build \"$BUILD_ROOT/generate\""

echo "[mnist_cnn] step 2/6 run generate"
"$BUILD_ROOT/generate/mnist_cnn_generate.exe"

echo "[mnist_cnn] step 3/6 configure + build train"
cmd /c "call \"$VCVARS\" && cmake -S \"$ACTION_C_ROOT/demo/mnist_cnn/train\" -B \"$BUILD_ROOT/train\" -G Ninja -DCMAKE_C_COMPILER=clang && cmake --build \"$BUILD_ROOT/train\""

echo "[mnist_cnn] step 4/6 run train"
"$BUILD_ROOT/train/mnist_cnn_train.exe"

echo "[mnist_cnn] step 5/6 configure + build infer"
cmd /c "call \"$VCVARS\" && cmake -S \"$ACTION_C_ROOT/demo/mnist_cnn/infer\" -B \"$BUILD_ROOT/infer\" -G Ninja -DCMAKE_C_COMPILER=clang && cmake --build \"$BUILD_ROOT/infer\""

echo "[mnist_cnn] step 6/6 run infer"
"$BUILD_ROOT/infer/mnist_cnn_infer.exe"

echo "[mnist_cnn] demo completed successfully"
