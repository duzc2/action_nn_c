# Demo Full Flow

本文档描述当前仓库的完整流程：**三次编译 -> 一次生成运行 -> 训练 -> 推理 -> 测试**。

## 1. 启用网络类型（第一次编译前）

在 CMakeLists 开关启用网络类型：

```bash
cmake -S . -B build -DENABLE_NN_TRANSFORMER=ON -DENABLE_NN_SEVENSEG=ON
cmake --build build --config Debug --target profiler
```

## 2. 生成阶段（第一次运行）

```bash
build/Debug/profiler.exe
```

运行时由用户程序传入网络结构体，profiler 生成训练 `.c` 与推理 `.c`，并复制固定模板 `.h`。

## 3. 训练（第二次编译）

常用：

```bash
cmake --build build --config Debug --target sevenseg_train
```

`sevenseg_train` 会生成：

- `demo_vocab_sevenseg.txt`
- `demo_weights_sevenseg.bin`
- `demo_weights_sevenseg_export.c`
- `demo_network_sevenseg_functions.c`

## 4. 推理（第三次编译）

```bash
cmake --build build --config Debug --target sevenseg_infer_bin
build/demo/sevenseg/Debug/sevenseg_infer_bin.exe
```

也可运行：

- `sevenseg_infer_c_array`
- `sevenseg_infer_c_func`
- `goal_demo`
- `step_demo`

## 5. 基准与测试

```bash
cmake --build build --config Debug --target sevenseg_benchmark
build/demo/sevenseg/Debug/sevenseg_benchmark.exe

ctest --test-dir build -C Debug --output-on-failure
```

## 6. 约束

- 新增网络类型只允许通过注册配置/注册宏与 CMake 开关接入
- 不允许修改 profiler 主流程与旧网络实现来适配新类型
- 未启用网络类型不参与编译、注册、生成、训练与推理
