# Demo Full Flow

本文档描述当前仓库的完整流程：**配置图拓扑 -> 生成 spec -> 训练 -> 推理 -> 测试**。

## 1. 配置图拓扑

在 `src/include/config_user.h` 中配置：

- `NETWORK_GRAPH_NODES`
- `NETWORK_GRAPH_EDGES`
- `NETWORK_GRAPH_INPUT_NODE_ID`
- `NETWORK_GRAPH_OUTPUT_NODE_ID`

节点类型使用语义化枚举值（如 `NETWORK_TOPO_NODE_LINEAR`、`NETWORK_TOPO_NODE_RNN`、`NETWORK_TOPO_NODE_CNN`）。

## 2. 生成 network_def

```bash
cmake --build build --config Debug --target profiler
build/Debug/profiler.exe -o src/include/network_def.h
```

生成文件包含 `network_def_build_spec(NetworkSpec*)`，供训练和推理复用。

## 3. 训练

常用：

```bash
cmake --build build --config Debug --target sevenseg_train
```

`sevenseg_train` 会生成：

- `demo_vocab_sevenseg.txt`
- `demo_weights_sevenseg.bin`
- `demo_weights_sevenseg_export.c`
- `demo_network_sevenseg_functions.c`

## 4. 推理

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

- 不再保留旧模型分类接口；所有流程必须显式使用 `NetworkSpec`
- 不再使用“线性/transformer 二选一开关”作为配置入口
- 网络结构以图拓扑为单一事实来源
