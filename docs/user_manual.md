# 用户手册

## 1. 你需要知道的三件事

- 网络结构由图拓扑配置决定
- profiler 负责把配置转成可执行规格
- 训练与推理都只接受 `NetworkSpec`

## 2. 环境准备

```bash
cmake -S . -B build
cmake --build build --config Debug
```

## 3. 配置网络

编辑 `src/include/config_user.h`：

- `NETWORK_GRAPH_NODES`
- `NETWORK_GRAPH_EDGES`
- `NETWORK_GRAPH_INPUT_NODE_ID`
- `NETWORK_GRAPH_OUTPUT_NODE_ID`

常见节点类型：

- `NETWORK_TOPO_NODE_LINEAR`
- `NETWORK_TOPO_NODE_TRANSFORMER_BLOCK`
- `NETWORK_TOPO_NODE_ATTENTION_HEAD`
- `NETWORK_TOPO_NODE_CNN`
- `NETWORK_TOPO_NODE_RNN`
- `NETWORK_TOPO_NODE_KNN`
- `NETWORK_TOPO_NODE_SELECT`
- `NETWORK_TOPO_NODE_MERGE`

## 4. 生成规格

```bash
cmake --build build --config Debug --target profiler
build/Debug/profiler.exe -o src/include/network_def.h
```

## 5. 运行示例

### sevenseg

```bash
cmake --build build --config Debug --target sevenseg_infer_bin
build/demo/sevenseg/Debug/sevenseg_infer_bin.exe
```

### step / goal

```bash
cmake --build build --config Debug --target step_demo
cmake --build build --config Debug --target goal_demo
```

## 6. 测试

```bash
ctest --test-dir build -C Debug --output-on-failure
```

## 7. 常见问题

- **Q: 程序提示找不到 sevenseg 数据文件？**  
  A: 优先运行 `sevenseg_train` 生成数据，或启动时传入 `demo/sevenseg/data`。

- **Q: 训练失败显示 spec 非法？**  
  A: 检查图节点是否有重复 id、边是否引用不存在节点、图是否有环、输入输出节点是否存在。
