# action_c

纯 C99 的训练/推理工程，核心是**图拓扑驱动的网络配置**。  
项目不再区分“线性模型接口 / transformer 模型接口”，统一由 `NetworkGraph -> NetworkSpec` 流程决定执行路径。

## 核心设计

- 网络配置来源：`src/include/config_user.h` 中 `NETWORK_GRAPH_NODES` 与 `NETWORK_GRAPH_EDGES`
- 结构构建入口：`network_graph_build` + `network_spec_build_from_graph`
- 推理统一入口：`workflow_runtime_init(..., const NetworkSpec* spec)`
- 训练统一入口：`workflow_train_from_csv` / `workflow_train_from_memory`（均要求 `network_spec`）
- profiler 负责生成 `src/include/network_def.h`，其中包含 `network_def_build_spec`

## 快速开始

```bash
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

## 常用可执行目标

- `profiler`
- `min_train_loop`
- `goal_demo`
- `step_demo`
- `transformer_simple_demo`
- `sevenseg_train`
- `sevenseg_infer_bin`
- `sevenseg_infer_c_array`
- `sevenseg_infer_c_func`
- `sevenseg_benchmark`
- `full_test_suite`

## sevenseg 示例

```bash
cmake --build build --config Debug --target sevenseg_train
cmake --build build --config Debug --target sevenseg_infer_bin
```

运行：

```bash
build/demo/sevenseg/Debug/sevenseg_infer_bin.exe
```

程序支持自动解析数据目录；也可显式传入：

```bash
build/demo/sevenseg/Debug/sevenseg_infer_bin.exe demo/sevenseg/data
```

## transformer 简易示例

```bash
cmake --build build --config Debug --target transformer_simple_demo
build/demo/transformer_simple/Debug/transformer_simple_demo.exe
```

## 文档索引

- `docs/user_manual.md`：使用手册
- `docs/developer_manual.md`：开发手册
- `docs/network_design_manual.md`：图拓扑网络设计手册
- `demo_full_flow.md`：端到端流程演示
