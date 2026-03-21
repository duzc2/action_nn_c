# action_c

纯 C99 的训练/推理工程，核心是**编译期开关 + 注册机制 + 用户结构体驱动生成**。  
系统面向可持续扩展：新增网络类型不改对外接口，不改 profiler 主流程，不改旧实现代码。

## 核心设计

- 网络类型启用：CMakeLists 开关（仅启用类型参与编译与注册）
- 用户输入：用户程序构建网络结构体并调用 profiler
- 生成阶段：profiler 生成训练/推理 `.c`，复制固定模板 `.h`
- 训练依赖：训练工程依赖推理 `.c` + 训练 `.c`
- 推理依赖：推理工程仅依赖推理 `.c`，可加载 `.bin` 或使用权重 `.c`

## 快速开始

```bash
cmake -S . -B build -DENABLE_NN_TRANSFORMER=ON -DENABLE_NN_SEVENSEG=ON
cmake --build build --config Debug --target profiler
build/Debug/profiler.exe
cmake --build build --config Debug --target sevenseg_train
cmake --build build --config Debug --target sevenseg_infer_bin
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
