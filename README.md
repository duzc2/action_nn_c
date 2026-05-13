

# action_nn_c

Neural Network Library in C with Code Generation

## 项目简介

action_nn_c 是一个纯 C 实现的神经网络库，具有代码生成（profiler）功能。该项目支持多种网络拓扑结构，包括 MLP、CNN、RNN、GNN 和 Transformer，可用于嵌入式系统和游戏 AI 等场景。

## 功能特性

- **多种网络类型支持**
  - MLP (多层感知机)
  - CNN (卷积神经网络，含 Dual Pool 模式)
  - RNN (循环神经网络)
  - GNN (图神经网络)
  - Transformer (注意力机制网络)

- **完整训练流程**
  - generate: 网络定义生成
  - train: 模型训练
  - infer: 模型推理

- **代码生成系统 (profiler)**
  - 自动网络拓扑分析
  - 代码生成与权重管理
  - 训练与推理运行时支持

## 目录结构

```
action_nn_c/
├── src/
│   ├── nn/                 # 神经网络核心实现
│   │   └── types/          # 网络类型实现
│   │       ├── mlp/        # MLP 实现
│   │       ├── cnn/        # CNN 实现 (含 Dual Pool 模式)
│   │       ├── gnn/        # GNN 实现
│   │       ├── rnn/        # RNN 实现
│   │       └── transformer/
│   ├── profiler/           # 代码生成器
│   ├── train/              # 训练运行时
│   ├── utils/              # 工具库 (arena, log, error)
│   └── CMakeLists.txt
├── tests/                  # 测试套件 (23 个测试)
│   ├── CMakeLists.txt
│   ├── test_harness.h      # 轻量测试框架
│   ├── utils/              # 工具库测试
│   ├── nn/                 # NN 核心测试
│   ├── profiler/           # Profiler 测试
│   ├── integration/        # 集成测试
│   └── demo/               # Demo 回归测试
├── demo/                   # 示例项目
│   ├── mnist/              # MNIST 手写数字识别
│   ├── mnist_cnn/          # MNIST CNN 版本
│   ├── move/               # 移动控制 demo
│   ├── target/             # 目标追踪 demo
│   ├── sevenseg/           # 七段显示识别
│   ├── nested_nav/         # 嵌套导航
│   ├── road_graph_nav/     # 道路图导航
│   ├── cnn_rnn_react/      # CNN+RNN 反应式控制
│   ├── hybrid_route/       # 混合路由
│   ├── transformer/        # Transformer 对话
│   └── cs/                 # CS 游戏 Demo
├── web_editor/             # 可视化网络编辑器 (Vite)
├── wasm/                   # WebAssembly 导出
├── scripts/                # 构建脚本
├── cmake/                  # CMake 模块
└── docs/                   # 开发文档
```

## 快速开始

### 构建与测试

使用 CMake 预设 (debug / release / sanitized):

```bash
# Debug 构建
cmake --preset debug
cmake --build --preset debug

# 运行测试套件
ctest --preset debug

# Release 构建
cmake --preset release
cmake --build --preset release

# Sanitizer 构建 (需 clang, ASan + UBSan)
cmake --preset sanitized
cmake --build --preset sanitized
ctest --preset sanitized

# 注意: sanitized 构建需将 ASan DLL 加入 PATH
# export PATH="$(llvm-config --libdir)/clang/$(llvm-config --version | cut -d. -f1)/lib/windows:$PATH"
```

### 运行 Demo

使用统一脚本:

```bash
# 运行单个 demo
bash scripts/run_demo.sh move

# 指定阶段 (generate / train / infer)
bash scripts/run_demo.sh move generate
```

### Demo 手动构建流程

每个 demo 项目都遵循相同的流程:

```bash
# 步骤1: 配置并编译生成器
mkdir -p build/generate && cd build/generate
cmake ../../demo/xxx -G "Unix Makefiles"
cmake --build .

# 步骤2: 运行生成器
./generate/xxx_generate

# 步骤3-6: 类似编译并运行 train / infer
```

## 网络类型

### MLP (多层感知机)

全连接神经网络，支持多种激活函数（ReLU、Sigmoid、Tanh、Leaky ReLU），适用于各种分类和回归任务。

### CNN (卷积神经网络)

包含卷积层和池化层，适合图像特征提取。

### RNN (循环神经网络)

处理时序数据，适合序列预测和时间序列分析。

### GNN (图神经网络)

基于图结构的神经网络，适合路网导航和图分析任务。

### Transformer

自注意力机制网络，适合对话和文本处理任务。

## Demo 说明

| Demo | 说明 | 网络类型 |
|------|------|----------|
| mnist | 手写数字识别 | MLP |
| mnist_cnn | 手写数字识别 (CNN版) | CNN |
| move | 移动控制 | MLP |
| target | 目标追踪 | MLP |
| sevenseg | 七段显示识别 | MLP |
| nested_nav | 嵌套导航 | MLP |
| road_graph_nav | 道路图导航 | GNN |
| cnn_rnn_react | 反应式控制 | CNN+RNN |
| hybrid_route | 混合路由 | Transformer |
| transformer | 对话示例 | Transformer |
| cs | CS 游戏 Demo | CNN+RNN |

## 开发指南

详细开发流程请参考 `docs/developer_manual.md`。

### 新增网络类型

1. 在 `src/nn/types/` 下创建新的网络类型目录
2. 实现推理相关函数
3. 在 `nn_infer_registry.h` 中注册网络类型
4. 更新 CMakeLists.txt

## 文档

- [用户手册](docs/user_manual.md)
- [开发手册](docs/developer_manual.md)
- [网络设计手册](docs/network_design_manual.md)
- [profiler 开发计划](docs/profiler_development_plan.md)

## License

见 LICENSE 文件。