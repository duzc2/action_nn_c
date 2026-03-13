# action_c

一个纯 C99 的轻量级“文本指令 + 状态输入 → 动作输出”项目，包含：
- 训练数据准备
- 最小训练闭环
- 权重导出/加载（bin 与 C 源码）
- 外部循环驱动的逐帧推理控制
- 大规模自动化测试套件（含模型专项测试）

## 项目目标

本项目用于验证一条可落地的嵌入式友好链路：
1. 用 CSV 准备训练数据
2. 在 C99 环境训练简化模型
3. 导出可部署权重
4. 在外部主循环中逐帧调用模型推理并执行小动作

它不是通用深度学习框架，而是面向“可解释、可移植、可测试”的工程样例。

## 主要能力

- **纯 C99 实现**：不依赖第三方推理框架
- **模型全链路**：数据、训练、导出、加载、推理全部在仓库内闭环
- **外部循环控制**：模型只做单帧决策，循环由外部驱动
- **多平台封装**：包含 PC/ESP32 驱动封装入口
- **测试体系完整**：单元/集成/错误/边界/压力/模型专项（泛化、OOD、对抗、极限、预期错误）
- **实时日志**：测试进度、目的、参数、期望值、实际值全部落盘

## 目录结构

```text
src/
  core/        核心算子与兼容入口
  tokenizer/   tokenizer 主实现与运行时封装
  platform/    平台驱动主实现与 PC/ESP32 封装
  model/       模型前向实现
  train/       CSV 训练数据加载
  tools/       profiler / min_train_loop / c99_full_demo
test/
  src/         全量测试与模型专项测试
  include/     测试框架头文件
docs/
  user_manual.md
  developer_manual.md
```

## 快速开始

### 1) 配置与构建

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build
```

### 2) 运行完整 Demo（数据 → 训练 → 导出 → 加载 → 推理）

```powershell
.\build\c99_full_demo.exe
```

运行后会生成：
- `demo_train_data.csv`
- `demo_weights.bin`
- `demo_weights_export.c`

### 3) 运行全量测试套件

```powershell
.\build\full_test_suite.exe
```

测试日志输出到：
- `test/logs/test_run_YYYYMMDD_HHMMSS.log`

## 从“能跑”到“会用”

如果你不只是想运行现成 demo，而是要做自己的模型与控制链路，请先看这份实战手册：
- [`docs/usage_playbook.md`](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/docs/usage_playbook.md)

它覆盖完整用户路径：
- 怎么设计模型与 I/O
- 怎么配置参数
- 怎么用 profiler 预估资源
- 怎么准备数据与训练
- 怎么导出和部署
- 怎么做单步推理
- 怎么做“一条命令多步推理”的外部循环控制

## 可执行目标

- `profiler`：根据配置生成 `src/include/network_def.h`
- `min_train_loop`：最小训练闭环验证
- `c99_full_demo`：端到端完整演示
- `full_test_suite`：全量测试（含模型专项）

## 文档导航

- 用户手册：[`docs/user_manual.md`](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/docs/user_manual.md)
- 开发手册：[`docs/developer_manual.md`](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/docs/developer_manual.md)
- 实战手册：[`docs/usage_playbook.md`](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/docs/usage_playbook.md)
- Demo 说明：[`demo_full_flow.md`](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/demo_full_flow.md)
