# 用户手册

## 1. 这是什么

本项目是一个可执行控制链路示例：输入文本指令与状态向量，输出执行动作。  
你可以直接运行 Demo，观察从训练到推理的完整闭环。

适用场景：
- 机器人/小车动作控制原型验证
- 嵌入式控制策略验证
- C99 环境下模型工程流程教学

## 2. 环境要求

- Windows / Linux / macOS（示例以 Windows 为主）
- CMake >= 3.16
- Clang 或 GCC（推荐 clang）
- Ninja（可选，但推荐）

## 3. 快速运行

### 步骤 1：构建

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build
```

### 步骤 2：运行完整 Demo

```powershell
.\build\c99_full_demo.exe
```

Demo 会自动执行：
1. 生成训练 CSV
2. 加载数据并训练
3. 导出权重（bin + C 源码）
4. 重新加载权重
5. 执行推理
6. 执行外部循环逐帧控制（目标点示例：15,15）

如果你的目标是“自己设计和落地模型”，不是只运行现成 demo，请直接阅读：
- [`usage_playbook.md`](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/docs/usage_playbook.md)

## 4. 外部循环模式说明

项目采用“外部循环驱动、模型单帧决策”：
- 每一帧只调用一次模型推理
- 每一帧只执行一个小动作
- 是否继续下一帧由外部控制器判断

这意味着：
- 模型不会自己“while 循环”
- 你的主循环可随时接管停止、切换目标、插入安全逻辑

## 5. 输出文件说明

运行 Demo 后常见产物：
- `demo_train_data.csv`：训练数据
- `demo_weights.bin`：二进制权重
- `demo_weights_export.c`：可编译进固件/程序的权重 C 文件

运行测试后产物：
- `test/logs/test_run_*.log`：详细测试日志

## 6. 如何跑测试

```powershell
.\build\full_test_suite.exe
```

测试覆盖：
- 单元测试
- 正确性测试
- 错误测试
- 临界值测试
- 压力测试
- 集成测试
- 模型专项测试（泛化/OOD/对抗/极限/预期错误）

日志内容包含：
- 测试进度
- 测试内容
- 测试目的
- 测试参数
- 期望返回值
- 实际返回值
- 指标统计（如 mse/mae/命中率/恢复率）

## 7. 常见问题

### Q1：控制台中文乱码

先执行：

```powershell
chcp 65001
```

再运行程序。日志文件通常是正常 UTF-8。

### Q2：找不到可执行文件

请先确认构建成功并检查 `build/` 目录下是否有：
- `c99_full_demo.exe`
- `full_test_suite.exe`

### Q3：为什么动作是一帧一帧执行

这是故意设计：便于把控制权留给外部主循环，提升可控性与安全性。

## 8. 最佳实践

- 先跑 `c99_full_demo` 了解全流程，再接入你的业务主循环
- 用 `full_test_suite` 评估改动影响，避免回归
- 变更模型策略后，重点关注模型专项测试日志中的极限与负向测试项
