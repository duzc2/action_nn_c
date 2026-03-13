# action_c

## 项目做什么

`action_c` 是一个 C99 控制决策样例工程。  
它把“命令文本 + 当前状态”转换成“当前帧动作”。

它适合以下问题：
- 文本指令驱动控制（例如 move left / move right fast）
- 小型实时控制决策（每帧调用一次模型）
- 需要 C99 落地和可移植代码的项目

它不适合以下问题：
- 大规模深度学习训练平台
- GPU/分布式训练
- 复杂强化学习训练系统

## 应用示例

- 机器人导航原型：命令 + 位姿误差 -> 本帧移动动作
- 游戏控制原型：命令 + 环境状态 -> 键鼠动作向量
- 工业控制原型：操作指令 + 传感器摘要 -> 执行器输出

## 系统如何运行

每一帧固定流程：
1. `tokenizer_encode`：文本命令转 token ids
2. 前向计算：token ids + state -> logits
3. `op_actuator`：logits 按通道激活映射为动作
4. `driver_stub_apply`：动作发送到平台层
5. 外部循环决定下一帧是否继续

说明：
- 模型只负责“单帧决策”
- 外部系统负责“任务生命周期”

## 快速运行

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build --target c99_full_demo
.\build\c99_full_demo.exe
```

## 文档

本项目只保留三份主文档：
- 项目总览：README（本文件）
- 完整用户手册：[user_manual.md](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/docs/user_manual.md)
- 开发维护手册：[developer_manual.md](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/docs/developer_manual.md)
