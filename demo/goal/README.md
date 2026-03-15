# Goal Demo

这是“目标驱动闭环”演示。

## 这个 Demo 在做什么

- 先在代码内准备“单一目标命令 + 目标状态样本”的训练数据
- 使用训练集训练一个专门用于 goal 闭环的模型参数
- 启动推理运行时后，循环执行“状态输入 → 单步推理 → 位姿更新”
- 在 CLI 网格中显示当前位置 `@` 与目标点 `G`
- 推理阶段随机采样目标点与最大步数，逐段执行闭环

它展示的是“外部控制循环 + 模型作为策略近似器”的最小闭环落地过程。

## 怎么实现的

核心实现入口是 `demo/goal/main.c`，流程如下：

1. 数据准备  
   - 在代码内构造训练样本（命令采用参数格式：`goal x y`）
   - 生成 `demo_vocab_goal.txt`

2. 训练与导出  
   - 调用 `workflow_train_from_memory(...)` 完成训练
   - 导出 `bin` 与 `c` 权重文件
   - 额外导出函数网络文件用于静态集成验证

3. 推理运行时初始化  
   - 调用 `workflow_runtime_init(...)` 加载词表和权重

4. 随机目标闭环  
   - 每段开始随机一个目标点 `(x,y)` 与最大步数 `max_steps`
   - 按参数命令格式生成 `goal x y`
   - `run_goal_loop(...)` 中根据当前位置与目标计算剩余量
   - 调用 `workflow_run_step(...)`，命令形如 `goal 15 4`
   - 根据动作更新 `pose`，并刷新 CLI 网格画面
   - 每按一次回车推进一帧，便于人工观察
   - 到达目标或走满 `max_steps` 后，立即切换到下一随机目标

## 设计思路

- 目标是验证“给目标、模型逐步逼近”的工程路径，不是只看单次前向
- 文本命令使用参数形式（`goal x y`），`x/y` 为坐标参数
- 状态包含目标相关上下文（remain_x/remain_y 等）
- 通过“随机目标 + 步数上限”验证模型在多段任务中的连续控制能力

## 入口与构建

- 代码入口：`demo/goal/main.c`
- 本地目标名：`goal_demo`

在仓库根目录执行：

```bash
cmake -S . -B build
cmake --build build --target goal_demo
```

## 运行

Windows:

```bash
.\build\demo\goal\Debug\goal_demo.exe
```

## 运行产物

运行时会写入到 `demo/goal/data/`：
- `demo/goal/data/demo_vocab_goal.txt`
- `demo/goal/data/demo_weights_goal.bin`
- `demo/goal/data/demo_weights_goal_export.c`
- `demo/goal/data/demo_network_goal_functions.c`

## 行为说明

- 这是“随机目标段落式”的模式，核心循环在 `run_goal_loop`
- CLI 会显示网格图形，`@` 为当前点，`G` 为目标点
- 每按一次回车，计算并渲染下一帧
- 每段在“到达目标”或“达到最大步数”后切换下一个随机目标
