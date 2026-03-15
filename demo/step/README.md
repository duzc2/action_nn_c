# Step Demo

这是“单步指令推理 + CLI 可视化”演示。

## 这个 Demo 在做什么

- 先生成一份用于“逐步指令响应任务”的训练集与词表
- 训练一个专门用于 step 场景的模型参数
- 每帧输入一条文本指令，模型只做一步推理
- 将输出动作映射为坐标位移，并在终端网格实时显示点位变化

它展示的是“命令驱动的离散步进控制”，不包含目标收敛逻辑。

## 怎么实现的

核心实现入口是 `demo/step/main.c`，流程如下：

1. 数据准备  
   - 在代码内构造训练样本（内存样本数组）
   - 生成 `demo_vocab_step.txt`

2. 训练与导出  
   - 调用 `workflow_train_from_memory(...)` 训练 step 模型
   - 导出 `bin` 与 `c` 权重文件
   - 导出函数网络文件用于静态集成验证

3. 推理运行时初始化  
   - 调用 `workflow_runtime_init(...)` 载入词表与权重

4. 单步可视化循环  
   - `run_step_draw_mode(...)` 每帧随机选一条指令
   - 调用 `workflow_run_step(...)` 得到动作输出
   - 更新 `pose` 后重绘网格
   - 按一次回车推进一帧，便于人工检查

## 设计思路

- 目标是验证“每步一条指令”的纯 step 用法
- 状态输入仅保留位姿基础信息，不引入目标字段
- 终端渲染采用低依赖字符画，保证在普通 CLI 环境即可观察轨迹
- 手动逐帧推进便于调试：可以逐条检查命令与动作响应是否符合预期

## 入口与构建

- 代码入口：`demo/step/main.c`
- 本地目标名：`step_demo`

在仓库根目录执行：

```bash
cmake -S . -B build
cmake --build build --target step_demo
```

## 运行

Windows:

```bash
.\build\demo\step\Debug\step_demo.exe
```

## 运行产物

运行时会写入到 `demo/step/data/`：
- `demo/step/data/demo_vocab_step.txt`
- `demo/step/data/demo_weights_step.bin`
- `demo/step/data/demo_weights_step_export.c`
- `demo/step/data/demo_network_step_functions.c`

## 交互说明

- CLI 会显示 `CLI Step Mode` 网格
- 点位符号为 `@`
- 每按一次回车，计算并渲染下一帧

提示：
- 这是“无目标点”的模式
- 不做闭环目标收敛，只做逐步动作可视化
