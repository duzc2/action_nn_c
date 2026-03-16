# Step Demo

这是“单步指令推理 + CLI 可视化”演示。

## 这个 Demo 在做什么

- 先生成一份用于“逐步指令响应任务”的训练集与词表
- 训练一个专门用于 step 场景的模型参数
- 按“指令段”执行：同一条指令连续保持多帧，再切换下一条
- 将输出动作映射为坐标位移，并在终端网格实时显示点位变化

它展示的是“命令驱动的离散步进控制”，不包含目标收敛逻辑。

## 怎么实现的

核心实现入口是 `demo/step/main.c`，流程如下：

1. 数据准备  
- 在代码内程序化生成大量训练样本（6x6 状态网格 × 多指令）
   - 生成 `demo_vocab_step.txt`

2. 训练与导出  
   - 调用 `workflow_train_from_memory(...)` 训练 step 模型
   - 导出 `bin` 与 `c` 权重文件
   - 导出函数网络文件用于静态集成验证

3. 推理运行时初始化  
   - 调用 `workflow_runtime_init(...)` 载入词表与权重

4. 单步可视化循环  
   - `build_step_training_samples(...)` 生成更大规模训练集
   - 对每条样本执行一致性校验（命令方向与 target 符号严格匹配）
   - `run_step_draw_mode(...)` 随机选指令并保持 6~12 帧
   - 调用 `workflow_run_step(...)` 得到动作输出
   - 直接使用模型输出动作更新位姿（仅做步长与边界裁剪）
   - 更新 `pose` 后重绘网格
   - 按一次回车推进一帧，便于人工检查

## 设计思路

- 目标是验证“按指令段执行”的 step 用法
- 状态输入仅保留位姿基础信息，不引入目标字段
- 训练样本覆盖横向、纵向、斜向、快慢速，并扩大状态覆盖范围
- 训练前严格校验样本正确性，避免“横向命令带纵向位移”等脏数据
- 终端渲染采用低依赖字符画，保证在普通 CLI 环境即可观察轨迹

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
