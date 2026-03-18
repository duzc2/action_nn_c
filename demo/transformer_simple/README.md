# transformer_simple_demo

最小可运行 Transformer 示例（多输出节点同时生效）：

- 在程序内构建图拓扑 `INPUT -> LINEAR -> TRANSFORMER_BLOCK -> TRANSFORMER_BLOCK`
- 训练内存样本并导出二进制权重
- 加载权重后执行四个控制场景，逐输出节点打印期望/预测

运行：

```bash
cmake --build build --config Debug --target transformer_simple_demo
build/demo/transformer_simple/Debug/transformer_simple_demo.exe
```

场景说明：

- 这是“控制向量回归/判定”示例，不是“单标签分类”
- 4个输出节点分别对应 4 个执行通道：
  - `THROTTLE`
  - `BRAKE`
  - `TURN`（单轴转向：`-1=左`，`0=中`，`1=右`）
  - `AUX`（预留通道）
- 同一时刻多个输出节点可同时生效（例如 `go_left` 会让 `THROTTLE=1` 且 `TURN=-1`）

输出解读：

- `epoch=... avg_loss=...`：训练误差，越小越好
- `scenario[k] ...`：一个控制场景
- `expected/predicted`：每个输出节点的目标与预测；`TURN` 通道输出是方向值而非开关值
- `summary: scenario_pass / channel_pass`：按场景与按节点两种通过统计
- `demo结论...`：本示例要点总结
