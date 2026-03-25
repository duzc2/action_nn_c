# MLP Network Type

本目录提供 demo 所使用的真实 MLP 实现。

包含内容：

- `mlp_layers.c/.h`：全连接层、激活函数与参数初始化。
- `mlp_infer_ops.c/.h`：推理上下文、前向传播、权重保存/加载。
- `mlp_train_ops.c/.h`：训练上下文、反向传播、参数更新、检查点能力。
- `nn_type_mlp_infer.c`：推理注册入口。
- `nn_type_mlp_train.c`：训练注册入口。

职责边界：

- MLP 算法逻辑保留在 `src/nn/types/mlp/`。
- profiler 生成代码只负责把网络定义转成统一调用路径。
- demo 的 move / sevenseg / target 都通过这里的统一 MLP 实现运行。
