# MLP Network Type

本目录提供通用 MLP 网络类型实现。

包含内容：

- `mlp_config.h`：用户侧与生成代码共享的 MLP 配置类型。
- `mlp_layers.c/.h`：全连接层、激活函数与参数初始化。
- `mlp_infer_ops.c/.h`：推理上下文、前向传播、权重保存/加载。
- `mlp_train_ops.c/.h`：训练上下文、反向传播、参数更新、检查点能力。
- `nn_type_mlp_infer.c`：推理注册入口。
- `nn_type_mlp_train.c`：训练注册入口。

职责边界：

- MLP 算法逻辑保留在 `src/nn/types/mlp/`。
- profiler 只负责转发用户侧提供的具体类型配置与调用路径。
- 生成代码负责把用户配置的 `MlpConfig` / `MlpTrainConfig` 耦合到注册入口。
- 核心算法、前向传播、反向传播与权重读写逻辑不下沉到 profiler 生成代码。
