# Transformer Network Type

本目录提供小型真实 Transformer 网络类型实现。

包含内容：

- `transformer_config.h`：用户侧与生成代码共享的 Transformer 配置类型。
- `transformer_infer_ops.c/.h`：字符级 tokenizer、嵌入、自注意力编码、响应分类推理。
- `transformer_train_ops.c/.h`：训练步骤、损失统计与参数更新。
- `nn_type_transformer_infer.c`：推理注册入口。
- `nn_type_transformer_train.c`：训练注册入口。

实现定位：

- 面向小参数量对话与分类场景。
- 保留真实参数、真实训练、真实权重保存/加载链路。
- profiler 只转发用户侧提供的 `TransformerModelConfig` / `TransformerTrainConfig`。
- 不把核心算法下沉到 profiler 生成代码中。
