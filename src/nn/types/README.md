# Network Types

本目录存放网络类型源码实现。

当前仓库已实现并接入的类型：

- `mlp/`：多层感知机，可变层数，6种激活函数，SGD/Adam优化器。
- `cnn/`：卷积网络，标准/深度可分离卷积，4种池化模式，BN，Dropout。
- `cnn_dual_pool/`：CNN 变体，固定 avg+max 双池化（与 CNN 共用底层，通过 CMake 开关独立编译）。
- `rnn/`：Elman 式循环网络，tanh 隐藏激活，序列输入→控制向量输出。
- `gnn/`：消息传递图网络，mean 聚合器，支持图池化和锚点两种读出方式。
- `transformer/`：单头自注意力 Transformer，字符级 tokenizer，位置编码，文本 QA/分类。

扩展计划参见 `docs/network_architecture_roadmap.md`。

接入规则：

- 仅通过注册表与 CMake 开关启用。
- 训练实现可依赖推理实现。
- 推理实现不得依赖训练实现。
- 用户侧具体类型配置通过 profiler 透明转发到生成代码。
- 生成代码只调用注册接口，不解析具体类型配置字段。
