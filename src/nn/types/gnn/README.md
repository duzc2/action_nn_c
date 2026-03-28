# gnn

这个目录提供仓库中的第一个轻量 GNN 网络类型实现。

它的定位是：

- 作为 `src/nn/types/` 下的**通用图网络叶子能力**
- 通过 CMake 开关启用
- 通过推理/训练注册表接入 profiler 主流程
- 支持组合网络中的前向、反向、保存、加载一致性
- 由 demo 展示其用法，但**不把场景语义硬编码进后端**

当前版本刻意保持轻量，重点先放在“能稳定接入工程主链路”，而不是一次性覆盖完整通用 GNN 家族。

当前已提供的通用能力：

- 固定节点数图输入
- 固定槽位邻接表 `neighbor_index`
- 可选节点激活掩码 `node_mask_feature_index`
- 可选主/次锚点选择特征 `primary_anchor_feature_index` / `secondary_anchor_feature_index`
- `GNN_AGG_MEAN` 消息聚合
- 两类读出：
  - `GNN_READOUT_GRAPH_POOL`
  - `GNN_READOUT_ANCHOR_SLOTS`

目录职责：

- `gnn_config.h`：用户侧 / profiler 共享的 config-only 类型
- `gnn_infer_ops.*`：推理实现
- `gnn_train_ops.*`：训练实现
- `nn_type_gnn_infer.c`：推理注册桥接
- `nn_type_gnn_train.c`：训练注册桥接

说明：

- `open/current/target` 之类含义不属于 GNN 后端内建语义
- 这些语义应由调用方通过节点特征编码自行定义
- `demo/road_graph_nav` 只是一个“图结构导航场景”的使用示例
