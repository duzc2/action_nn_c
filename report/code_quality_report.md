# 代码质量检查报告

日期：2026-03-29

## 1. 检查范围

- 依据：`docs/` 全部文档，重点对齐以下要求：
  - `docs/user_manual.md`
  - `docs/developer_manual.md`
  - `docs/network_topology_requirements.md`
  - `docs/network_design_manual.md`
  - `docs/profiler_api_contract_draft.md`
  - `docs/profiler_interface_draft.md`
  - `docs/profiler_development_plan.md`
- 本次同步重点：
  - `src/nn/types/gnn/*`
  - `demo/road_graph_nav/*`
  - `src/CMakeLists.txt`
  - `src/nn/CMakeLists.txt`
  - `demo/demo_common.cmake`
  - `src/nn/types/mlp/mlp_train_ops.c`
- 检查方式：
  - 静态代码审查
  - 目录/构建脚本一致性检查
  - 简单量化统计（注释占比、网络类型覆盖）
- 按用户要求，本次**未做编译、运行或测试执行**。

## 2. 总体结论

当前代码状态相比上一版报告，已经需要按下面的新口径同步：

- 仓库已新增一个受 CMake 开关控制、可注册接入主流程的 `gnn` 网络类型
- `gnn` 的配置语义已从业务特化字段，收敛为更通用的图网络字段：
  - 节点活动掩码
  - 主/次锚点特征
  - 聚合策略
  - 读出策略
- `road_graph_nav` 被实现为**通用 GNN + MLP 组合网络的一种使用示例**，而不是为 demo 特化的核心后端
- 现有 `target` demo 的自动演示脚本与地图可视化结论仍然成立

因此，当前状态更准确的结论应为：

> **核心构建门禁与 profiler 契约保持改善，网络类型能力从 4 种提升到 5 种，并新增了通用化 GNN 后端及其组合示例；但从文档验收口径看，注释密度与网络类型覆盖仍未达标。**

## 3. 量化结果

### 3.1 注释密度

按当前仓库静态统计：

- `src/`：67 个 `.c/.h` 文件，14594 行，注释行约 3482，注释占比约 **23.9%**
- `demo/`：32 个 `.c/.h` 文件，8133 行，注释行约 810，注释占比约 **10.0%**
- `src/ + demo/` 总体：99 个 `.c/.h` 文件，22727 行，注释行约 4292，注释占比约 **18.9%**

对照文档：

- `docs/network_topology_requirements.md`
- `docs/profiler_interface_draft.md`
- `docs/profiler_api_contract_draft.md`

上述文档都要求“源码与生成代码注释量不少于 50%”。当前静态统计仍不满足该要求。

### 3.2 网络类型覆盖

- 文档要求常见网络类型：**21** 种
- 当前 `src/nn/types/` 实际存在：`cnn`、`gnn`、`mlp`、`rnn`、`transformer`，共 **5** 种
- 仍缺失：`knn`、`rbfn`、`autoencoder`、`variational_autoencoder`、`tcn`、`ssm`、`mamba_s4`、`esn`、`siamese_triplet`、`unet_encoder_decoder_skip`、`capsule`、`kan`、`som`、`tree_random_forest_xgboost`、`svm`、`tiny_tcn`

## 4. 当前已同步修正的结论

### 4.1 已同步：新增 GNN 网络类型并接入编译期开关/注册链路

当前已确认：

- `src/CMakeLists.txt` 新增 `ACTION_C_ENABLE_NN_GNN`
- `src/nn/CMakeLists.txt` 在开关启用时纳入 `gnn` 类型源码
- `demo/demo_common.cmake` 已支持 demo 侧按类型启用 `gnn`
- `src/nn/types/gnn/` 已包含：
  - config-only 头文件
  - infer/train 两套 ops
  - infer/train 两个注册桥接文件
  - 目录级说明文档

这与文档要求的“通过 CMake 开关 + 注册配置接入，不修改 profiler 主流程做特例”的方向保持一致。

### 4.2 已同步：GNN 后端语义已从场景特化改为通用图结构语义

当前 `GnnConfig` 已不再使用“open/current/target”这类业务命名作为后端固有配置，而是改为：

- `node_mask_feature_index`
- `primary_anchor_feature_index`
- `secondary_anchor_feature_index`
- `aggregator_type`
- `readout_type`

同时：

- 推理侧支持 `GNN_READOUT_GRAPH_POOL`
- 推理/训练共同支持 `GNN_READOUT_ANCHOR_SLOTS`
- 权重保存/加载头部已同步校验聚合类型、读出类型与相关特征索引

这使 `gnn` 更符合“`src/` 提供通用网络能力，demo 仅展示用法”的工程定位。

### 4.3 已同步：`road_graph_nav` 的定位已调整为“使用示例”

当前 `demo/road_graph_nav/` 更准确的理解应为：

- 这是 `gnn(graph_encoder) -> mlp(decision_head)` 的组合网络示例
- `open/current/target/x/y` 是**场景特征编码**，不是 GNN 后端内建语义
- 示例通过 BFS 教师策略生成监督信号，用于说明图结构网络在局部导航场景中的一种可行用法

因此，这个目录的职责是“展示一个通用网络组合如何用于图结构导航”，而不是“为 demo 定制一个专用网络类型”。

### 4.4 仍然成立：严格告警 / 警告即错误门禁已落地

当前仍已确认：

- `src/CMakeLists.txt` 与 `demo/demo_common.cmake` 都定义了严格告警函数
- MSVC 使用 `/W4 /WX`
- 非 MSVC 使用 `-Wall -Wextra -Wpedantic -Werror`

并且这些规则继续覆盖核心库与 demo 目标。这一点与文档要求保持一致。

## 5. 当前仍然存在的主要问题

### 5.1 高优先级：网络类型覆盖仍明显不足

虽然当前已从 4 种提升到 5 种，但距离文档要求的 21 种常见网络类型仍有明显差距。

影响：

- 当前实现范围仍小于文档承诺范围
- 若按现有文档直接验收，覆盖度仍不足

### 5.2 高优先级：注释密度仍明显低于文档要求

现状：

- `src/` 约 23.9%
- `demo/` 约 10.0%
- 总体约 18.9%

影响：

- 与“源码与生成代码注释不少于 50%”的文档要求不一致
- 新增 GNN 与组合示例虽然已补充部分结构性注释，但整体仍未达标

### 5.3 中优先级：GNN 当前仍是轻量版本，不等同于完整通用 GNN 家族覆盖

当前 `gnn` 已经完成“通用图网络叶子能力”的第一步，但它仍然是轻量后端，主要提供：

- 固定节点数
- 固定槽位邻接
- 均值聚合
- 图池化 / 锚点槽位两类读出

这满足了当前工程主链路接入要求，但不等于已经覆盖更广义的 GNN 变体能力。

## 6. 正向观察

- `gnn` 已按文档要求接入 `src/nn/types/`、CMake 开关与注册桥接
- GNN 配置字段已从业务语义收敛到通用图语义
- 推理与训练读出逻辑已经同步到新的通用配置
- 权重头部的保存/加载兼容性校验已与新字段保持一致
- `road_graph_nav` 作为示例，已经比较清楚地体现了“通用叶子能力 + 场景特征编码 + 组合网络”的关系
- 之前修复的严格告警门禁、target demo 可视化与脚本修复结论仍然有效

## 7. 建议的后续修复顺序

1. 继续决定并收敛“21 种网络类型”是当前阶段刚性要求还是阶段目标。
2. 若当前文档口径不变，则继续扩展缺失网络类型；若要分阶段交付，应先明确文档阶段边界。
3. 系统性补齐注释密度，优先覆盖：
   - 所有权与生命周期
   - profiler 输入/输出合同
   - train/infer 依赖边界
   - 新增 GNN 的读出/保存/加载一致性
4. 在能力覆盖与注释密度稳定后，再继续扩展更丰富的 GNN 聚合/读出策略。

## 8. 一致性结论

按当前仓库重新审视后的更准确结论是：

- 与文档**更趋一致**的部分：
  - profiler 主链路下的编译期启用 / 注册接入思路
  - 严格告警与警告即错误门禁
  - 新增 `gnn` 网络类型的通用化定位
  - `road_graph_nav` 作为使用示例而非核心后端特例
- 与文档**仍不一致**的部分：
  - 网络类型覆盖
  - 注释密度

因此，本次更新后的结论应为：

> **当前仓库在“工程主链路正确性”和“新增 GNN 的通用化设计”方面已经前进了一步，但实现范围与注释密度仍未满足文档验收口径。**
