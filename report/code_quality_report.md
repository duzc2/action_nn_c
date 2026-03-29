# 代码质量检查报告

日期：2026-03-29

## 1. 检查依据与范围

- 依据文档：
  - `docs/user_manual.md`
  - `docs/developer_manual.md`
  - `docs/network_topology_requirements.md`
  - `docs/network_design_manual.md`
  - `docs/profiler_api_contract_draft.md`
  - `docs/profiler_interface_draft.md`
  - `docs/profiler_development_plan.md`
- 本次同步重点：
  - `src/nn/types/mlp/*`
  - `src/nn/types/gnn/*`
  - `src/nn/types/cnn/*`
  - `src/nn/types/rnn/*`
  - `src/nn/types/transformer/*`
  - `src/profiler/prof_codegen.c`
  - `demo/*/generate_main.c`
- 检查方式：
  - 静态代码审查
  - 限制常量与固定容量搜索
  - 注释密度与网络类型覆盖统计
- 按用户要求，本次**未做编译、运行或测试执行**。

## 2. 本次同步后的结论

当前代码状态相较上一版报告，最重要的同步点已经变为：

1. 已实现的 5 种网络类型（`cnn`、`gnn`、`mlp`、`rnn`、`transformer`）中，**面向网络规模/复杂度的内建 MAX 限制已被清理**。
2. `mlp` 与 `gnn` 的类型配置已经改为**可变尺寸 blob**：
   - `MlpConfig.hidden_sizes[]`
   - `GnnConfig.neighbor_index[]`
3. `cnn`、`rnn`、`transformer` 中原先依赖固定栈数组的 scratch / cache，已改为**按用户配置动态分配**。
4. `transformer` 不再保留固定词表上限常量，词表容量改为由 `TransformerModelConfig.vocab_size` 指定。
5. profiler/codegen 侧已经按**原始配置字节 blob**传递类型配置，避免把可变尺寸配置错误地重建成固定结构。
6. 相关 demo 的 `generate_main.c` 已同步传入新的 transformer 配置字段。

因此，当前更准确的结论应为：

> **已实现网络后端的内建容量上限已显著收敛，网络规模主要由用户配置决定；但按仓库文档的验收口径，注释密度与网络类型覆盖仍未达标。**

## 3. 本次已确认同步到代码的事项

### 3.1 MLP：移除固定隐藏层容量与若干训练侧固定数组

- `MlpConfig` 已改为带柔性数组成员的可变尺寸配置。
- 推理上下文保存配置 blob 与配置大小，不再假设固定 `hidden_sizes[4]`。
- 训练侧已移除固定长度的：
  - batch 临时数组
  - loss 临时梯度数组
  - registry 兼容 dummy 样本数组
  - 固定 100 项 loss history
- 当前 MLP 的缓冲区大小由实际网络尺寸推导，不再由编译期常量限定。

### 3.2 GNN：移除拓扑相关 MAX 常量

- `GNN_MAX_*` 常量已移除。
- `GnnConfig` 的邻接表改为扁平柔性数组。
- 推理/训练阶段的邻接访问均通过配置尺寸与访问器完成。
- 与消息传递阶段相关的中间缓冲区已改为运行期按配置分配。

### 3.3 CNN / RNN：移除固定 scratch 容量

- `cnn` 中池化临时缓冲已改为动态分配。
- `rnn` 中前向/反向的隐藏态 scratch 与输出梯度缓冲已改为动态分配。

### 3.4 Transformer：移除固定词表与固定张量上限

- `TRANSFORMER_MAX_*` 与固定词表常量已去除。
- `TransformerModelConfig` 现包含：
  - `vocab_size`
  - `model_dim`
  - `max_seq_length`
  - `max_response_classes`
  - `max_text_length`
  - `seed`
- 推理/训练缓存均已改为堆分配。
- 图模式训练已对应更新到 `graph_projection_weight / graph_projection_bias`。
- 权重文件头中的容量字段已切换到 64 位存储，避免新的固定上限。

### 3.5 profiler / codegen：与可变尺寸配置保持一致

- `NNCodegenInferConfig.hidden_sizes` 已从固定数组改为指针视图。
- 生成代码对 infer/train 类型配置统一按**原始字节数组**下发。
- 对 `mlp` / `gnn` 这类可变尺寸配置，不再假设 `sizeof(typed_config)` 就是完整配置大小。
- `prof_codegen.c` 中 infer/train 生成阶段对类型配置初始化的格式化输出已拆分，避免单个 `append_format` 参数数量与占位符失配，修复 `road_graph_nav_generate` 在生成 `infer.c` / `train.c` 前崩溃的问题。

## 4. 量化结果

### 4.1 注释密度

按当前静态统计：

- `src/`：67 个 `.c/.h` 文件，15192 行，注释行约 3154，注释占比约 **20.8%**
- `demo/`：32 个 `.c/.h` 文件，8731 行，注释行约 842，注释占比约 **9.6%**
- `src/ + demo/` 总体：99 个 `.c/.h` 文件，23923 行，注释行约 3996，注释占比约 **16.7%**

对照文档中“源码与生成代码不少于 50% 注释量”的要求，当前仍**不达标**。

### 4.2 网络类型覆盖

- 文档列出的常见网络类型：**21** 种
- 当前 `src/nn/types/` 实际实现：`cnn`、`gnn`、`mlp`、`rnn`、`transformer`，共 **5** 种
- 当前仍缺失：`knn`、`rbfn`、`autoencoder`、`variational_autoencoder`、`tcn`、`ssm`、`mamba_s4`、`esn`、`siamese_triplet`、`unet_encoder_decoder_skip`、`capsule`、`kan`、`som`、`tree_random_forest_xgboost`、`svm`、`tiny_tcn`

## 5. 当前仍存在的主要问题

### 5.1 注释密度仍明显低于文档要求

虽然本次对若干核心文件补充了结构性注释，但整体注释占比仍显著低于文档要求。

### 5.2 网络类型覆盖仍不足

当前实现从工程主链路角度已能覆盖多种示例，但与文档列出的网络类型范围仍有较大差距。

### 5.3 Transformer 语义层仍然是轻量实现

当前 transformer 后端已经去除了固定容量限制，但其文本训练/分类与图模式投影仍然是轻量实现；这不再是“容量被写死”的问题，而是“算法能力范围仍有限”的问题。

## 6. 正向观察

- 已实现网络后端的内建 MAX 限制已基本清理
- 可变尺寸配置已经贯通到 profiler/codegen 桥接层
- demo 侧配置已与新的 transformer 结构保持同步
- `src/` 侧后端注释风格比此前更接近“说明所有权、尺寸来源、训练/推理边界”的要求

## 7. 建议的后续顺序

1. 由用户自行编译验证本次去限制改造后的接口一致性。
2. 若继续强化通用性，优先完善 transformer 的更通用 tokenizer / 序列语义设计，而不是再引入新的固定便捷常量。
3. 系统性补齐注释密度，优先覆盖：
   - 配置 blob 生命周期
   - train/infer 共享参数边界
   - profiler 传参与生成代码重建逻辑
4. 再决定缺失网络类型是继续补齐，还是把文档改为阶段性目标。

## 8. 一致性结论

按当前代码重新审视：

- 与文档**更趋一致**的部分：
  - 用户配置驱动网络尺寸与复杂度
  - profiler 主链路下的类型配置传递方式
  - `src/` 提供通用后端、demo 仅展示使用方式
- 与文档**仍不一致**的部分：
  - 网络类型覆盖
  - 注释密度

因此，本次报告更新后的结论为：

> **当前仓库在“已实现网络后端不再内建固定容量上限”这一点上已经明显改进，但文档验收层面的覆盖度与注释密度问题仍然存在。**
