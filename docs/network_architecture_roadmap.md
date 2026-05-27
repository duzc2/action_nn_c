# 网络结构扩展路线图

## 当前已实现

| 类型 | 注册名 | 能力 |
|---|---|---|
| MLP | `mlp` | 全连接，可变层数，6种激活函数，SGD/Adam |
| CNN | `cnn` | 标准/深度可分离卷积，4种池化，BN，Dropout |
| CNN Dual Pool | `cnn_dual_pool` | CNN 变体，固定 avg+max 双池化 |
| RNN | `rnn` | Elman式单层RNN，tanh隐藏激活 |
| GNN | `gnn` | 消息传递图网络，mean聚合，图池化/锚点读出 |
| Transformer | `transformer` | 单头自注意力，字符级tokenizer，文本QA |

## 待实现（按优先级排列）

### P0 — 基础组件补齐（直接影响现有网络能力上限）

**1. Residual / Skip Connection（残差连接）**

- 当前状态：所有网络均为纯前馈，无法构建深层网络
- 影响范围：MLP、CNN 均受限于深度增加后梯度消失/退化问题
- 实现方式：作为 CNN/MLP 的可选配置项，支持 `RESIDUAL` 跳过一层或多层后将输入加到输出
- 参考：He et al., "Deep Residual Learning for Image Recognition", 2015

**2. LSTM / GRU**

- 当前状态：仅实现 Elman RNN，无法处理长序列依赖
- 影响范围：`cnn_rnn_react` 等时序 demo，序列预测类任务
- 实现方式：新增 `src/nn/types/lstm/` 或作为 RNN 的扩展配置，提供 forget/input/output gate 和 cell state
- 参考：Hochreiter & Schmidhuber, "Long Short-Term Memory", 1997; Cho et al., "GRU", 2014

**3. Multi-Head Attention（多头注意力）**

- 当前状态：Transformer 仅实现单头注意力，表达能力受限
- 影响范围：`hybrid_route`、`transformer` 等 demo 的效果上限
- 实现方式：在现有 Transformer 中增加 head 数量配置，将 Q/K/V 拆分为多组并行计算后拼接
- 参考：Vaswani et al., "Attention Is All You Need", 2017

### P1 — 通用组件提取（跨类型复用）

**4. Layer Normalization / Batch Normalization（归一化层）**

- 当前状态：BN 仅在 CNN 内部耦合实现，无 LayerNorm
- 影响范围：Transformer（LayerNorm 是其标准组件）、RNN/LSTM（序列归一化）、MLP
- 实现方式：抽象为 `src/nn/norm/` 共享模块，各网络类型按需引用
- 参考：Ba et al., "Layer Normalization", 2016; Ioffe & Szegedy, "Batch Normalization", 2015

**5. Dropout（通用正则化）**

- 当前状态：仅在 CNN 内部有 ad hoc 实现
- 影响范围：所有网络类型均可受益
- 实现方式：抽象为 `src/nn/regularization/` 共享模块，提供训练时随机失活 + 推理时缩放

### P2 — GNN 增强

**6. GNN 扩展**

- 当前状态：GNN README 明确标注为"刻意轻量"版本，仅 mean 聚合器、仅邻接关系（无边权重）
- 待补充：
  - Sum / Max / Attention 聚合器
  - 边特征支持（带权重的消息传递）
  - 多头注意力消息传递（GAT 风格）
- 参考：Veličković et al., "Graph Attention Networks", 2018

### P3 — 高层架构（需要较多新代码）

**7. Seq2Seq / Encoder-Decoder（序列到序列）**

- 当前状态：无任何 encoder-decoder 结构
- 影响范围：序列转换任务（如翻译、时序预测、轨迹规划）
- 实现方式：在 profiler 层支持连接两个网络（encoder + decoder），decoder 接收 encoder 的输出作为上下文
- 参考：Sutskever et al., "Sequence to Sequence Learning with Neural Networks", 2014

**8. Embedding Layer（可学习嵌入层）**

- 当前状态：Transformer 直接使用字符级 tokenizer，无可学习 embedding
- 影响范围：大词汇量 NLP 任务、类别特征编码
- 实现方式：新增通用 `nn/embedding/` 模块，或集成到 Transformer/Router 中
- 参考：Mikolov et al., "Efficient Estimation of Word Representations in Vector Space", 2013

### P4 — 特定场景（按需实施）

**9. U-Net**

- 当前状态：无图像分割/像素级预测网络
- 前提条件：需要 P0-1（残差连接）和 P0-2（跳跃连接）作为基础
- 影响范围：边缘视频预处理中的像素级运动分割、道路语义分割等
- 参考：Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015

---

## 依赖关系

```
P0-1 (残差连接)
├── 几乎所有深层网络的基石
└── P4-1 (U-Net) 的前提

P0-2 (LSTM/GRU)
├── 替代弱化现有 RNN
└── P3-1 (Seq2Seq) 的常见 encoder/decoder 单元

P0-3 (多头注意力)
├── 升级 Transformer 至实用水平
└── P2-1 (GNN Attention) 的技术基础

P1-1 (LayerNorm)
├── 配合 P0-2、P0-3 使用
└── 自身为独立共享模块

P1-2 (Dropout)
└── 独立共享模块，无其他依赖
```

---

*最后更新：2026-05-27*
