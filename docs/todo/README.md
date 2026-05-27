# TODO 索引

> 由 `docs/network_architecture_roadmap.md` 拆分而来。
> 每个文件包含该功能的：概述、项目必要性、详细子任务、文件清单、API 设计、测试方案、验收标准。

## 当前状态

| 阶段 | 状态 |
|------|------|
| Phase 0 — 核心模块 | ✅ 3/3 已实现，7,522 assertions 全部通过 |
| Phase 0 — 网络接入 | 🔲 MLP/CNN/RNN/GNN/Transformer 接入待完成 |
| Phase 1 | 🔲 待开始 |
| Phase 2 | 🔲 待开始 |
| Phase 3 | 🔲 待开始 |
| Phase 4 | 🔲 远期按需 |

## 文件清单

### Phase 0 — 基础设施（零依赖，三项可并行）

| 文件 | 模块 | 工作量 | 核心实现 | 网络接入 |
|---|---|---|---|---|
| [p0_01_rms_norm.md](p0_01_rms_norm.md) | RMSNorm 归一化层 | 小 | ✅ | 🔲 |
| [p0_02_residual_laurel.md](p0_02_residual_laurel.md) | LAuReL-RW 残差连接 | 中 | ✅ | 🔲 |
| [p0_03_dropout.md](p0_03_dropout.md) | Dropout + Stochastic Depth 正则化 | 小 | ✅ | 🔲 |

### Phase 1 — 网络能力升级

| 文件 | 模块 | 工作量 | 新增文件 |
|---|---|---|---|
| [p1_01_min_gru.md](p1_01_min_gru.md) | minGRU 序列模型 | 中 | `src/nn/types/rnn/` 扩展 |
| [p1_02_multihead_attention.md](p1_02_multihead_attention.md) | 多头注意力 + GQA | 中 | `src/nn/attention/` |
| [p1_03_swiglu.md](p1_03_swiglu.md) | SwiGLU 激活函数 | 小 | 无新目录 (MLP/Transformer 扩展) |
| [p1_04_muon_optimizer.md](p1_04_muon_optimizer.md) | Muon 优化器 | 中 | `src/train/optimizer_muon.h/.c` |

### Phase 2 — 领域特化增强

| 文件 | 模块 | 工作量 | 新增文件 |
|---|---|---|---|
| [p2_01_gnn_edge_gatv2.md](p2_01_gnn_edge_gatv2.md) | GNN 边特征 + EC-GATv2 | 中 | `src/nn/types/gnn/` 扩展 |
| [p2_02_data_augmentation.md](p2_02_data_augmentation.md) | 数据增强框架 | 小 | `src/train/augment.h/.c` |
| [p2_03_distillation.md](p2_03_distillation.md) | 知识蒸馏框架 | 中 | `src/train/distill.h/.c` |
| [p2_04_int8_quantization.md](p2_04_int8_quantization.md) | INT8 后训练量化 | 中 | `src/infer/quantize.h/.c` |
| [p2_05_hyperparameter_search.md](p2_05_hyperparameter_search.md) | 超参数搜索器 | 中 | `tools/hyper_search/` (Python) |

### Phase 3 — 新能力

| 文件 | 模块 | 工作量 | 新增文件 |
|---|---|---|---|
| [p3_01_embedding_rope.md](p3_01_embedding_rope.md) | Embedding Layer + RoPE | 中 | `src/nn/embedding/` |
| [p3_02_seq2seq.md](p3_02_seq2seq.md) | Seq2Seq Encoder-Decoder | 大 | profiler 扩展 |

### Phase 4 — 远期按需

| 文件 | 模块 | 说明 |
|---|---|---|
| [p4_01_reserved.md](p4_01_reserved.md) | MobileNet / U-Net / Mamba-2 | 待有明确需求时再拆分详细 TODO |

## 依赖关系

```
P0 (可同时开工)
├── p0_01 RMSNorm       ──→ P1_01, P1_02, P2_01, P3_01, P3_02
├── p0_02 LAuReL-RW     ──→ P1_01, P1_02, P3_02
└── p0_03 Dropout       ──→ P2_02

P1 (P0 完成后)
├── p1_01 minGRU        ──→ P3_02
├── p1_02 MHA+GQA       ──→ P2_01, P3_01, P3_02
├── p1_03 SwiGLU        (几乎独立)
└── p1_04 Muon          (独立)

P2
├── p2_01 GNN EC-GATv2  (需 P0_01 + P1_02)
├── p2_02 Data Aug      (需 P0_03 + P1_04 协同)
├── p2_03 Distillation  (独立)
├── p2_04 INT8 Quant    (独立, 可立即启动)
└── p2_05 Hyper Search  (独立, 可立即启动)

P3 (需 P0 + P1)
├── p3_01 Embed+RoPE    (需 P0_01 + P1_02)
└── p3_02 Seq2Seq       (需 P0_01 + P0_02 + P1_01 + P1_02)
```

## 使用方式

1. 选择要实现的模块，打开对应文件
2. 文件内的子任务按**执行顺序**排列，完成任务后勾选 `[x]`
3. 每个子任务标注了涉及的文件和依赖关系
4. 完成后参考"验收标准"自检
