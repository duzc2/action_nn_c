# P4-01: 远期预留 — MobileNet / U-Net / Mamba-2

> 阶段: Phase 4 | 工作量: 各方向独立, 按需启动 | 状态: **预留 (不纳入当前开发计划)**

---

## 概述

Phase 4 包含三个方向，均需较重的 Phase 0/1 前置依赖。在当前应用场景中暂时不需要，待未来出现明确需求时再拆分为详细 TODO。

## 方向 1: MobileNet 风格轻量 CNN 配置

### 是什么

在现有 CNN 基础上，系统化提供 MobileNetV2 风格的轻量配置组合：
- **Inverted Residual Block**: 先升维(1×1 expand) → 深度卷积(3×3) → 降维(1×1 project) + skip
- **Linear Bottleneck**: 降维层不用激活（保留信息完整性）
- **Efficient Activation**: h-swish / ReLU6 替代标准 ReLU

### 为什么是远期

- 当前边缘视觉 demo (`edge_video_preprocess`) 已有深度可分离卷积 (`CNN_CONV_DEPTHWISE`)
- 深度可分离卷积是 MobileNet 的核心操作，当前已可用
- 系统化的 Inverted Residual + Linear Bottleneck 配置是锦上添花，不是必需品
- 优先完成 Phase 0-2 的核心能力升级

### 前置依赖

- P0-01 (RMSNorm) — MobileNet 变体用 RMSNorm/BN
- P0-02 (LAuReL-RW) — Inverted Residual 的 skip 连接
- P0-03 (Dropout/Stochastic Depth) — 深层 MobileNet 的正则化

### 2025-2026 最新进展

| 版本 | 核心创新 | 参考 |
|---|---|---|
| MobileNetV4 (2024) | Universal Inverted Bottleneck + Mobile MQA | 移动生态系统统一模型 |
| RepVGG 风格 | 训练时多分支，推理时融合为单路 | 零推理开销的结构重参数化 |
| CNN + Attention 融合 | 深度可分离 + 通道注意力 (SENet/ECA-Net) | 轻量注意力增强 |

### 当前可用的替代方案

- `CNN_CONV_DEPTHWISE` + `CNN_CONV_POINTWISE` 组合即可构建类 MobileNet 网络
- 开发者可以手动配置网络 JSON 实现，不必等系统化支持

---

## 方向 2: U-Net — 像素级预测

### 是什么

编码器-解码器架构用于像素级任务（语义分割、运动分割），特征是通过跳跃连接将 encoder 的特征图直接拼接到 decoder 对应层。

```
Encoder: Conv → Pool → Conv → Pool → ... → bottleneck
Decoder: UpConv → Concat(encoder特征) → Conv → UpConv → Concat → Conv
```

### 为什么是远期

- 像素级任务仅 `edge_video_preprocess` 可能需要（运动分割）
- 当前 demo 的运动检测基于帧差/光流，U-Net 分割是更高级但非必须的方案
- U-Net 的跳跃连接需要 P0-02 (残差/skip) 和更深层 CNN 支持
- 当前 Phase 0-2 的能力升级对当前 demo 的影响远超 U-Net

### 前置依赖

- P0-01 (RMSNorm)
- P0-02 (LAuReL-RW) — 跳跃连接需要 skip 机制
- P1-03 (SwiGLU) — encoder/decoder CNN 的激活升级
- Phase 3 完成后才可能具有基础设施

---

## 方向 3: Mamba-2 / SSD — 长序列状态空间

### 是什么

Mamba-2 (SSD: State Space Duality) 是一种使用半可分矩阵的状态空间模型，具有线性时间复杂度和与注意力理论的数学对偶关系。

```
SSD 核心:
  Y = (L ⊙ Q·K^T) · V

其中 L 是半可分 SSD 矩阵, ⊙ 是逐元素乘。
结果是: Q·K^T 和 V 之间的结构化矩阵变换
```

### 关键挑战

1. **segsum 稳定化**: 对长期依赖的数值稳定性是核心难点
2. **实现复杂度**: 远高于 minGRU（minGRU 的 parallel scan 仅需 ~50 行 C）
3. **当前序列长度不需要**:
   - weather: 3653 天（线性复杂度关注点，但 minGRU 已足够）
   - snake: 游戏帧数 ~100s
   - cs: 导航步数 ~100s
4. **minGRU 已覆盖序列建模**: P1-01 的 minGRU 提供了并行 scan 训练 + O(1) 推理能力

### 为什么保留

- 当 weather 扩展到极长序列（万 ~ 十万级），线性复杂度的优势显著
- Mamba-2 的数学对偶性质在理论上优雅，未来可能有更多应用
- 学术价值：验证 SSD 在 C11 纯代码中的可行性

### 前置依赖

- P0-01 (RMSNorm)
- P1-01 (minGRU) — parallel scan 实现可复用
- segsum 稳定化是独立研究工作

### 2025-2026 最新进展

| 方向 | 代表工作 | 说明 |
|---|---|---|
| Mamba-2 (SSD) | Dao & Gu, ICML 2024 | 半可分矩阵统一 SSM 和注意力 |
| S4 | Gu et al., ICLR 2022 | 原始结构化状态空间模型 |
| xLSTM | Beck et al., 2024 | LSTM 复兴, 并行化训练 |
| minLSTM/minGRU | Feng et al., ICLR 2025 | ICLR 2025 最佳论文候选，极简实现 |

---

## 启动条件

当以下任一条件满足时，将对应方向从本节拆分为独立 TODO 文件：

1. **MobileNet**: 有 demo 明确需要系统化的轻量 CNN 配置（而非手动组合 depthwise + pointwise）
2. **U-Net**: `edge_video_preprocess` 需要像素级运动分割，或新增语义分割 demo
3. **Mamba-2**: weather 或 cs 的序列长度超过 10,000，minGRU 遇到瓶颈

---

**参考论文:** MobileNetV1-V4 (2017-2024); U-Net, Ronneberger et al. 2015; Mamba-2/SSD, Dao & Gu, ICML 2024
