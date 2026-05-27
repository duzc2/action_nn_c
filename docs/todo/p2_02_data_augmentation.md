# P2-02: 数据增强框架

> 阶段: Phase 2 | 工作量: 小 | 前置依赖: P0-03 (Dropout RNG), P1-04 (Muon 协同)

---

## 1. 是什么

在训练端提供标准数据增强操作，提高模型泛化能力。训练在桌面进行，增强后的数据送入训练循环。

## 2. 为什么需要

- edge_video_preprocess、mnist 等视觉 demo 的训练数据有限
- Mixup + CutMix 与 Muon 优化器有共生效应（梯度谱更丰富）
- 每个 demo 目前没有统一的数据增强框架

## 3. 详细子任务

### 3.1 图像级增强 (`src/train/augment.h` + `augment.c`)

- [ ] `random_horizontal_flip(img_w, img_h, channels, keep_prob, rng_state)`
- [ ] `random_crop_resize(src_w, src_h, dst_w, dst_h, channels, rng_state)`
- [ ] `random_erasing(w, h, channels, max_area_ratio, count, rng_state)`
- [ ] `cutmix(a, b, label_a, label_b, w, h, channels, beta_alpha, rng_state)` → 合成图 + 混合标签
- [ ] `mixup(a, b, label_a, label_b, channels, alpha, rng_state)` → 混合图 + 混合标签

### 3.2 特征级增强

- [ ] `gaussian_noise(x, n, std)` — 训练时注入噪声
- [ ] `feature_dropout(x, n, keep_prob, rng_state)` — 输入维度的 dropout

### 3.3 标签增强

- [ ] `label_smoothing(y_onehot, n_classes, epsilon)` — 软化标签到 `[eps/n, ..., 1-eps+eps/n]`

### 3.4 测试

- [ ] cutmix/mixup 输出形状和标签混合比例正确
- [ ] mnist + Mixup: 验证集准确率 > 无增强的基准

## 4. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `src/train/augment.h` |
| **新建** | `src/train/augment.c` |
| **修改** | `src/train/CMakeLists.txt` |

## 5. 验收标准

1. Mixup(α=0.2): 混合后的图 = α·A + (1-α)·B，标签同理
2. CutMix: 被贴的区域占指定比例，标签按面积混合
3. Random Erasing: 矩形区域被置零，长宽比在指定范围内
4. label_smoothing: 正确类概率 = 1-ε+ε/C, 其余 = ε/C

---

**参考论文:** Mixup, Zhang et al. 2018; CutMix, Yun et al. 2019; RandAugment, Cubuk et al. 2020
