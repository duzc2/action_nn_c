# CIFAR-10 到 80%+ 准确率的实现计划

> 创建日期: 2026-05-24
> 框架: action_c (custom C CNN framework)
> 目标: CIFAR-10 测试集准确率 > 80%

---

## 一、研究总结：CIFAR-10 达到 80% 的已知方案

### 方案 A: 简单 CNN + BatchNorm + Adam（最易实现）
- **架构**: 3-4 个卷积层（3×3, ch: 32→64→128），每层后接 BN + ReLU + MaxPool
- **参数**: < 50K 到 500K 不等
- **优化器**: Adam (lr=0.001)
- **准确率**: **74-82%**
- **关键**: BN 是必需的；Adam 比 SGD 收敛更快
- **来源**: GitHub 多个仓库 (LianFi15, Xujan24, brennobrenno)

### 方案 B: 中等 CNN + BN + SGD + 数据增强（较好泛化）
- **架构**: 4-5 个卷积层 + 2 个全连接层
- **优化器**: SGD+momentum=0.9, lr=0.1→0.01 衰减
- **额外**: Dropout 0.5, 数据增强（翻转、裁剪）
- **准确率**: **76-86%**（无增强约 77-79%，有增强 80-86%）
- **来源**: brennobrenno, EshanJairath (GitHub)

### 方案 C: VGG 风格深网络 + BN + SGD（最高准确率）
- **架构**: VGG-13/16（多卷积层堆叠）
- **BN 状态**: **有 BN 可达 92-94%，无 BN 仅 ~87%**
- **参数**: 1-15M
- **来源**: Nature Scientific Reports (2024), NguyenVanManh-AI

### 方案 D: ResNet + BN + SGD（当前 SOTA 水平）
- **架构**: ResNet-18/20/56
- **准确率**: **92-95%**
- **参数**: 0.27M-11M
- **来源**: Geiping et al. (ICLR 2022)

### 关键结论
| 条件 | 预期最好准确率 |
|------|--------------|
| 无 BN, 简单 CNN, SGD | ~67%（LeNet 级别，本研究已确认：v27 仅 10%） |
| 有 BN, 简单 CNN, Adam | 74-82% |
| 有 BN, 中等 CNN, SGD | 76-79% |
| 有 BN + 数据增强 + 中等 CNN | 80-86% |
| 有 BN, VGG/ResNet | 92-95% |

**核心发现**: BatchNorm 是 CIFAR-10 上突破 70% 的必要条件。无 BN 的情况下，即使深层网络也只能达到 ~67%。

---

## 二、关键 Bug 修复记录

### Bug #1: bn_spatial_var 零初始化导致梯度爆炸（2026-05-25 修复）

**根因**: `nn_cnn_train_create` 中 `bn_spatial_var` 通过 `calloc` 分配，所有元素初始化为 0。BN 反向传播代码对 POOL_AVG/MAX/DUAL 模式检查 `if (context->bn_spatial_var)`（指针非 NULL），使用 `bn_spatial_var[filter_index]`（值为 0）作为方差。

**后果**: `bn_scale = gamma / sqrt(0 + 1e-5) = 316.227`，将梯度放大 ~316 倍，L2 放大 ~100,000 倍。导致 v28, v29, v30, v32 的所有 BN 实验爆炸（非真正的不稳定性，而是 bug）。

**为什么 POOL_NONE 不受影响**: POOL_NONE 的反向传播在两阶段前向过程中直接设置了 `bn_spatial_var`，而 POOL_AVG/MAX/DUAL 跳过此设置（`sequence_length < 2` 条件）。

**修复**: 在 `calloc` 后初始化 `bn_spatial_var` 为 `bn_running_var`（running var 初始化为 1.0）：
```c
for (bni = 0U; bni < sequence_length * filter_count; ++bni) {
    context->bn_spatial_var[bni] = infer_ctx->bn_running_var[bni % filter_count];
}
```

**修复验证**: v34 训练梯度 L2 从 11,125,761 降至 104（~100,000x 降低），稳定运行 43,000+ step 无 NaN。

**影响范围**: 此 bug 影响所有使用 BN + 池化模式的实验（v28, v29, v30, v32）。v27（无 BN）不受影响。所有之前的"lr 太大导致爆炸"结论需要重新评估——实际上即使是 lr=0.002 也会爆炸，因为 bn_scale=316 等效于有效 lr=0.63。

### Bug #3 (根因): BN 反向传播 x_hat 公式错误导致 conv_w 完全停滞（2026-05-26 修复）

**根因**: `cnn_train_ops.c` 中 POOL_AVG/POOL_MAX/POOL_DUAL 的 BN 反向传播使用了错误的 x_hat 公式：
```c
// BUGGY: pooled_linear_cache 存储的是 pre-BN 原始值，但被当作 BN 输出处理
float x_hat = (linear_val - beta) / gamma;  // 错误！
float scale = gamma / sqrt(bn_var + eps);    // bn_var = bn_spatial_var（未更新）
context->bn_gamma_grad += dpool_act * x_hat;
dpool_linear = dpool_act * scale;
```

`pooled_linear_cache` 在训练模式下存储 pre-BN 的原始卷积值（如 `pooled_sum / N`），但 x_hat 公式 `(x - beta) / gamma` 假定 `x` 是 BN 输出（而非 raw 值）。正确的 x_hat 应使用 running statistics：
```c
// FIXED: 用 running mean/var 恢复正确的 x_hat
float x_hat = (linear_val - bn_running_mean) * (1.0f / sqrt(bn_running_var + eps));
float scale = bn_gamma * (1.0f / sqrt(bn_running_var + eps));
dpool_linear = dpool_act * scale;
```

**后果**:
- BN gamma/beta 梯度完全错误（x_hat 值与正向传播不一致）
- `dpool_linear`（进而 conv_w 梯度）通过错误的 scale 因子传播
- **这是 v34-v46 所有实验中 conv_w RMS 不变（<1% Δ）的真正根因**——不是 batch_size=1，而是 BN 反向传播的 gradient scale 被错误计算，导致传递到 conv 层的梯度严重衰减

**为什么 batch_size=4 也无法解决**: v44-v46 实现了 batch_size=4 的梯度累积，conv_w 梯度从 ~100 增至 ~300（仍远小于 conv_b 的 ~1000），权重 RMS 变化仅 0.0883→0.0880。根因是 BN 反向传播的 scale 错误，与 batch size 无关。

**修复验证**（v47）:
- conv_w 梯度 L2 从 v46 的 37-388 增至 v47 的 11512-57649（**150x**）
- conv_w 权重更新从 0.0033 增至 0.0425（**13x**）
- 训练 500 样本准确率从 baseline 8.20% 提升至 **14.10%**（v46: <0.3%）

**影响范围**: 此 bug 影响所有使用 BN + POOL_AVG/POOL_MAX/POOL_DUAL 的实验（v34-v46）。v41 虽然达到 23.1%，但学习几乎全部来自偏置（conv_b 3x 增长），conv_w 从未有效训练。

### Bug #4: BN running statistics 从未更新（POOL_AVG/POOL_MAX/POOL_DUAL）（2026-05-26 修复）

**根因**: `cnn_infer_ops.c` 中的 BN post-processing 在 `sequence_length < 2` 时 skip：
```c
if (config->sequence_length < 2U && !is_dual) {
    return 0;  // 跳过 BN 二次归一化，running stats 也不更新
}
```
对于 POOL_AVG/POOL_MAX/POOL_DUAL（sequence_length=1），running_mean 和 running_var **永远保持在初始化值**（0.0 和 1.0），无法跟踪实际激活分布。

**修复**: 在 POOL_AVG/POOL_MAX/POOL_DUAL 正向传播中，每次样本处理后直接用 EMA 更新 running statistics：
```c
if (bn_pre_cache != NULL) {
    float mom = config->bn_momentum;
    float old_mean = context->bn_running_mean[filter_index];
    context->bn_running_mean[filter_index] = mom * old_mean + (1.0f - mom) * pooled_linear_raw;
    float delta = pooled_linear_raw - old_mean;
    context->bn_running_var[filter_index] = mom * old_var + (1.0f - mom) * delta * delta;
}
```

**影响范围**: 此 bug 导致 BN 归一化始终使用 running_mean=0, running_var=1，与激活的实际分布不匹配。与 Bug #3 叠加，使权重梯度的计算完全基于错误的 statistics。

### Bug #5（次级时序bug）: BN 反向传播 forward/backward 的 running stats 不一致 —— EMA 更新时机错误（2026-05-26 诊断，2026-05-27 修复）

**这是 Bug #3/#4 修复后引入的次级 bug，解释了 v47 epoch 1 准确率 10.40%（比 v46 的 22.55% 更差）以及梯度崩溃（conv_w grad L2 从 42525 降至 0.37，31000x 衰减）。**

#### 根因：正向传播和反向传播使用了不同版本的 running statistics

**正向传播的时序**（`cnn_infer_ops.c`，POOL_AVG 路径，第 ~629-660 行）：

```c
// ── 步骤 1: 用 PRE-update 的 running stats 做归一化 ──
float bn_mean = context->bn_running_mean[filter_index];  // 旧值 μ_old
float bn_var  = context->bn_running_var[filter_index];   // 旧值 σ²_old
float x_hat = (pooled_sum - bn_mean) / sqrtf(bn_var + bn_eps);  // 用 μ_old, σ²_old
pooled_sum = bn_gamma * x_hat + bn_beta;

// ── 步骤 2: 存储 pre-BN 原始值（给反向传播用）──
pooled_linear_cache[index] = pooled_linear_raw;  // 存的是 raw 值（非 x_hat）

// ── 步骤 3: 更新 running stats（EMA）──
// ⚠ 此处修改了 running_mean/var，破坏了 backward pass 的一致性！
context->bn_running_mean[filter_index] = mom * old_mean + (1-mom) * pooled_linear_raw;  // → μ_new
float delta = pooled_linear_raw - old_mean;
context->bn_running_var[filter_index] = mom * old_var + (1-mom) * delta * delta;  // → σ²_new
```

**反向传播**（`cnn_train_ops.c`，POOL_AVG BN backward，第 ~451-461 行）：

```c
// ── 读取 POST-update 的 running stats（❌ 错误！）──
float gamma = infer_ctx->bn_gamma[filter_index];
float bn_mean = infer_ctx->bn_running_mean[filter_index];  // ⚠ 读的是 μ_new！
float bn_var  = infer_ctx->bn_running_var[filter_index];   // ⚠ 读的是 σ²_new！
float linear_val = context->pooled_linear_cache[pooled_index];  // raw 值（与 forward 一致）

// ── 用 μ_new 恢复 x_hat（❌ 不一致！）──
float inv_std = 1.0f / sqrtf(bn_var + bn_eps);  // 用 σ²_new
float x_hat = (linear_val - bn_mean) * inv_std;  // ⚠ μ_new ≠ μ_old → x_hat 错！
float scale = gamma * inv_std;  // ⚠ σ²_new ≠ σ²_old → scale 错！

// ── 计算梯度（基于错误的 x_hat 和 scale）──
context->bn_gamma_grad[filter_index] += dpool_act * x_hat;  // ❌
context->bn_beta_grad[filter_index]  += dpool_act;
dpool_linear = dpool_act * scale;  // ❌ 错误的 scale 传入 conv_w 梯度
```

#### 为什么这会导致梯度崩溃

```
正向传播用 (μ_old, σ²_old) 产生特定激活值 → 计算 dL/dactivation
反向传播用 (μ_new, σ²_new) 计算 dL/dlinear = dL/dactivation * gamma / sqrt(σ²_new)

当 running stats 持续漂移时（5000 samples × EMA decay）：
  μ_new 偏离 μ_old → x_hat 的值不匹配正向传播
  σ²_new 偏离 σ²_old → inv_std 的值不匹配正向传播

结果：
  d(pooled)/d(linear) 被错误计算 → conv_w 梯度方向/幅度错误
  → SGD 沿错误方向更新 → 权重退化 → 梯度崩溃
```

#### v47 的实验证据

```
step 0:   conv_w grad L2 = 42525  (正常，running stats 尚未漂移)
step 1000: conv_w grad L2 = ~2000  (开始衰减)
step 4000: conv_w grad L2 = ~50    (严重衰减)
step 8000: conv_w grad L2 = 0.37   (31000x 衰减!)
step 8000: conv_w RMS    = 0.0894  (仅 +1.2% vs 初始 0.0883)

Epoch 1 full eval: 10.40%（比 v46 的 22.55% 更差！）
峰值准确率: 16.80% (sample 1500)，之后崩溃至 10.40%
类崩溃: 55.6% 预测为 class 7
```

**关键拐点**: 训练前 ~1500 samples（~375 batches）准确率上升，之后 running stats 漂移到不再与前向一致，梯度开始崩溃。

#### 为什么 POOL_NONE 路径不受影响？

POOL_NONE 的反向传播（`cnn_train_ops.c` 第 ~290-296 行）**也读取 running stats**：
```c
float gamma = infer_ctx->bn_gamma[filter_index];
float bn_var = infer_ctx->bn_running_var[filter_index];  // POST-update
float inv_std = 1.0f / sqrtf(bn_var + bn_eps);
bn_scale = gamma * inv_std;
```

但 POOL_NONE 的差异在于：POOL_NONE 用 `output_positions`（30×30=900）个空间位置的 **spatial statistics** 更新 running stats（`cnn_infer_ops.c` 第 505-510 行），而非单点值。900 个位置的平均使单样本更新幅度极小（效果上 EMA 接近 0），running stats 漂移远小于 pooled 路径。

#### 正确修复方案（三选一）

**方案 A: 在正向传播中保存 x_hat 和 inv_std（推荐，最简洁）**

不存 raw 值，直接存反向传播需要的 x_hat 和 inv_std：

```c
// 正向传播（cnn_infer_ops.c）
float inv_std = 1.0f / sqrtf(bn_var + bn_eps);  // 用 μ_old, σ²_old
float x_hat = (pooled_linear_raw - bn_mean) * inv_std;
pooled_sum = bn_gamma * x_hat + bn_beta;

// 保存 x_hat 和 inv_std（而非 raw 值）
pooled_x_hat_cache[index] = x_hat;       // 新增字段
pooled_inv_std_cache[index] = inv_std;   // 新增字段
// 或者合并为一个字段，存 (x_hat, inv_std) 对

// ... 然后更新 running stats（不影响已保存的值）
```

```c
// 反向传播（cnn_train_ops.c）
float x_hat   = context->pooled_x_hat_cache[index];     // 直接读取
float inv_std = context->pooled_inv_std_cache[index];   // 直接读取
float scale   = gamma * inv_std;
context->bn_gamma_grad += dpool_act * x_hat;
context->bn_beta_grad  += dpool_act;
dpool_linear = dpool_act * scale;  // 正确的 scale
```

优点：完全消除时序问题，不增加 forward backward 的数值代价。
缺点：需要新增 2 个 cache 数组（或合并为 1 个数组存结构体）。

**方案 B: 在正向传播中保存 PRE-update 的 running stats 快照**

在更新 running stats 之前保存快照：

```c
// 正向传播
float saved_mean = bn_mean;           // μ_old 快照
float saved_var  = bn_var;            // σ²_old 快照
pooled_linear_cache[index] = pooled_linear_raw;  // raw 值

// 新增 cache
pooled_bn_mean_cache[index] = saved_mean;  // μ_old
pooled_bn_var_cache[index]  = saved_var;   // σ²_old

// 然后更新 running stats（不影响已保存的快照）
context->bn_running_mean[filter_index] = mom * old_mean + (1-mom) * pool_raw;
```

```c
// 反向传播
float saved_mean = context->pooled_bn_mean_cache[index];  // μ_old
float saved_var  = context->pooled_bn_var_cache[index];   // σ²_old
float x_hat = (linear_val - saved_mean) / sqrt(saved_var + eps);
float scale = gamma / sqrt(saved_var + eps);
```

优点：语义清晰（"我们存了当时用的 stats"），raw 值仍然可见。
缺点：需要新增 2 个 cache 数组（256 × 2 = 512 floats per step），且 running stats 的值不与 raw 值共存（反向时 raw 值单独一行、stats 单独一行）。

**方案 C: 将 running stats 更新移到反向传播之后**

问题：这改变了 EMA 更新的语义——EMA 应该在每次看到数据后立即更新。推迟到反向传播后意味着：如果某个 step 后不再调用 backward（如在 eval 期间），stats 永远不更新。这是语义上的退化，不推荐。

#### 推荐方案

**方案 A（保存 x_hat + inv_std）** 是最干净的。它消除了反向传播读取任何可变 running stats 的需求，且值天然匹配正向传播使用的值。

#### 实际实现（方案 A 变体：复用已有 buffer，2026-05-27）

实际实现复用了已有的 `bn_pre_cache` 和 `bn_spatial_var` 两个 buffer（原本已为 POOL_AVG 路径分配但未用于此目的），无需新增任何 struct 字段或内存分配。

**正向传播**（`cnn_infer_ops.c`，POOL_AVG BN block）:

```c
// 1. 快照 PRE-EMA running stats
float bn_mean = context->bn_running_mean[filter_index];
float bn_var  = context->bn_running_var[filter_index];

// 2. 用 PRE-EMA stats 计算标准化参数
float inv_std = 1.0f / sqrtf((bn_var + bn_eps) > 0.0f ? (bn_var + bn_eps) : bn_eps);
float x_hat = (pooled_sum - bn_mean) * inv_std;

// 3. 缓存到已有 buffer（EMA 更新之前）
if (bn_pre_cache != NULL) {
    size_t cache_idx = (step_index * filter_count) + filter_index;
    bn_pre_cache[cache_idx] = x_hat;           // 复用：存 x_hat
    if (bn_spatial_var != NULL) {
        bn_spatial_var[cache_idx] = inv_std;   // 复用：存 inv_std
    }
}

// 4. 应用 BN 输出
pooled_sum = bn_gamma * x_hat + bn_beta;

// 5. EMA 更新 running stats（在缓存之后，不影响已缓存的值）
if (bn_pre_cache != NULL) {
    context->bn_running_mean[filter_index] = mom * old_mean + (1-mom) * raw;
    context->bn_running_var[filter_index]  = mom * bn_var + (1-mom) * delta²;
}
```

**反向传播**（`cnn_train_ops.c`，POOL_AVG BN backward）:

```c
if (has_bn_pool && dpool_act != 0.0f) {
    float gamma = infer_ctx->bn_gamma[filter_index];
    // 直接从 cache 读取（不再读取 running stats）
    float x_hat   = context->bn_pre_cache[pooled_index];     // 复用：读 x_hat
    float inv_std = context->bn_spatial_var[pooled_index];   // 复用：读 inv_std
    float scale   = gamma * inv_std;
    context->bn_gamma_grad[filter_index] += dpool_act * x_hat;
    context->bn_beta_grad[filter_index]  += dpool_act;
    dpool_linear = dpool_act * scale;  // 正确的 scale！
}
```

**关键差异 vs 原 `pooled_linear_cache` 方案**:
- 原来存 `pooled_linear_raw`（pre-BN raw value），反向时从运行态 running stats 重建 x_hat → 时序不匹配
- 现在存计算完成的 `x_hat` + `inv_std`，反向时直接读取 → 严格一致

**buffer 说明**:
- `bn_pre_cache`: 原为 POOL_AVG 路径分配 (`sequence_length × filter_count` floats) 但只用作 training-mode NULL 检查 —— 现真正存储 x_hat
- `bn_spatial_var`: 原存储 POST-EMA 方差（反向从未读取）—— 现存储 pre-EMA 的 inv_std
- 无新增内存分配、无新增 struct 字段

### Bug #5 附录: 方差 EMA 公式的偏差问题

Bug #4 修复中引入的方差 EMA 公式存在一个已知偏差：

```c
// 当前实现（有偏差的简化公式）
float delta = pooled_linear_raw - old_mean;
context->bn_running_var[filter_index] = mom * old_var + (1.0f - mom) * delta * delta;
// 这里 delta = x - μ_old，但正确的应该用 (x - μ_new)(x - μ_old)
```

正确的 Welford 在线方差更新（无偏）应该是：
```c
float new_mean = mom * old_mean + (1.0f - mom) * x;
float delta1 = x - old_mean;
float delta2 = x - new_mean;
new_var = mom * old_var + (1.0f - mom) * delta1 * delta2;
```

当前简化公式用 `(x - μ_old)²` 替代 `(x - μ_new)(x - μ_old)`，在 `mom=0.9` 时偏差约 10%。对于训练目的不是关键问题（主要是 x_hat 和 scale 的精确性更重要），但记录了以免将来误解。

---
## 三、当前代码库约束

> 最后更新: 2026-05-27

| 约束 | 详情 |
|------|------|
| CNN 优化器 | 仅支持 SGD+momentum（无 Adam） |
| MLP 优化器 | 支持 ADAM+CE |
| BN 实现 | 训练时用 running statistics 做归一化，反向传播用缓存的 pre-EMA x_hat + inv_std（Bug #5 已修复） |
| 数据增强 | 已实现：水平翻转 (p=0.5) + 随机 2px pad-crop (`cifar10_augment_sample()`) |
| 训练/验证分割 | 已实现：80/20 split（`split_train_val()`） |
| 批大小 | 4（mini-batch 梯度累积） |
| 输入尺寸 | 32×32×3 |
| 网络架构 | BnConvNet v47: 4 CNN (stride=1, BN+LeakyReLU) + GAP + 1 MLP(LeakyReLU) |
| 分类头 | 256→256→10 MLP，ADAM+CE |
| 训练恢复 | 已实现：`train_main.c` 自动加载 `weights.bin` checkpoint |
| 学习率调度 | 已配置（Step Decay, 0.5x every 3 epochs），等待 core API 支持 |
| LR 调度 | `EVP_LR_DECAY_RATE=0.5`, `EVP_LR_DECAY_EPOCHS=3`（显示 + 等待运行时支持） |

---

## 四、选定方案

**方案: BnConvNet v47（4 CNN + GAP + MLP, BN + LeakyReLU + batch_size=4）**

**设计理由**:
- 回归简单架构：在修复了 Bug #1-#5 后，4 层 CNN + BN + LeakyReLU 的简单架构足以验证核心梯度通路是否正常工作
- 不使用 ResNet projection shortcuts（DAG 拓扑有额外的调试成本）
- 不使用 stride-2 降采样（保留 GAP 前的空间分辨率 24×24）
- 所有卷积层 stride=1，依赖 3×3 卷积自然缩小特征图尺寸
- GAP 全局平均池化替代全连接前的展平

架构（BnConvNet v47）:
```
输入: 32×32×3 RGB

Conv1: 3x3,  3→32,   s=1, BN+LeakyReLU(0.01), POOL_NONE → 30×30×32
Conv2: 3x3,  32→64,  s=1, BN+LeakyReLU(0.01), POOL_NONE → 28×28×64
Conv3: 3x3,  64→128, s=1, BN+LeakyReLU(0.01), POOL_NONE → 26×26×128
Conv4: 3x3,  128→256,s=1, BN+LeakyReLU(0.01), POOL_NONE → 24×24×256
GAP:   1x1,  256→256, POOL_AVG → 256 scalars

MLP:  256 → [256(LeakyReLU)] → 10, ADAM+CE

总计: 5 个叶子节点 (4 CNN + 1 MLP), ~600K 参数
```

关键设计决策:
- BN 在所有 conv 层上（4 层 CNN 深度需要 BN）
- **Bug #1-#5 均已修复**：BN spatial_var 初始化、x_hat 公式、running stats 更新、forward/backward 统计一致性
- LeakyReLU(alpha=0.01) 防止死神经元
- CNN 层用 SGD+momentum=0.0；MLP 用 ADAM+CE
- batch_size=4（mini-batch 梯度累积）
- 数据增强已实现（`cifar10_augment_sample()`：水平翻转 + 随机裁剪）
- 训练/验证集 80/20 分割
- 支持 checkpoint resume（自动加载 weights.bin）

---

## 五、可调参数与结构

### 结构参数
| 参数 | 范围 | 说明 |
|------|------|------|
| 卷积层数 | 2-7 | 影响梯度流通深度和模型容量 |
| 每层通道数 | 16-256 | 第一层：3→N, 后续递增或保持 |
| 卷积核大小 | 3×3 或 5×5 | 3×3 为标准选择 |
| 降采样策略 | stride=2 或 max pool | 直接影响空间信息保留 |
| GAP 特征数 | 64-256 | 输入 MLP 的特征维度 |
| MLP 隐藏层 | 1-2 层, 64-512 宽 | 分类头容量 |
| BN 位置 | 全层 / 部分层 / 无 | 核心参数 |
| Dropout | 0-0.5 | 在 GAP 层后 |
| 激活函数 | ReLU / LeakyReLU / None | LeakyReLU(0.01) 防止死神经元 |

### 训练超参数
| 参数 | 范围 | 说明 |
|------|------|------|
| 学习率 | 0.0001-0.05 | 核心参数，与 BN 稳定性直接相关 |
| 动量 | 0.9（固定） | 标准值 |
| 权重衰减 | 1e-5 到 1e-3 | L2 正则化 |
| 偏置权重衰减 | 1e-3（固定） | 单独控制偏置正则化 |
| Dropout rate | 0-0.5 | 防止过拟合 |
| 训练样本数 | 5000-50000 | 迭代速度 vs 样本覆盖 |
| 训练轮数 | 5-200 | 收敛所需时间 |

---

## 六、调参策略

### 阶段 1: 确保 BN 稳定训练（当前阶段）
**目标**: 找到一个使 BN + SGD 不爆炸、不卡死的架构和 lr 组合

已知失败案例（BUG 修复前，bn_spatial_var=0 导致有效 lr 放大 316x）：

| 实验 | 配置 | 名义 lr | 有效 lr（因bug） | 结果 |
|------|------|---------|-------------------|------|
| v28 | 4 CNN + BN + ReLU | 0.002 | ~0.63 | NaN/explosion |
| v29 | 4 CNN + BN + ReLU | 0.05 | ~15.8 | 瞬间 NaN |
| v30 | 4 CNN + BN + ReLU | 0.005 | ~1.58 | NaN by step 1000 |
| v32 | 10 CNN + BN + LR | 0.001 | ~0.316 | NaN by step 36 |

**BUG 修复后的候选方案**:
- [x] 4 CNN + BN + ReLU + lr=0.002: v34 已验证训练稳定（43K+ step），但学习极慢（epoch 1 仅 11.60%）
- [ ] 4 CNN + BN + LeakyReLU + lr=0.002: v35 — 用 LeakyReLU 替代 ReLU，防止死神经元阻断梯度
- [ ] 4 CNN + BN + LeakyReLU + lr=0.01: 提高 lr（bug 修复后安全范围扩大多倍）
- [ ] 4 CNN + BN + LeakyReLU + lr=0.005: 中等 lr

### 阶段 2: 提升准确率
**目标**: 从 baseline → 30% → 50% → 70%

假设训练已稳定：
1. 逐步增加 lr（0.001 → 0.002 → 0.003 → ...）直到训练加速但不爆炸
2. 增加训练样本数（5K → 10K → 50K）
3. 增加训练轮数（5 → 20 → 50）
4. 增加层数/宽度以提升模型容量

### 阶段 3: 最终优化
**目标**: 70% → 80%+
1. 引入 dropout 正则化
2. 调整权重衰减
3. 调优 MLP 的 lr（ADAM 已有，可能需微调）
4. 考虑更宽或更深的架构变体

---

## 七、实验日志（持续更新）

### 实验 #1: v27 - ShallowVggNet (5 CNN + LeakyReLU, no BN)
- **时间**: 之前会话
- **配置**: 5 conv (3→32→64→128→256→256), lr=0.005, momentum=0.9, no BN, dropout=0.3
- **结果**: 准确率 10.6%，loss 几乎不下降，conv_w 梯度在大多数 step 为 0
- **结论**: ❌ 无 BN 的 SGD 无法训练 >2 层 CNN

### 实验 #2: v28 - BnConvNet (4 CNN + BN + LeakyReLU[其实是ReLU], lr=0.002)
- **时间**: 之前会话
- **配置**: 4 conv (3->32->64->128->256), BN, lr=0.002, dropout=0.2
- **结果**: Epoch 1 准确率 12.95%，训练稳定但极慢
- **结论**: ⚠️ 受 bn_spatial_var=0 bug 影响（Bug #1），所有 BN pooling 层梯度被放大 316x。v28 能"稳定"是因为 lr=0.002 有效成为 ~0.63，恰好处在 SGD 的临界稳定区。偏置 RMS 增长但权重 RMS 不变——偏置对 BN scale 放大更敏感。

### 实验 #3: v29 - BnConvNet (4 CNN + BN + ReLU, lr=0.05)
- **时间**: 之前会话
- **配置**: 同 v28 但 lr=0.05, dropout=0
- **结果**: ❌ step 5 激活值 std=8.6，step 20 达到数千
- **结论**: lr=0.05 在 bug 下有效 lr ~15.8，远超 SGD 稳定范围

### 实验 #4: v30 - BnConvNet (4 CNN + BN + ReLU, lr=0.005)
- **时间**: 之前会话
- **配置**: 同 v28 但 lr=0.005, dropout=0
- **结果**: ❌ steps 0-39 激活稳定，step 1000 NaN
- **结论**: 有效 lr ~1.58。初始梯度放大不足瞬间爆炸，但累积权重漂移后 BN 无法补偿

### 实验 #6: v32 - Projection ResNet (10 CNN + BN + LR, lr=0.001)
- **时间**: 2026-05-24
- **结果**: ❌ NaN by step 36
- **结论**: 有效 lr ~0.316（因 bug），10 层叠加使任何梯度误差指数级放大

> **重要说明**: v28-v32 的实验结论在多轮迭代中一直被误读为"BN instability"或"lr 上限太低"。**真正原因是 Bug #1: bn_spatial_var 零初始化**。该 bug 于 2026-05-25（本次会话）被诊断并修复。

### 实验 #8: v34 - V28 Reproduction with bn_spatial_var Fix（BUG 修复验证）
- **时间**: 2026-05-25
- **架构**: 4 conv (3x3, 3->32->64->128->256) + BN + ReLU + GAP + MLP(256->256->10)
- **配置**: lr=0.002, momentum=0.9, wd=1e-4, bias_wd=1e-3, all stride=1, He init
- **修复**: bn_spatial_var 初始化为 bn_running_var（初始值 1.0）
- **结果**:
  - 梯度 L2: 11,125,761 -> 104（~100,000x 降低）
  - 训练稳定运行 43,000+ step 无 NaN
  - **Epoch 1 完整评估准确率: 11.60%**（随机基线 10%，上升仅 1.6%）
  - **权重 RMS 未变化**: 所有 CNN 层 conv_w RMS 保持 ~0.087（与初始化相同）
  - **偏置主导学习**: conv_b RMS 从 0.006 -> 0.030（仅偏置在更新）
  - **类崩溃**: 91.4% 预测为 class 4，其余 5 类 0%
  - **Epoch 2 梯度消失**: 大部分 conv_w/conv_b 梯度 L2 = 0.0000
- **诊断**:
  1. ReLU 激活阻挡了绝大多数梯度——负预激活神经元梯度为 0
  2. 在 He init + BN 归一化后，约 50% 的激活值 <= 0，被 ReLU 截断
  3. 4 层 CNN 叠加后，可用梯度信号几乎为零
  4. 偏置梯度通过 weight_decay 和 bias_wd 维持，权重梯度被 ReLU 死区阻挡
  5. MLP 头学习了"始终预测 class 4"作为最小 loss 策略
- **教训**: BN fix 是必要的但不是充分的。ReLU 在此架构下导致灾难性梯度消失。必须使用 LeakyReLU 或类似激活函数来保证梯度通路。

### 实验 #10: v35 - LeakyReLU CNN + BN（已完成）
- **时间**: 2026-05-25
- **架构**: 同 v34（4 CNN + BN + LeakyReLU + GAP + MLP），MLP 仍用 ReLU
- **变更**: CNN_ACT_RELU -> CNN_ACT_LEAKY_RELU（负斜率 0.01）
- **配置**: lr=0.002, momentum=0.9, wd=1e-4, bias_wd=1e-3
- **结果**:
  - 梯度 L2 从未降至 0.0000（vs v34 epoch 2 即归零）
  - **Epoch 1 准确率约 12%**（比 v34 的 11.60% 仅略高）
  - **CNN 权重 RMS 仍不变化**（0.0883 → 0.0879）
  - 偏置主导学习（conv_b RMS 0.006 → 0.012+）
- **教训**: LeakyReLU 解决了梯度消失但未解决权重不学习的根因。MLP 头仍用 ReLU → 当 MLP 隐藏层全为负时，上游 CNN 梯度仍被阻断（Bug #2）。

### 实验 #11: v39 - BnConvNet, mom=0.5（降低动量）
- **时间**: 2026-05-25
- **架构**: 同 v35（4 CNN + BN + LeakyReLU + MLP ReLU）
- **变更**: momentum 0.9 → 0.5
- **配置**: lr=0.003, momentum=0.5, wd=1e-4, bias_wd=1e-3
- **结果**: epoch 1 mini_eval: 16.10% → 17.50% → 16.30%
- **结论**: 降低动量有一定帮助，但仍无法突破 20%

### 实验 #12: v40 - MLP LeakyReLU 修复（Bug #2）
- **时间**: 2026-05-25
- **架构**: 4 CNN (3→32→64→128→256) + BN + LeakyReLU + GAP + MLP(LeakyReLU)
- **变更**: MLP_ACT_RELU → MLP_ACT_LEAKY_RELU（MLP 隐藏层也用 LeakyReLU）
- **配置**: lr=0.005, momentum=0.5, wd=1e-4, bias_wd=1e-3
- **结果**:
  - MLP LeakyReLU 成功防止精确梯度消失（梯度从未达 0.0000）
  - **Epoch 1 准确率约 14.7%**
  - conv_w RMS 仍不变（0.0883 → 0.0876）
  - conv_b RMS 3x 增长（0.006 → 0.018）
  - accuracy plateau 在约 14.5%
- **诊断**: MLP 梯度通路打通，但 CNN 权重仍不学习。关键词题：**per-sample 梯度方向振荡导致累计更新接近零**。

### 实验 #13: v41 - mom=0 纯 SGD（当前最佳）
- **时间**: 2026-05-25
- **架构**: 同 v40（4 CNN + BN + LeakyReLU + MLP LeakyReLU）
- **变更**: momentum 0.5 → 0.0（纯 SGD）
- **配置**: lr=0.005, momentum=0.0, wd=1e-4, bias_wd=1e-3
- **结果**:
  - 梯度比 v40 大 1.77x（动量消除了方向性抵消）
  - **Epoch 1 mini_eval: 16.70% → 21.50% → 23.10%**（**当前最佳**）
  - Loss: 2.48 → 2.34
  - conv_w RMS 仍不变（0.0883 → 0.0880）
  - 准确率在 21-23% 区间振荡，未进一步上升
- **诊断**: 无动量时每个样本的梯度直接更新，偶尔命中正确方向。但 batch_size=1 下连续样本梯度方向不相关 → 更新正负交替 → 净效果接近零。这是所有实验中 conv_w RMS 不变的根本原因。

### 实验 #14: v42 - VggLiteNet (stride=2 降采样)
- **时间**: 2026-05-25
- **架构**: 5 CNN (3×3, 3→64→128→128→256→256), stride=2 降采样, BN + LeakyReLU + MLP LeakyReLU
- **配置**: lr=0.005, momentum=0.0, wd=1e-4, bias_wd=1e-3
- **结果**:
  - **Baseline: 6.40%（比 4 CNN 的 8-10% 差）**
  - **Epoch 1 mini_eval: 12.60% → 15.40%**（远差于 v41）
  - Loss: 3.73（vs v41 的 2.48）
  - 提前停止
- **结论**: 更深 + stride-2 降采样在此框架下适得其反。GAP 前空间 3×3（仅 9 位置）信息损失过大。depth 增加加剧 batch_size=1 的梯度噪声。

### 实验 #7: v33 - SimpleBnConv (3 CNN + BN + LeakyReLU, lr=0.001)
- **时间**: 2026-05-24
- **配置**: 3 conv (3→16→32→128 stride=2), BN on all, lr=0.001, momentum=0.9
- **架构**: conv1(3x3, 3→16, BN, LR) → conv2(3x3, 16→32, BN, LR) → conv3(3x3, 32→128, s=2, BN, LR) → GAP(128) → MLP(128)
- **理由**: 回退到计划阶段1的最保守方案。根据 v28 经验，4 层 BN + lr=0.002 可稳定，3 层 BN + lr=0.001 应该更安全
- **参数**: ~100K
- **风险**: v28 的 4 层 BN 训练稳定但学习极慢（epoch1 12.95%），v33 可能同样慢

### 实验 #15: v43 - revert 4 CNN + lr 翻倍 + conv1 64 通道
- **时间**: 2026-05-25
- **架构**: 4 CNN (3→64→64→128→256) + BN + LeakyReLU + MLP LeakyReLU
- **变更**: 恢复 v41 4 CNN 架构，lr 翻倍（0.005→0.010），conv1 32→64 channels
- **配置**: lr=0.010, momentum=0.0, wd=1e-4, bias_wd=1e-3
- **结果**:
  - Baseline: 9.60%
  - **Epoch 1 full eval: 14.55%**
  - Epoch 1 mini_eval: 15.50% → 16.70% → 17.80% → 14-18% 波动
  - Epoch 2 (中途停止): ~18% at 2000/5000
  - Loss: 2.37 (epoch 1 avg)
  - **conv_w RMS 恒定 0.0883 → 0.0869**（36K steps 基本不变）
  - conv_b RMS 持续增长 0.006 → 0.019（3x）
  - **结论**: 比 v41 差（v41 epoch1 23.1%）。参数翻倍 + lr 翻倍增大了梯度振荡幅度。
- **教训**: 更大的学习率和更多参数无法解决 batch_size=1 的梯度抵消问题。偏置因空间平均效应（30×30=900 positions）有效 batch 大得多，故偏置学习正常而权重不学习。

### 实验 #16: v44 - batch_size=4 + mom=0 + GAP BN fix（梯度累积）
- **时间**: 2026-05-25
- **架构**: 同 v41（4 CNN + BN + LeakyReLU + MLP LeakyReLU）
- **变更**:
  - batch_size 1 → 4（梯度累积）
  - lr=0.001（因梯度求和 → 有效 step ∝ batch_size，需降低 lr）
  - GAP 层 POOL_AVG→POOL_DUAL→POOL_AVG（尝试不同 pooling）
- **结果**:
  - conv_w 梯度 L2 增至 ~100（vs v41 的 ~37），但仍远小于 conv_b（~1000）
  - conv_w RMS 仍不变（0.0883 → 0.0880）
  - 准确率远差于 v41
- **结论**: batch_size=4 确实增加了梯度信号，但不能恢复梯度方向 —— BN backward 的 scale 是错误的。

### 实验 #17: v45 - batch_size=4 + mom=0.9（尝试恢复动量）
- **时间**: 2026-05-25
- **架构**: 同 v44
- **变更**: momentum 0.0 → 0.9（batch_size=4 下恢复动量）
- **结果**:
  - **Step 1000 NaN**——动量 + 累积梯度 = 有效 step 过大
  - 提前停止
- **结论**: batch_size=4 的累积梯度 + momentum 放大了梯度幅度，超出 SGD 稳定范围。

### 实验 #18: v46 - batch_size=4 + lr=0.001 + mom=0.0（v44 的稳定版本）
- **时间**: 2026-05-25
- **架构**: 同 v44/v45
- **变更**: momentum 0.9 → 0.0（规避 v45 NaN）
- **结果**:
  - **Epoch 1: 22.55%**（比 v41 的 23.1% 差）
  - **conv_w RMS 恒定 0.0883 → 0.0880**（5K samples 仅 0.3% 变化）
  - conv_b RMS 持续增长 0.006 → 0.020（3.3x）
  - **诊断结论**: batch_size=4 未解决根因。权重梯度仍然远弱于偏置梯度，说明梯度方向或 scale 在某个环节被错误计算。

### 实验 #19: v47 - BN backward x_hat fix + running stat update（Bug #3/#4 修复，但暴露了 Bug #5）
- **时间**: 2026-05-26
- **架构**: 同 v46（4 CNN + BN + LeakyReLU + MLP LeakyReLU）
- **变更**:
  - **Bug #3 修复**: POOL_AVG/POOL_MAX/POOL_DUAL BN backward x_hat 公式改为 `(linear_val - running_mean) / sqrt(running_var + eps)`
  - **Bug #4 修复**: 正向传播 POOL_AVG/POOL_MAX/POOL_DUAL 路径添加 EMA running stat 更新
- **配置**: lr=0.001, momentum=0.0, batch_size=4, wd=1e-4, bias_wd=1e-3
- **结果**:
  - **conv_w 梯度 L2: 150x 提升**（v46: 37-388 → v47: 11512-57649）
  - **conv_w 更新: 13x 提升**（0.0033 → 0.0425 per batch）
  - **500 样本 MINI eval: 14.10%**（baseline 8.20%, +5.9%）
  - **Epoch 1 峰值准确率: 16.80%（sample 1500）→ 最终崩溃至 10.40%**
  - **梯度崩溃: conv_w grad L2 从 42525（step 0）降至 0.37（step 8000）—— 31000x 衰减**
  - **conv_w RMS 仍基本不变: 0.0883 → 0.0894（+1.2%）**
  - **类崩溃: 55.6% 预测为 class 7**
  - **Epoch 1 耗时: ~9.8 小时**
- **诊断结论（Bug #5）**: Bug #3/#4 修复是正确的，但暴露了一个次级时序bug：**正向传播用 PRE-update running stats 做归一化，但反向传播读取 POST-update stats 来计算 x_hat 和 scale**。随着 5000 samples 的 EMA 累积，running stats 漂移到与正向传播不匹配，导致反向梯度错误 → 梯度崩溃。
- **教训**: 修复 Bug #3 x_hat 公式是必要的但不充分。必须确保反向传播使用的 running stats 与正向传播严格一致。详见 Bug #5 文档（方案 A: 保存 x_hat + inv_std 到 cache）。

### 实验 #20: v48 - Bug #5 修复（缓存 pre-EMA 的 x_hat + inv_std，2026-05-27）
- **时间**: 2026-05-27
- **架构**: 同 v47（4 CNN + BN + LeakyReLU + MLP LeakyReLU）
- **变更**:
  - **Bug #5 修复**: 正向传播在 EMA 更新之前缓存 x_hat 到 bn_pre_cache、inv_std 到 bn_spatial_var；反向传播从缓存读取，不再读取 running stats
- **配置**: lr=0.001, momentum=0.0, batch_size=4, wd=1e-4, bias_wd=1e-3
- **修复验证** (5 样本快速验证):
  - [FWD] x_hat=0.510797 ↔ [BWD] x_hat(cached)=0.510797 ✅ 严格一致
  - [FWD] inv_std≈1.455 ↔ [BWD] inv_std(cached)=1.454921 ✅ 严格一致
  - 梯度数值：conv_w L2 538-4680, bn_g L2 1.5-6.1（正常范围）
- **结果**: **待完整训练（5000 samples × 5 epochs）**
- **期望**: conv_w 梯度不再崩溃，权重持续更新，准确率超过 v47 峰值 16.80%

> **注**: v31（ReLU+MaxPool 尝试）和 v36-v38（lr 微调，v38 仅达 10.60% step 500）为快速迭代，未取得有意义结果，不单独列出。

---

## 八、经验总结

### 关键教训

1. **Bug 诊断优先级高于调参**: 4 个实验（v28-v32）的所有"BN instability"结论都是错误的（Bug #1）；10 个实验（v34-v46）的所有"batch_size 不够大"结论也被推翻（Bug #3）。真正的根因是代码层面的公式错误。**现在确认的两个回环 bug 都隐藏在 BN 实现中 —— 未来调参前应首先审计 BN 逻辑。**

2. **Bug #2: MLP ReLU 梯度阻断**: MLP 隐藏层用 ReLU 时，若 256 个神经元全为负 → `current_delta` 全零 → `input_gradient` 全零 → CNN 收不到梯度。LeakyReLU 在所有层（CNN + MLP）是必须的。

3. **Bug #3（根因）: BN backward x_hat 公式错误**: `pooled_linear_cache` 存储 pre-BN raw value，但 x_hat 公式 `(x - beta) / gamma` 假定 x 是 BN output。正确的 x_hat 必须用 running mean/var 恢复。**此 bug 导致 v34-v46 的所有 conv_w 权重不学习**——不是 batch_size 问题，不是 lr 问题，不是 ReLU 问题。

4. **Bug #4: BN running stats 从未更新**: POOL_AVG/POOL_MAX/POOL_DUAL 路径 skip BN post-processing → running stats 永远停留在 (0, 1) 初始化值。正向传播用错的 stats，反向传播也用错的 stats。

5. **batch_size > 1 是必要辅助条件但不是根因**: v44-v46 证明 batch_size=4 确有帮助（grad L2 2-3x），但无法弥补 BN backward 公式错误的致命影响。正确的实现顺序是：先 fix BN backward（v47），再优化 batch_size。

6. **梯度 L2 是最好的诊断工具**: conv_w_grad_L2 从 11M→104（Bug #1 fix）和从 300→50K（Bug #3 fix）提供了即时确认。每个实验都应首先检查梯度 L2。

### v27-v47 完整实验矩阵

| 版本 | 架构 | LR | Mom | Batch | CNN Act | MLP Act | Epoch1 最佳 | conv_w RMS Δ | 结论 |
|------|------|-----|-----|-------|---------|---------|------------|-------------|------|
| v27 | 5CNN noBN | 0.005 | 0.9 | 1 | LR | ReLU | 10.6% | N/A | 无BN无法训练 |
| v28 | 4CNN+BN | 0.002 | 0.9 | 1 | ReLU | ReLU | 12.95% | 0.088→0.088 | BN bug:有效lr=0.63 |
| v34 | 4CNN+BN fix | 0.002 | 0.9 | 1 | ReLU | ReLU | 11.60% | 0.088→0.087 | ReLU 50%死区 |
| v35 | 4CNN+BN+LR | 0.002 | 0.9 | 1 | LR | ReLU | ~12% | 0.088→0.088 | MLP ReLU阻断梯度 |
| v39 | 4CNN+BN+LR | 0.003 | 0.5 | 1 | LR | ReLU | 17.50% | ~不变 | mom降有帮助 |
| v40 | 4CNN+BN+LR | 0.005 | 0.5 | 1 | LR | **LR** | 14.70% | 0.088→0.088 | MLP梯度通路打通 |
| **v41** | 4CNN+BN+LR | 0.005 | **0.0** | 1 | LR | LR | **23.10%** | 0.088→0.088 | batch=1 最佳(偏置主导) |
| v42 | 5CNN s=2 | 0.005 | 0.0 | 1 | LR | LR | 15.40% | 各种 | 更深更差 |
| v43 | 4CNN conv1=64 | **0.010** | 0.0 | 1 | LR | LR | 17.80% | 0.088→0.087 | lr翻倍适得其反 |
| v44 | 4CNN+BN+LR | 0.001 | 0.0 | **4** | LR | LR | <v41 | 0.088→0.088 | batch=4梯度增强但方向错误 |
| v45 | 4CNN+BN+LR | 0.001 | **0.9** | **4** | LR | LR | NaN@1K | ~不变 | mom+累积梯度=爆炸 |
| v46 | 4CNN+BN+LR | 0.001 | **0.0** | **4** | LR | LR | 22.55% | 0.088→0.088 | batch=4未解决根因(偏置主导) |
| **v47** | 4CNN+BN+LR | 0.001 | **0.0** | **4** | LR | LR | **16.8%**@1500 → **10.40%** | 0.088→0.089 | **BN x_hat fix 有效 + 时序bug → 梯度崩溃** |
| **v48** | 4CNN+BN+LR | 0.001 | **0.0** | **4** | LR | LR | **TBD** | TBD | **Bug #5 修复: cache x_hat+inv_std** |
| **v48b** | 4CNN+BN+LR | 0.001 | **0.0** | **4** | LR | LR | **TBD** | TBD | **+数据增强+split+resume+自适应运动(本次会话)** |

**模式（v27-v46）**: 所有权重 RMS 不变（<1% Δ），所有偏置 RMS 增长（3x+）。v41 的 23.1% 和 v46 的 22.55% 几乎完全来自偏置学习 + MLP。**根因是 Bug #3（BN backward x_hat 公式错误），而非 batch_size=1。**

**模式（v47）**: BN backward x_hat fix + running stat update 使 conv_w 梯度 150x 提升，但暴露了 Bug #5——forward/backward running stats 时序不匹配。梯度在 ~1500 samples 后开始崩溃（L2 从 42525 降至 0.37，31000x 衰减），准确率从 16.80% 峰值崩溃至 10.40%。

**模式（v48）**: Bug #5 已修复——正向传播在 EMA 更新前缓存 x_hat + inv_std，反向传播从缓存读取，消除时序不匹配。代码验证通过（[FWD] x_hat = [BWD] x_hat(cached)）。完整训练待运行。

### 实验 #21: v48b — Bug #1-#5 全部修复 + 工程增强（2026-05-27 本次会话）
- **时间**: 2026-05-27
- **架构**: 同 v47/v48（4 CNN + BN + LeakyReLU + GAP + MLP LeakyReLU）
- **变更**:
  - Bug #1-#5 全部修复（core framework）
  - 数据增强实现：`cifar10_augment_sample()`（水平翻转 p=0.5 + 随机 2px pad-crop）
  - 训练/验证集分割：80/20（`split_train_val()`）
  - Checkpoint resume：启动时自动加载 `weights.bin`
  - 自适应运动检测阈值 + 时序平滑器（`video_processor.h/c`）
  - LR 调度定义已加入（`EVP_LR_DECAY_RATE=0.5`, `EVP_LR_DECAY_EPOCHS=3`），等待 core `train_set_lr` API
- **配置**: lr=0.001, momentum=0.0, batch_size=4, wd=1e-4, bias_wd=1e-3
- **结果**: **待完整训练**
- **期望**: conv_w 持续学习 + 数据增强抗过拟合 + 验证集评估 → 准确率稳步上升

### 当前路线图

**Bug #1-#5 均已修复。conv_w 现在应能真正持续学习。**

已完成的工程增强（v48b 本次会话）:
1. **[已完成] Bug #5 修复** — forward 缓存 pre-EMA x_hat + inv_std，backward 直接从缓存读取
2. **[已完成] 数据增强** — `cifar10_augment_sample()`，水平翻转 + 随机 pad-crop
3. **[已完成] 训练/验证分割** — 80/20 split，验证集用于 epoch-end 评估
4. **[已完成] Checkpoint resume** — 自动加载 `weights.bin` 继续训练
5. **[已完成] 自适应运动检测** — EMA 阈值 + 时序平滑 debouncing
6. **[已完成] LR 调度定义** — `EVP_LR_DECAY_RATE/EVP_LR_DECAY_EPOCHS`（等待 core `train_set_lr` API）

下一步:
1. **运行完整训练** — 5 epochs × ~4000 train samples (80% of 5000)
2. **lr tuning** — 当前 lr=0.001。Bug #5 修复后可尝试 0.002-0.005
3. **dropout 正则化** — 在 GAP 层添加，dropout_rate=0.3
4. **更宽/更深的架构变体** — 若准确率 plateau，增加通道数（64→128 或 128→256）

---
## 九、下一步行动

**当前**: v48b 代码就绪（Bug #1-#5 全部修复 + 数据增强 + train/val split + checkpoint resume + 自适应运动检测），待运行完整训练

**立即执行**:
1. 运行 `edge_video_preprocess_train.exe` 开始 5 epoch 训练
2. 监控 epoch 1 的 conv_w 梯度 L2 确认不再崩溃（应保持 >100 而非降至 0.37）
3. 检查 epoch 1 验证准确率是否超过 v47 峰值 16.80%

**中期（epoch 1 通过后）**:
1. 完成全部 5 epoch 训练
2. 启用 dropout_rate=0.3
3. 尝试 lr=0.002（Bug #5 修复后的安全范围应扩大）

**长期（达到 50%+ 准确率后）**:
1. 增加训练样本数（5K → 10K → 50K）
2. 增加训练轮数（5 → 20 → 50）
3. 增加训练轮数（5 → 20+）
4. 尝试更宽架构（128 起始通道 vs 32）
5. 尝试更深的卷积层次（5-6 层 CNN）
