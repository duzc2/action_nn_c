# 网络结构深度研究：论文细节与实现指南

> 配套文档：`network_architecture_roadmap.md`（结构路线图）
> 最后更新：2026-05-27

本文档深入展开路线图中各项技术的最新论文细节、数学公式与C语言实现注意事项。

---

## 1. minLSTM / minGRU — "Were RNNs All We Needed?"

**论文：** Feng et al., ICLR 2025 (arXiv:2410.01201)
**代码：** [github.com/BorealisAI/minRNNs](https://github.com/BorealisAI/minRNNs)

### 核心简化

标准 GRU/LSTM 的门控依赖 `h_{t-1}` 是 BPTT 串行训练的根源。minLSTM/minGRU 的核心洞察：**去掉门控函数中的 `h_{t-1}` 依赖**后，所有gate可以直接用 parallel scan 并行计算。

#### minGRU 形式

| 组件 | 标准 GRU | minGRU |
|---|---|---|
| Update gate | `z_t = σ(Linear([x_t, h_{t-1}]))` | `z_t = σ(Linear_dh(x_t))` |
| Reset gate | `r_t = σ(Linear_dx([x_t, h_{t-1}]))` | **已移除** |
| Candidate state | `h̃_t = tanh(Linear([x_t, r_t⊙h_{t-1}]))` | `h̃_t = Linear_dx(x_t)` |
| Hidden recurrence | `h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t` | **完全相同**（但现在可并行） |

#### minLSTM 形式

| 组件 | 标准 LSTM | minLSTM |
|---|---|---|
| Forget gate | `f_t = σ(Linear_dh([x_t, h_{t-1}]))` | `f̃_t = σ(Linear_dh(x_t))` |
| Input gate | `i_t = σ(Linear_dx([x_t, h_{t-1}]))` | `ĩ_t = σ(Linear_dx(x_t))` |
| Output gate | `o_t = σ(Linear_dh([x_t, h_{t-1}]))` | **已移除** |
| Gate normalize | — | `f_t = f̃_t/(f̃_t+ĩ_t)`, `i_t = ĩ_t/(f̃_t+ĩ_t)` |
| State recurrence | `c_t = f_t⊙c_{t-1} + i_t⊙c̃_t` | `h_t = f_t⊙h_{t-1} + i_t⊙h̃_t` |

#### Parallel Scan 实现

两者的循环均可表达为统一形式：

```
v_t = a_t ⊙ v_{t-1} + b_t
```

- **minGRU:** `a_t = 1 - z_t`, `b_t = z_t ⊙ h̃_t`
- **minLSTM:** `a_t = f_t`, `b_t = i_t ⊙ h̃_t`

Parallel prefix scan 使用关联二元操作符：

```
(A₂, B₂) ⊕ (A₁, B₁) = (A₂·A₁, A₂·B₁ + B₂)
```

#### 对数空间实现（数值稳定性）

标准方式（指数空间）容易溢出。实践中用对数空间：

```
log(σ(x)) = -softplus(-x)

minGRU log-space:
  log(z) = -softplus(-k)     // k = W_z · x
  log(1-z) = -softplus(k)

minLSTM log-space:
  log(f') = -softplus(softplus(-p) - softplus(-k))
  log(i') = -softplus(softplus(-k) - softplus(-p))
```

Scan 使用 `logcumsumexp` 解析循环。

#### C 语言实现要点

1. 需要实现 `softplus(x) = log(1 + exp(x))`（数值稳定版）
2. Parallel scan 实现标准的 up-sweep + down-sweep（Blelloch scan）
3. gate 计算阶段完全并行（只依赖 x_t），仅 scan 阶段需要 log T 步
4. 参数：`W_z ∈ R^{hidden×input}` + `W_h ∈ R^{hidden×input}`，无 `W_hh` 权重

---

## 2. LAuReL — Learned Augmented Residual Layer

**论文：** Menghani et al., ICML 2025 (arXiv:2411.07501) — Google
**代码：** [openreview.net/forum?id=rUDRWP9WvZ](https://openreview.net/forum?id=rUDRWP9WvZ)

### 三个变体

#### LAuReL-RW（Residual Weights）— 最小改动

```
y = α · F(x) + β · x          // α, β 用 sigmoid/softmax 归一化
```

仅 +2 个参数/层（0.003% ResNet-50），获得接近多一层的收益。

#### LAuReL-LR（Low-Rank）

```
y = F(x) + A(Bᵀx)              // A ∈ R^{D×r}, B ∈ R^{D×r}
```

低秩矩阵为残差路径增加线性学习能力，`r=16` 典型值。

#### LAuReL-PA（Previous Activations）

```
y = F(x) + Σᵢ₌₁ᵏ γᵢ · rᵢ + g̃(x)   // rᵢ 是前 k 层的残差输出
```

#### 组合使用

| 变体 | 公式 | 额外参数 |
|---|---|---|
| **RW** | `y = α·F(x) + β·x` | 2 |
| **RW+LR** | `y = α·F(x) + β·(A(Bᵀx))` | 2rD+2 |
| **RW+LR+PA** | `y = α·F(x) + β·(Σγᵢrᵢ + A(Bᵀx))` | 2rD+k+2 |

#### 关键结果

| 实验 | 提升 | 参数增加 |
|---|---|---|
| ResNet-50 ImageNet (RW) | +0.15% acc | 0.003% |
| ResNet-50 (RW+LR, r=16) | +0.25% acc（匹敌多一层） | 1.68% |
| 1B LLM 预训练 | +2.5~20% 下游任务 | 0.012% |

#### C 语言实现要点

1. 最优先实现 LAuReL-RW：forward 时额外 2 个标量乘法
2. backward 时 α、β 需各自求导
3. 归一化：`α = sigmoid(α_raw)`, `β = 1 - α`（或 `softmax`）
4. 初始化：`α_raw = 0` → 初始时 α ≈ 0.5, β ≈ 0.5

---

## 3. 多头注意力 + GQA + KHA

### 3.1 GQA — Grouped-Query Attention

**论文：** Ainslie et al., 2023 (arXiv:2305.13245) — Google

```
h 个 query head → g 个 KV head (g ≤ h)
每个 group 内的所有 query 共享一对 K, V
```

KV cache 缩减：`(h-g)/h`。GQA-8 (64→8)：88% 减少。

Llama 3/3.1、Qwen 2.5、Gemma 3、Mistral Large 均使用 GQA。

### 3.2 KHA — Knocking-Heads Attention

**论文：** Zhou et al., Oct 2025 (arXiv:2510.23052)

**核心机制：** 在 Q/K/V 投影后插入共享变换层：

```
Q̃_i = Q_i · T_Q       // T_Q ∈ R^{d_k×d_k} 所有头共享
K̃_i = K_i · T_K
Ṽ_i = V_i · T_V
```

**关键技巧 — 对角初始化：** `T_Q/T_K/T_V` 初始化为单位矩阵，训练初期行为等价于标准 MHA。随训练推进，非对角元素逐渐发展跨头交互。

**推理零开销：** 线性变体的共享矩阵可吸收进原始投影权重（`W'_i = W_i · T`），推理时无任何额外 FLOPs。

**训练开销：** 仅总 FLOPs 的 ~0.5-1%。

### C 语言实现要点

1. MHA 改造：将现有单头改为 `num_heads` × `head_dim` 分块
2. GQA：增加 `num_kv_heads`，≤ `num_heads`；计算 K/V 时只算 g 组投影
3. KHA（可选）：在投影后乘 `T_Q/T_K/T_V`，对角初始化
4. 推理时 KHA 的 T 矩阵直接左乘到 W 中，消除推理开销
5. 建议先抽象出独立 `nn/attention/` 模块

---

## 4. 归一化 — RMSNorm、Peri-LN、FlashNorm

### 4.1 RMSNorm — 工业标准

**论文：** Zhang & Sennrich, 2019

```
RMSNorm(x) = x / RMS(x) · γ       // 无均值减法
RMS(x) = sqrt(mean(x²) + ε)
```

2026 年研究 (Gupta et al.) 证明 LayerNorm 的均值减法冗余：LLM 的自然操作方向与均匀向量正交，被减去的分量平均为 ~0。

### 4.2 Peri-LN — 训练稳定性

**论文：** Kim et al., ICML 2025 (arXiv:2502.02732) — KAIST/NAVER Cloud

**核心问题：** Pre-LN 导致隐藏状态指数方差增长 → FP16 溢出 → 训练崩溃（OPT-175B 在 9 天内重启 40+ 次）。

```
Post-LN:    x + Norm(Module(x))        ← 梯度消失
Pre-LN:     x + Module(Norm(x))        ← 激活爆炸
Peri-LN:    x + Norm(Module(Norm(x)))  ← 线性方差增长 + 梯度自正则
```

已被 **Gemma 2/3**、**OLMo 2** 静默采用。

### 4.3 FlashNorm — 推理加速

**论文：** Graef et al., Apr 2026 (arXiv:2407.09577v4) — OpenMachine

三个数学等价变换（无需重训练）：

1. **权重折叠：** `W*[i,j] = g[i] × W[i,j]`，消除独立 norm 权重张量
2. **延迟归一化：** `(a × W*) × (1/RMS(a))`，矩阵乘与 RMS 并行执行
3. **RMSNorm 级联消除：** `RMSNorm→Linear→RMSNorm` 中首层可完全消除

结果：Llama-7B 推理 12-14% 加速（T4 GPU）。

### C 语言实现要点

1. RMSNorm 实现最简单：计算 mean(x²) + sqrt + 逐元素除法 + 乘以 γ
2. Peri-LN 模式：MLP/CNN/Transformer 中统一采用 `Norm_after(Module(Norm_before(x)))` + 残差
3. FlashNorm：对已训练的模型加载时可折叠权重（`W_folded[i,*] = γ[i] * W[i,*]`）
4. LayerNorm 如果需要：`y = (x - mean(x)) / sqrt(var(x) + ε) * γ + β` 但优先用 RMSNorm

---

## 5. Dropout + Stochastic Depth

### 5.1 经典 Dropout

```
训练:  output = mask(keep_prob) * x / keep_prob   // dropout_mask
推理:  output = x                                    // 直接通过（或乘以 keep_prob）
```

### 5.2 Stochastic Depth / DropPath

```
训练:  if random() < p_drop:
         output = x           // 跳过整层
       else:
         output = F(x) / (1 - p_drop)
推理:  output = F(x)          // 全层通过
```

深度越深，drop 概率越大（线性或指数衰减）。

### 5.3 Lipschitz-Guided Stochastic Depth

**论文：** Nayal et al., Sep 25 (arXiv:2509.10298)

Principled 方法，从目标 Lipschitz 常数推导 drop 概率：

```
p(l) = 1 - κ_target^(l/L)    // l 为层编号（深→大）, L 为总层数
```

ViT-Tiny (CIFAR-10)：清洁准确率 90.88% vs 91.0% baseline，**AutoAttack 鲁棒性 28.5% → 31.1%**，**FLOPs 减少 75%**。

### 5.4 PerNodeDrop

**论文：** Dec 2025 (arXiv:2512.12663)

每次前向传播对每个神经元独立抽样，保留有益的共生适应模式，抑制有害过拟合。

### C 语言实现要点

1. 共享模块 `nn/dropout/`：提供 `dropout_forward(float *x, int n, float keep_prob, uint32_t *seed)` 接口
2. 训练/推理模式：通过全局标志或上下文参数切换
3. 随机数：使用 PCG 或 xoshiro256** 高效 RNG（避免 libc `rand()`）
4. DropPath 需要知道层索引以决定概率调度
5. 实现 `rescale` 除法而非乘法（推理更简单）

---

## 6. GNN 增强 — Edge-Conditioned GATv2

**论文：** Zhang et al., Sep 2025 (mmWave IAB deployment)

### 数学完整定义

对于每层每头 k（K 个头）：

```
① 投影:
   ĥᵢᵏ = Wₖ · hᵢ        // 源节点
   ĥⱼᵏ = Wₖ · hⱼ        // 目标节点
   êᵢⱼᵏ = Wₑ,ₖ · eᵢⱼ    // 边特征 ← 核心创新

② 三方注意力得分:
   α'ᵢⱼᵏ = LeakyReLU(aₖᵀ · [ĥᵢᵏ ∥ ĥⱼᵏ ∥ êᵢⱼᵏ])

③ Softmax:
   αᵢⱼᵏ = exp(α'ᵢⱼᵏ) / Σᵤ exp(α'ᵢᵤᵏ)

④ 消息聚合 + 残差:
   mᵢᵏ = Σⱼ αᵢⱼᵏ · ĥⱼᵏ
   hᵢ' = ∥ₖ σ(mᵢᵏ) + hᵢ      // 多头拼接 + 残差
```

### 版本对比

| 模型 | 注意力输入 | 边感知 | 注意力类型 |
|---|---|---|---|
| Vanilla GAT | `[Wxᵢ ∥ Wxⱼ]` | 否 | 静态 |
| GATv2 | `W[xᵢ ∥ xⱼ]` | 否 | 动态 |
| Edge-Conditioned GATv2 | `[Whᵢ ∥ Whⱼ ∥ Wₑeᵢⱼ]` | **是** | 动态 |

### 其他 GNN 前沿方向

| 方向 | 论文 | 要点 |
|---|---|---|
| **Dir-Poly** | Gupta et al., Aug 2025 | 有向图：分别聚合入邻居/出邻居，L-多项式注意力 |
| **EB-GNN** | Barceló et al., Oct 2025 | 以**边**为中心传递消息，可证明 > 1-WL 表达力 |
| **GESC** | Choi et al., Nov 2025 | 复数域自干扰消除 → `ĥ⊥ = ĥ - proj_h(ĥ)` 消除冗余分量 |

### C 语言实现要点

1. 新增 `GnnAggregatorType` 枚举值：`SUM`、`MAX`、`ATTENTION`
2. 新增 `GnnConfig.edge_dim` 和 `W_e` 权重矩阵
3. Attention 聚合器复用 `nn/attention/` 模块
4. 保持向后兼容：`edge_dim=0` 时回退到旧行为

---

## 7. Mamba-2 / SSD — State Space Duality

**论文：** Dao & Gu, ICML 2024 (arXiv:2405.21060)
**代码：** [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

### 核心洞察：SSM = 结构化掩码注意力

选择性 SSM（标量 A）等价于乘以 1-半可分矩阵 L：

```
Y = (L ∘ C·Bᵀ) · X
```

其中 L 通过 `logcumsumexp` 从 A 构造。

### 分块算法（4 步）

将序列分为长度 Q 的块（Q=64）：

| 步骤 | 操作 | 硬件 |
|---|---|---|
| **Step 1**: 块内 | 注意力形式，初始状态=0 | Tensor Core 矩阵乘（批量） |
| **Step 2**: 块终态 | 当前块的最终状态 | Tensor Core 矩阵乘（批量） |
| **Step 3**: 跨块循环 | 在 T/Q 个元素上 scan（如16-32个块） | Scan（向量单元） |
| **Step 4**: 状态→输出 | 每个块的初始状态贡献 | Tensor Core 矩阵乘（批量） |

**关键：** 只有 Step 3 需要 scan，且仅在 ~32 个元素上执行；其余全部是 GPU 友好的矩阵乘。

### Mamba-1 vs Mamba-2

| | Mamba-1 (S6) | Mamba-2 (SSD) |
|---|---|---|
| A 矩阵 | 对角（N×N/通道） | 标量×单位矩阵 |
| 状态维度 N | 16 | 64-256 |
| 头维度 P | 1 | 64-128 |
| 训练速度 | 基线 | **2-8× 更快** |
| 长序列(16K) | — | 比 FlashAttention2 快 **6×** |

### segsum 原语（关键陷阱）

用 `ratio of cumprods` 或 `log-space + cumsum difference` 会因 catastrophic cancellation 导致 NaN。必须实现只使用加法的稳定分段求和。

### C 语言实现要点

1. 需要关联 scan 基础操作（同 minLSTM 的 parallel scan）
2. log-space 所有操作以避免溢出
3. segsum 稳定的实现是主要复杂度所在
4. 对于 codebase 当前规模，Mamba-2 是中期目标而非第一步

---

## 8. 优化器 — 超越 Adam

### 8.1 Muon（MomentUm Orthogonalized by Newton-Schulz）

**提出：** Keller Jordan, 2024（NanoGPT modded 博客）; 被 Moonshot AI 扩展到前沿 LLM

**核心算法：**
```
1. 动量: M_t = β·M_{t-1} + G_t
2. Newton-Schulz 正交化（5 次迭代，五阶多项式）:
   X = G / (‖G‖ + 1e-7)
   for _ in range(5):
       A = X @ Xᵀ
       B = b·A + c·(A@A)     // a=-3.4445, b=-4.7750, c=2.0315
       X = a·X + B@X
3. 参数更新: θ = θ - lr·U + weight_decay
```

**特点：**
- 仅适用于 2D 矩阵参数（attention W, MLP W, Conv W）
- embedding/bias/norm 参数仍用 AdamW
- 动量存储 1 份（vs AdamW 的 2 份）：**33% 优化器内存减少**
- Newton-Schulz 额外 ~1% FLOPs
- Moonshot 报告达到相同 loss 仅需 AdamW **48-52%** 的 compute

### 8.2 Sophia（Second-order Clipped Stochastic Optimization）

**论文：** Liu et al., 2024 (arXiv:2305.14342v4) — Stanford

**核心：** 廉价的对角 Hessian 估计（每 k=10 步更新一次，开销 <5%）

```
h_t = β₂·h_{t-1} + (1-β₂)·diag(Ĥ_t)    // EMA 的对角 Hessian
θ_{t+1} = θ_t - η_t·clip(m_t / max{γ·h_t, ε}, 1)
```

```
2× 加速 LLM 预训练（vs AdamW）
gap 随模型增大而增加（Sophia@540M ≈ AdamW@770M）
```

### C 语言实现策略

1. 当前 codebase 已有 SGD + Adam，添加 Muon 是最高 ROI 的优化器增强
2. Muon 的 Newton-Schulz 在 C 中代价较高（需要 SVD 替代或至少 SGEMM 调用）
3. Muon 对 small batch 不友好，需评估 demo 的 batch 大小
4. 如果内存受限，Muon 的优势（1x 动量 vs 2x m+v）会更明显

---

## 9. 激活函数 — SwiGLU 与替代品

### SwiGLU — 工业标准

**论文：** Shazeer, 2020 "GLU Variants Improve Transformer"

```
SwiGLU(x) = (Swish(x·W₁) ⊙ (x·V)) · W₂    // Swish(x) = x·σ(x)
```

**使用：** Llama 2/3、PaLM、Mistral、DeepSeek-V3、Qwen2、Gemini

**注意事项：**
- 需要 3 个权重矩阵（W₁, V, W₂），但用 2/3 规则将隐藏维减少到 `int(2*d_ff/3)` 后总参数量不变
- 无 bias（Llama/Mistral 风格）
- Kaiming 初始化 gate projection，Xavier 初始化 output projection

### 替代方案

| 函数 | 公式 | 适用场景 |
|---|---|---|
| ReLU² | `(max(0,x))²` | 边缘部署，90-95% 稀疏，Nemotron-4 340B 使用 |
| GELU | 标准 | 基准，ViT 兼容 |

### C 语言实现要点

1. 在 MLP 和 Transformer FFN 中添加 SwiGLU option
2. 实现 `swish(x) = x * sigmoid(x)`（其中 `sigmoid` 已有）
3. SwiGLU 层结构：`dense(W₁) → swish → gate(V) → multiply → dense(W₂)`
4. 注意 2/3 规则调整隐藏层维度

---

## 10. Embedding — RoPE 位置编码

**论文：** Su et al., 2023 — 旋转位置编码

### 核心原理

```
RoPE(x, pos) = x ⊙ cos(pos·θ) + rotate(x) ⊙ sin(pos·θ)
```

其中 `θ_i = base^{ -2i/d }`（base 通常 = 10000），频率递减。

**已采用：** Llama 3、Gemma、Mistral、Qwen、所有主流 LLM

**关键优势：**
- 相对位置编码（注意力得分仅依赖相对距离）
- **长序列外推**：通过调整 θ base 值（如从 10000→500000）支持更长上下文
- 可直接乘进 Q/K 中实现

### C 语言实现要点

1. 预计算 `cos(m·θ)` 和 `sin(m·θ)` 向量
2. 对奇数维和偶数维分别处理旋转
3. RoPE 与注意力模块耦合：在 `QKᵀ` 之前注入位置信息

---

## 11. MCU 推理部署关键词

推理二进制 (`src/infer/`) 为纯 C11、零外部依赖，可编译到 MCU。以下是特别适合 MCU 推理端的技术：

| 技术 | MCU 收益 | 说明 |
|---|---|---|
| **minLSTM/minGRU** | 参数为 LSTM 的 15%，训练快 1000x（但训练在桌面端） | 推理极简：仅 element-wise + parallel scan |
| **ReLU²** | 90-95% 激活稀疏 | 稀疏推理跳过大量计算 |
| **RMSNorm** | 比 LayerNorm 少一次均值和一次累加 | 推理更快 |
| **INT8 权重量化 (W8A8)** | 4× 模型压缩，~600KB→150KB | 放得进更多 MCU flash |
| **GQA (g=2)** | 极度压缩 KV cache | 仅当 Transformer 推理时相关 |
| **SwiGLU (2/3 rule)** | 总参数不变，不额外占用 flash | 训练端收益不增加推理成本 |

**注意：** 训练始终在桌面/服务器端进行。MCU 只运行 profiler 生成的推理二进制。

---

---

## 12. 权重量化 — INT8/INT4/FP8

### 格式对比

| 格式 | 压缩比 | 精度损失 | 最佳场景 |
|---|---|---|---|
| **INT8 (W8A8)** | ~4× | 1-3% | 通用兼容，平衡速度/精度 |
| **INT4 (W4A16)** | ~8× (仅权重) | <1-3% perplexity | LLM 部署，消费级 GPU |
| **FP8 (W8A8-FP)** | ~4× | **无损** | H100/B200 高端 GPU |
| **GPTQ (2/3/4/8 bit)** | 最大 16× | <1% perplexity (4-bit) | 极度压缩 |

### 2025-2026 最佳实践

1. **逐通道量化已成标准** — 每个输出通道独立 scale，精度显著优于逐张量，开销忽略不计
2. **INT4 需要保持激活精度** — 纯 W4A4 精度崩坏严重，不推荐。W4A8 是甜点，W4A16 最广泛支持
3. **Hadamard 旋转已成标配** — QuaRot, SpinQuant, DartQuant 等方法用正交旋转抑制激活离群值
4. **选择性/混合精度** — attention output projection + LayerNorm 保持高精度；大 FFN 层激进 INT4
5. **校准集质量** — GPTQ 需要 200-2000 条代表性 token；50 样本显著优于 10 样本

### 2025-2026 新兴趋势

| 趋势 | 说明 |
|---|---|
| **QServe W4A8KV4** | KV cache 量化到 4-bit + W4A8 权重 → 1.5× 注意力加速 |
| **NVFP4 / MXFP4** | 4-bit 浮点挑战 INT4 地位；B200 GPU 上 ~15% 更高吞吐 |
| **DartQuant** | 47× 更便宜的旋转校准；单 RTX 3090 上 3 小时量化 70B 模型 |
| **KV Cache 量化** | KVLinC, ThinKV, Expected Attention → learned corrections 压缩 KV cache 到 4-bit |

---

## 13. Mixture of Experts (MoE) 轻量化路由

### 2025-2026 前沿路由方法

| 方法 | 论文 | 关键创新 |
|---|---|---|
| **PreMoE** | May 2025 | 免训练：Predicted Expert Utility (PEU) 从 router logit 估算专家重要性，达 50% 稀疏无性能损失 |
| **DirMoE** | Feb 2025 | 解耦路由：Bernoulli (专家选择) + Dirichlet (权重分配)，Gumbel-Sigmoid 全可微，显式稀疏度旋钮 |
| **AME-TS** | May 2026 | 轻量 regime predictor 估算时序描述符 → soft structural prior 引导 token 路由 |
| **AIR-MoE** | May 2026 | 两级倒排索引：VQ 粗筛 + 精细评分（粒度 expert 场景路由成本大幅降低）|
| **FEP-MoE** | May 2026 | 生物启发：LIF 神经元 + 预测路由 → domain transition 正确 expert 概率 0.006→0.748 (124×) |
| **SSR** (Sinkhorn) | 2025 | 最优传输：轻量 Sinkhorn 路由替代辅助 loss，平衡 expert 利用率 |
| **MaxScore** | ACL 2025 | 最大流问题：SoftTopk 算子，消除 capacity-constrained routing 中的 token 丢弃 |

### 核心趋势

简单 Top-k+Softmax 路由正在被更原则化的方法替代：最优传输、变分推断、向量量化、甚至神经科学（脉冲神经网络）引入的思路。

---

## 14. 知识蒸馏 (Knowledge Distillation)

### 2025-2026 关键方向

| 方向 | 代表性工作 | 成果 |
|---|---|---|
| **多教师蒸馏** | TinyLLM (WSDM 2025) | 多教师 CoT 蒸馏，学生仅 1.1% 教师大小，+23.4% 提升 |
| **双认知蒸馏** | DualMind (Mar 2026) | 3 个 30B 教师 → 1.7B 学生，拓扑感知 TKD，涌现双认知推理 |
| **自适应蒸馏** | EA-KD (ICCV 2025) | 熵自适应样本加权，普适即插即用 |
| **动态温度** | DTS (IEEE 2026) | 基于 cross-entropy gap 动态调整温度，先软后硬 |
| **极端压缩** | LIFT and PLACE (CVPR 2026) | 1.6% 教师参数 → FID 15.73 |
| **预训练效率** | Low-Rank Clone (Dec 2025) | 1.7B 学生仅用 20B tokens 匹配 Qwen3 (1000× 训练效率) |
| **混合蒸馏** | MoD (IJCAI 2025) | 从 CLIP/SAM/Grounding DINO 蒸馏，72.48% ImageNet 无标注 |

### 趋势：从 Hinton 的 logit matching 走向多源、结构化、推理保留的蒸馏

---

## 15. 对比学习与自监督表示学习 (SSL)

### 2025-2026 突破

| 工作 | 创新 | 关键结果 |
|---|---|---|
| **CLAMP** (2025) | 物理启发：manifold packing → 排斥势函数动态定位类子流形 | 竞争 SOTA |
| **IConE** (Mar 2026) | 批次独立防崩缩：可学习辅助嵌入表 → batch size=1 也能稳定训练 | 适合高维科学数据 |
| **LEASE** (CVPR 2026) | 生成+判别统一：配对 codebook + token 级对比 loss，无在线 tokenizer | ImageNet-1K 统一 SOTA |
| **BOD-VCL** (2026) | Koopman 算子解耦静态/动态语义，防止视频 SSL 的虚假相关学习 | IJCV |
| **VACE** (May 2026) | 时序异常检测：速度一致性目标 + Mahalanobis 定位，无负样本 | TSB-AD-M SOTA |

### 趋势：突破批次依赖、物理理论化、统一生成与判别、领域特化

---

## 16. 数据增强 (Data Augmentation)

### 核心技巧（2025-2026 共识配方）

1. RandomResizedCrop — 位置不变性
2. RandAugment — 光度/几何多样性
3. Mixup + CutMix — 同时或交替
4. Random Erasing — 遮挡鲁棒性
5. Label Smoothing (0.1) — 防过自信

### 新兴技巧

| 技巧 | 说明 |
|---|---|
| **TdAttenMix** | 注意力引导 Mixup：用标签作为"任务"引导 attention 选择语义一致的 patch |
| **SMMix** | ViT 自注意力图指导 Mixup 的混合比例 |
| **CutRot** | 随机 patch 旋转 90°/180°/270°（不删除信息，特别适合小样本学习） |
| **CutCov** | 同图内交换两个随机 patch（不破坏判别信息） |

### 重要发现（2026）

Muon 优化器与强数据增强（Mixup+CutMix）有**共生效应**：增强使梯度矩阵奇异值谱更丰富，Muon 的 polar-factor 更新能利用这些方向，而 AdamW 集中在少数主导模式。

---

## 17. 神经架构搜索 (NAS) — 边缘 MCU

### 2025-2026 关键项目

| 项目 | 方法 | 亮点 |
|---|---|---|
| **PrototypeNAS** (Mar 2026) | 零样本 NAS：ensemble zero-shot proxies + Hypervolume 子集选择 | 数分钟找到 MCU 可用模型，12 数据集验证 |
| **Shapley-DARTS** (Mar 2026) | Shapley value 公平评估操作贡献，剪枝低贡献操作 | CIFAR-10: 2.7M params, 95.45% acc, 0.26 GPU-day |
| **μNAS** | Aging Evolution + 结构化剪枝 + KD，显式建模 3 瓶颈（size/latency/memory） | MCU 优化最成熟的开源代码 |
| **MicroNAS** (2025) | Differentiable NAS + lookup-table latency estimation，Time-Reduce/Sensor-Fusion cell | STM32/Arduino 实时时序学习 |
| **SpikeNAS** (2025) | SNN 专用 NAS + memory budget awareness | 29-117× faster search |
| **HW-NAS for MCU** (2025) | 笔记本 CPU 可运行的 HW-aware NAS（无需 GPU） | 3.5 小时完成搜索 |
| **COLE** (May 2026) | 用 LM embedding 编码架构（PyTorch class 文本）→ NAS 性能预测 | 34% 评估预算减少 |

---

## 18. Test-Time Training (TTT) — 推理时学习

### 核心突破

| 工作 | 创新 | 关键数据 |
|---|---|---|
| **TTT-E2E** (Dec 2025) | Stanford/NVIDIA/Berkeley。双循环架构，仅更新最后 25% MLP 层 | 2M 上下文：35× 快于 full attention |
| **In-Place TTT** (ICLR 2026 Oral) | 将 MLP 最终投影矩阵当作 fast weight，无缝为现有 LLM 赋 TTT | 4B 模型，128K 上下文 |
| **LaCT** (ICLR 2026 Poster) | 用极大块更新（2K-1M token）代替传统小 batch | GPU 利用率从 <5% 提升到接近 100% |
| **BETA** (Apr 2026) | 黑盒 API 场景：小型 steering model 本地创建梯度通路，零额外 API 调用 | +7.1% ImageNet-C acc，250× 低成本 |
| **PonderTTT** (Dec 2025) | 免训练门控：用 TTT 的自监督 reconstruction loss 作为门控信号 | 82-89% Oracle Recovery |

### 趋势：从"训练-部署"到"推理时动态权重适应"

---

## 19. 持续学习 (Continual Learning) — 边缘端

### 2025-2026 前沿

| 方向 | 代表工作 | 指标 |
|---|---|---|
| **SNN 脉冲网络** | CLP-SNN on Loihi 2 | 113× 低延迟, 6600× 低能耗 vs GPU |
| **睡眠增强重放** | SESLR | 1-bit 特征存储 → 32× 内存节省，30% acc 提升 |
| **免重放方法** | AGMP | Astrocyte-gated 多时间尺度可塑性 |
| **超维计算** | ImageHD (FPGA) | 40.4× speedup, 383× energy efficiency vs GPU |
| **知识转换** | KT (Dec 2025) | KD + Active Learning + 因果推理 → edge 标签不足场景 |

### 趋势：SNN （脉冲网络）是边缘持续学习的明确领先者；sleep-wake 的生物启发机制有效防遗忘；超维计算 (HDC) 作为轻量替代方案兴起

---

## 20. 片上训练 (On-Device Training)

### 2025-2026 核心方法

| 方法 | 内存预算 | 策略 |
|---|---|---|
| **MeZO/ZO (BP-free)** | ~推理级内存 | 零阶梯度估计（前向 pass 近似梯度），收敛慢但最终等价。**能训 256KB SRAM 上完整模型** |
| **Forward-Forward/PEPITA** | 极低 | 消除反向传播，ground-truth class 注入前向网络 |
| **TinyProp** | 20-98% 减少 | 自适应稀疏 BP，动态调整每步反向传播比例 |
| **Dynamic Sparse Update** | 98% feature memory 减少 | 仅更新重要 channel/layer 梯度 |
| **TinyMP** (DAC 2025) | 80.8% memory 节省 | Gradient Condensing + Alternant Partial Update |
| **BioTrain** (Apr 2026) | <1MB, <50mW | 完整 BP fine-tuning on GAP9 MCU，17 samples/s |
| **TrainDeeploy** (DATE 2026) | RISC-V heterogeneous SoC | Transformer fine-tuning + LoRA: 15× 参数减少，11 images/s |

### 核心发现

激活存储（而非可训练参数数量）是片上训练的真正瓶颈。MCUNet-in1 的完整 BP 需要 ~7.4MB 仅用于激活，远超常见 MCU 的 256KB-1MB SRAM。Zeroth-order 方法是目前 **唯一能在 256KB MCU 上完整训练模型**的方法，无需辅助内存。

> **本项目定位：** 训练始终在桌面完成，MCU 只运行推理二进制。本章为文献综述，不纳入路线图。

---

## 21. 激活稀疏性 (Activation Sparsity)

### 2025-2026 突破

| 方法 | 创新 | 成果 |
|---|---|---|
| **WAS** (EMNLP 2025) | 权重-激活耦合 + 贝叶斯优化分配逐块稀疏比 | 60% 稀疏度 → 1.68× 推理加速，免训练 |
| **SPON** | 生物启发：注入输入无关的激活向量作为"锚点"，吸收进 bias → 零推理开销 | 1.178% avg 提升 |
| **DuoGPT** (NeurIPS 2025) | 权重 + 激活双稀疏 → unified spMspV workload | 比结构化剪枝高 9.17% acc (iso-speedup) |
| **2:4 + v:n:m Sparsity** (Meta, Feb 2026) | 硬件加速 2:4 权重 + v:n:m 激活稀疏 → **预训练阶段**应用 | 1.4-1.7× 端到端训练加速 |
| **SURGEON** (CVPR 2025) | 逐层动态激活剪枝比（梯度重要性 + 激活内存效率） | CNN + Transformer SOTA |

### 趋势：从静态→动态稀疏分配；从推理→训练加速；从单一→双稀疏（权重+激活复合）

---

## 22. 联邦学习 (Federated Learning) — 边缘

### 2025-2026 轻量方案

| 框架 | 通信减少 | 内存减少 | 关键机制 |
|---|---|---|---|
| **FTTE** | 69% | 80% | 内存感知参数选择 + 稀疏半异步聚合 |
| **RELIEF** (Apr 2026) | — | 弹性 | 模态分解 LoRA + divergence-guided elastic training |
| **EdgeFD** | 数量级 (logits) | 低计算 | 联邦蒸馏 + KMeans-DRE 客户端过滤 |
| **FedCDC** (May 2026) | 高效 | — | Daisy-chain 簇内训练 + proximal 簇间对齐 |
| **HFEL** (2026) | 36.2% | — | 多层级聚合 + 轻量差分隐私 |
| **FEMBNet** (2025) | 13.9k 参数减少 | — | FL 原生轻量 attention (FEMA module) |

### 趋势：稀疏感知训练仅传必要内容；联邦蒸馏替代梯度交换；分簇架构减少跨设备通信；LoRA/PEFT 适应 FL 场景

---

## 23. 深度可分离卷积与轻量 CNN

### MobileNet 演进

| 版本 | 核心创新 | 关键差异 |
|---|---|---|
| **MobileNetV1** (2017) | Depthwise Separable Conv：depthwise + pointwise | 8-9× 计算减少 |
| **MobileNetV2** (2018) | Inverted Residual + Linear Bottleneck | 更高效的特征复用 |
| **MobileNetV3** (2019) | NAS + NetAdapt + h-swish | 硬件感知搜索 |
| **MobileNetV4** (2024) | Universal Inverted Bottleneck + Mobile MQA | 移动生态系统统一模型 |

### 2025-2026 趋势

- **RepVGG 风格**：训练时多分支，推理时融合为单路（3×3 Conv + 1×1 Conv + identity → 单个 3×3 Conv）
- **深度可分离 + Attention 融合**：CNN 中引入轻量通道注意力（如 SENet、ECA-Net）
- **动态卷积**：CondConv、DY-Conv — 输入依赖的卷积核生成
- **边缘专用加速器**：CNN 加速器 + 固定功能单元（如 im2col 优化）

### 当前 codebase 已有

你的 CNN 实现已支持 `CNN_CONV_DEPTHWISE` 模式 — 这是 MobileNet 风格深度可分离卷积的核心操作。可以考虑进一步添加 RepVGG 风格的训练-推理分离和通道注意力。

---

*基于 2025-2026 年最新论文调查（ICLR/ICML/NeurIPS/EMNLP/CVPR 2025-2026 及 arXiv 预印本），覆盖 12 个扩展领域*
