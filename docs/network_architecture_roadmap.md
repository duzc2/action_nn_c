# 网络结构扩展路线图

> 最后更新：2026-05-28 | 基于 2025–2026 论文调查 + 项目应用场景分析
>
> **Phase 0 状态：已完成** — RMSNorm、LAuReL-RW、Dropout 三个共享模块已集成到全部 6 种网络类型（MLP、CNN、CNN Dual Pool、RNN、GNN、Transformer），含 70 项 P0 集成测试，全部通过。

---

## 项目场景定位

| 维度 | 特征 |
|---|---|
| **训练端硬件** | 桌面 CPU（x86-64）或服务器 — 承担 profiler 代码生成 + 完整 FP32 训练 |
| **推理端硬件** | 桌面 CPU / Wasm 浏览器 / 边缘设备 / **MCU** — 推理二进制为纯 C11、零外部依赖，可部署到任意支持 C 编译链的平台 |
| **模型规模** | 无架构层面的硬限制。当前 demo 典型范围 82 参数 ~ 600K 参数，但系统可支持任意规模（参考：snake 的 387 字节可直接放入 MCU flash） |
| **核心范式** | "大模型做策略，小网络做执行" — 教师蒸馏到小型 C 可执行网络，训练在桌面完成，推理可任意部署 |
| **主要领域** | 游戏 AI 智能体控制、边缘视觉、导航/路径规划、时序预测 |
| **核心架构** | 代码生成管线（profiler → train → infer），三者独立编译，推理端不携带训练代码 |
| **训练重要性** | 与推理同等重要，必须支持从零训练和微调 |
| **代码约束** | 纯 C11、无外部依赖、可分离编译、CMake 开关控制 |

---

## 当前已实现（6 种类型）

| 类型 | 注册名 | 能力 |
|---|---|---|
| MLP | `mlp` | 全连接，可变层数，6 种激活函数，SGD/Adam，**RMSNorm + Dropout + Skip (ID/LAuReL-RW)** |
| CNN | `cnn` | 标准/深度可分离卷积，4 种池化，BN，Dropout，**RMSNorm + Skip (ID)** |
| CNN Dual Pool | `cnn_dual_pool` | CNN 变体，avg+max 双池化，**RMSNorm + Skip** |
| RNN | `rnn` | Elman 式单层 RNN，tanh 隐藏激活，**RMSNorm + Dropout + Skip (ID/LAuReL-RW)** |
| GNN | `gnn` | 消息传递图网络，mean 聚合，图池化/锚点读出，**RMSNorm + Dropout + Skip (ID/LAuReL-RW)** |
| Transformer | `transformer` | 单头自注意力，字符级 tokenizer，文本 QA，**RMSNorm + Dropout + Skip (ID/LAuReL-RW)** |

---

## 优化后的开发 TODO（按阶段）

---

### Phase 0 — 基础设施（零依赖，三项可并行开发）✅ 已完成

这三项已完成。RMSNorm (P0-1)、LAuReL-RW (P0-2)、Dropout (P0-3) 均已实现并集成到所有 6 种网络类型中。每个模块都有独立的 C 实现、前向/反向传播、以及 70 项跨网络类型的集成测试。

| 模块 | 实现文件 | 测试覆盖 |
|---|---|---|
| RMSNorm | `src/nn/norm/rms_norm.h/.c` | MLP(6), CNN(6), RNN(5), GNN(5), Transformer(5) |
| LAuReL-RW Skip | `src/nn/residual/skip_connection.h/.c` | MLP(4), CNN(2), RNN(3), GNN(2), Transformer(4) |
| Dropout/DropPath | `src/nn/dropout/dropout.h/.c` | MLP(3), CNN(0*), RNN(3), GNN(2), Transformer(3) |

> *CNN dropout 已有内置实现，此次改为调用共享 dropout API

测试文件：
- `verify/test_nn_p0_integration.c` — MLP + CNN 集成 (26 tests)
- `verify/test_rnn_p0.c` — RNN 集成 (14 tests)
- `verify/test_gnn_p0.c` — GNN 集成 (14 tests)
- `verify/test_transformer_p0.c` — Transformer 集成 (16 tests)
- `verify/test_cnn_full.c` — CNN 已有测试 (含 6 P0 tests, 30 total)

这三个模块互相独立、零前置依赖，每一个完成后所有 6 种现有网络类型立即受益。可以同时开工，不需要等任何一项完成。

---

#### [P0-1] RMSNorm — 归一化层共享化

**为什么现在需要：**
- 当前 BN 仅耦合在 CNN 内部，其它网络类型无可用的归一化
- Transformer、RNN、MLP 深层训练时极易梯度爆炸（Peri-LN 论文证明 Pre-LN 导致指数方差增长）
- 2026 年研究证明 LayerNorm 的均值减法冗余，RMSNorm 更简洁高效

**做什么：**
```
新增 src/nn/norm/rms_norm.h + rms_norm.c
  - RMSNorm_forward(x, n, gamma, epsilon) → 仅需 (sum of squares + sqrt + 标量乘)
  - RMSNorm_backward(dout, x, gamma) → 反向传播

各网络类型增加 norm 配置项:
  - MLP:    每层 dense 后可选 RMSNorm
  - CNN:    现有 BN 保留，增加 RMSNorm 可选替代
  - RNN:    输入→隐藏 投影后加 RMSNorm
  - GNN:    消息聚合后加 RMSNorm
  - Transformer: Attention 和 FFN 各子层输出后加 RMSNorm (Peri-LN 模式)
```

**工作量：** 小（RMSNorm 算法极简，核心 ~30 行 C）
**受益 demo：** 所有 demo，尤其是 transformer、cnn_rnn_react、hybrid_route 的训练稳定性

---

#### [P0-2] 残差连接 — LAuReL-RW

**为什么现在需要：**
- 所有网络均为纯前馈。nested_nav、edge_video_preprocess 等 demo 的网络深度受限
- LAuReL-RW 仅需 +2 参数/层，获得接近多一层的收益，ROI 极高
- snake (82 params) 这种极小型网络加残差后可以有显著的训练质量提升

**做什么：**
```
各层增加 skip_mode 枚举:
  NONE     = y = F(x)
  IDENTITY = y = F(x) + x                    // 标准 ResNet 残差
  LAUREL_RW = y = α · F(x) + β · x         // 可学习残差
    where α = sigmoid(α_raw), β = 1 - α    // softmax 归一化

实现范围:
  - MLP dense 层:    skip over 一层
  - CNN conv 层:     skip over 一层 (shape 匹配时)
  - Transformer:     skip over attention sublayer + skip over FFN sublayer
  - RNN:             skip over 时间步 (SCORE 式共享块迭代)
  - GNN:             消息传递轮次间 skip
```

**工作量：** 中（LAuReL-RW 仅 2 标量参数，但需要各类型 forward/backward 适配）
**受益 demo：** nested_nav, edge_video_preprocess, cnn_rnn_react, hybrid_route

---

#### [P0-3] Dropout + Stochastic Depth — 正则化共享化

**为什么现在需要：**
- 当前 Dropout 仅在 CNN 内部 ad hoc 实现，MLP/Transformer 训练时容易过拟合
- snake (82 params) 和 sevenseg 这类小模型在训练样本不足时需要正则化
- Stochastic Depth 在深层网络中比逐神经元 Dropout 更有效

**做什么：**
```
新增 src/nn/dropout/dropout.h + dropout.c
  - dropout_forward(x, n, keep_prob, rng_state)    // 标准 Dropout
  - droppath_forward(x, n, keep_prob, rng_state)   // Stochastic Depth (整层随机跳过)

提供统一 training/eval 模式接口:
  - training 模式: mask 随机，除法缩放
  - eval 模式: 直接通过

各网络类型接入:
  - MLP dense 层后:    可选 Dropout
  - CNN conv 层后:      已有实现，改为调 shared 模块
  - Transformer FFN 后: 可选 Dropout
  - 所有网络的 skip 路径: 可选 DropPath
```

**工作量：** 小（核心算法简单，主要是抽象和接入）
**受益 demo：** mnist, sevenseg, snake, transformer（小样本训练场景）

---

### Phase 1 — 网络能力升级（直接增强现有类型）

Phase 0 完成后开始。SwiGLU 几乎无依赖可最早完工；Muon 独立于其它项；minGRU 和 MHA+GQA 需要 Phase 0 的 RMSNorm + 残差作为配套。Phase 1 内部也可并行。

---

#### [P1-1] minGRU — 替换 Elman RNN

**为什么现在需要：**
- 现有 Elman RNN 无法处理长序列依赖，cnn_rnn_react 和 cs 的行为记忆能力弱
- minGRU 比 LSTM 更适合：参数少 62-85%，用 parallel scan 训练（不依赖 BPTT），C 实现极简
- 与传统 LSTM 不同，minGRU 的 gate 不含 h_{t-1} → 天然并行化 → training 大幅加速

**做什么：**
```
新增 src/nn/types/rnn/rnn_min_gru.h + rnn_min_gru.c  (或新的 minrnn/ 类型)

minGRU 核心 (每时间步):
  z_t   = sigmoid(W_z @ x_t + b_z)       // update gate (仅依赖 x_t!)
  h̃_t   = W_h @ x_t + b_h                // candidate (无 tanh!)
  h_t   = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  // 与标准 GRU 相同的循环

训练: parallel scan (O(log T) 步, log-space 稳定实现):
  a_t = 1 - z_t,   b_t = z_t ⊙ h̃_t
  v_t = a_t ⊙ v_{t-1} + b_t    ← prefix scan with (A₂·A₁, A₂·B₁+B₂)

向后兼容: 保留现有 Elman RNN, minGRU 作为 RNN 的 activation_type 扩展或独立注册类型
```

**工作量：** 中（forward 极简，parallel scan 需要仔细实现 log-space 数值稳定）
**受益 demo：** cnn_rnn_react (行为记忆), cs (视觉导航状态追踪), transformer (序列编码器)

---

#### [P1-2] 多头注意力 + GQA

**为什么现在需要：**
- 当前 Transformer 仅单头注意力，hybrid_route 的 cue 理解能力受限
- GQA 已成为工业标准（Llama/Gemma/Mistral），head 分组共享 KV 可减少 KV cache
- KHA 跨头交互是可选的后期增强（对角初始化+零推理开销）

**做什么：**
```
现有 Transformer 增加配置:
  num_heads:    1..N (默认 1，向后兼容)
  num_kv_heads: 1..num_heads (默认 = num_heads, 即 MHA; <num_heads 时即 GQA)
  head_dim:     d_model / num_heads

新增独立注意力模块:
  src/nn/attention/scaled_dot_product.h + .c
  - mha_forward(Q, K, V, num_heads, head_dim) → multi-head output
  - gqa_forward(Q, K, V, num_heads, num_kv_heads, head_dim) → grouped-query output
  - attention_backward(...) → dQ, dK, dV

接入范围:
  - Transformer self-attention:   直接替换
  - Transformer cross-attention:  为 Seq2Seq 准备
  - GNN Edge-Conditioned GATv2:   复用注意力计算
```

**工作量：** 中（heads 分块、拼接、投影的 C 实现，GQA 的 KV 共享逻辑）
**受益 demo：** hybrid_route, transformer

---

#### [P1-3] SwiGLU 激活 — MLP/Transformer FFN 升级

**为什么现在需要：**
- 现有 MLP 只有 ReLU/Sigmoid/Tanh/LeakyReLU 等逐点激活
- SwiGLU 的门控机制表达力更强，无需增加总参数量（通过 2/3 规则压缩隐藏维）
- 是 Llama/Mistral/DeepSeek 的标配，早已充分验证

**做什么：**
```
新增激活函数:
  swish(x) = x · sigmoid(x)

新增 MLP 层选项:
  FFN_SWIGLU:
    gate  = swish(W_gate @ x)        // Swish 门控
    value = V @ x                    // 值投影 (无激活)
    hidden = gate ⊙ value           // 逐元素乘
    output = W_out @ hidden
  // hidden_dim = int(2/3 * original_hidden_dim)   // 2/3 规则

接入范围:
  - MLP dense layer:     新增 SwiGLU 作为 activation_type
  - Transformer FFN:     新增 ffn_swiglu 作为 FFN type
```

**工作量：** 小（已有 sigmoid 实现，核心仅 swish + 门控乘法 + 2/3 比例调整）
**受益 demo：** 所有使用 MLP 的 demo（snake, nested_nav, weather, move, target, sevenseg, hybrid_route, mnist）

---

#### [P1-4] 优化器升级 — Muon

**为什么现在需要：**
- 当前仅 SGD + Adam。Muon 用 1 份动量存储替代 Adam 的 m+v（2 份），**33% 优化器内存节省**
- 同一 loss 仅需 Adam 52% 的迭代步数（Moonshot AI 数据）
- 牛顿-舒尔茨迭代对小型矩阵（本项目典型规模 ~几十到几百维）计算代价极低
- 矩阵参数占本项目模型的绝大部分（MLP dense weight、CNN kernel）

**做什么：**
```
新增 src/train/optimizer_muon.h + optimizer_muon.c

Muon 核心:
  1. 动量:    M = β·M + G            // 标准 SGD 动量 (β=0.95)
  2. NS 正交化: X = NewtonSchulz(M, 5) // 5 次五阶多项式迭代
  3. 更新:    θ = θ - lr·X - lr·wd·θ  // weight decay

适用范围:
  - 2D 矩阵参数:  Muon (MLP dense, CNN conv kernel, Transformer Q/K/V/O, GNN edge)
  - 1D 参数:      AdamW (bias, norm gamma/beta, embedding)

Newton-Schulz 在 C 中的实现:
  矩阵乘积 X@X^T → 多项式迭代 → 最多 5 次，在几十维矩阵上极快
  注意: 若 rows > cols 需转置后再迭代（transpose trick）
```

**工作量：** 中（NS 迭代 + 矩阵乘的实现，需要判别矩阵/向量参数类型）
**受益 demo：** 所有 demo 的训练阶段

---

### Phase 2 — 领域特化增强

依赖 Phase 1 的部分模块。其中 P2-4 (INT8 量化) 和 P2-5 (超参数搜索) 与 Phase 0/1 完全解耦，可以提前启动。

---

#### [P2-1] GNN 边特征 + Edge-Conditioned GATv2

**为什么现在需要：**
- road_graph_nav 仅用邻接关系，road 的权重/距离等边特征完全忽略
- 边特征直接提升路径规划质量
- Edge-Conditioned GATv2 是 2025 年最新标准

**做什么：**
```
GnnConfig 扩展:
  edge_dim:       0 (默认，向后兼容) 或 >0 (启用边特征)
  aggregator_type: 新增 ATTENTION (复用 attention 模块)

Edge-Conditioned GATv2 (当 edge_dim > 0 且 aggregator_type == ATTENTION):
  ① 投影: ĥᵢ = W·hᵢ,  ĥⱼ = W·hⱼ,  êᵢⱼ = We·eᵢⱼ
  ② 注意力: αᵢⱼ = softmax_j(LeakyReLU(aᵀ·[ĥᵢ∥ĥⱼ∥êᵢⱼ]))
  ③ 聚合: hᵢ' = Σ αᵢⱼ·ĥⱼ + hᵢ

新增聚合器:
  GNN_AGG_SUM:  hᵢ' = Σ_{j∈N(i)} W·hⱼ
  GNN_AGG_MAX:  hᵢ' = max_{j∈N(i)} W·hⱼ
  GNN_AGG_ATTN: 当 edge_dim=0 → GAT; 当 edge_dim>0 → EC-GATv2
```

**工作量：** 中（边特征投影矩阵 + 三方注意力 + 向后兼容）
**受益 demo：** road_graph_nav

---

#### [P2-2] 数据增强框架

**为什么现在需要：**
- 训练与推理的独立编译管道使数据增强天然适合放在训练端
- edge_video_preprocess、mnist、mnist_cnn 等视觉 demo 的训练数据有限
- Mixup + CutMix 是 ViT/CNN 训练的标准配方，Muon 优化器与强增强有共生效应

**做什么：**
```
新增 src/train/augment.h + augment.c

图像级增强 (CNN 输入):
  - random_crop_resize():    随机裁剪+缩放（位置不变性）
  - random_horizontal_flip(): 随机水平翻转
  - random_erasing():         随机矩形遮挡
  - cutmix(a, b, label_a, label_b):     从图 A 切一块贴到图 B，标签按面积混合
  - mixup(a, b, label_a, label_b, α):   两张图线性混合，α~Beta(0.2,0.2)

特征级增强 (MLP 输入):
  - gaussian_noise(x, σ):   高斯噪声注入
  - feature_dropout(x, p):  随机遮罩输入特征

标签增强:
  - label_smoothing(y_true, ε):  软化 one-hot 标签，ε=0.1
```

**工作量：** 小（每个增强函数独立，可选组合使用）
**受益 demo：** edge_video_preprocess, mnist_cnn, mnist（训练质量提升）

---

#### [P2-3] 知识蒸馏框架 — 系统化 teacher-student 模式

**为什么现在需要：**
- 项目核心范式就是 teacher-student（A*→nested_nav, BFS→road_graph_nav, heuristic→snake）
- 目前每个 demo 都手工实现蒸馏逻辑
- 系统化后可以减少 demo 重复代码，并提供更强大的蒸馏能力

**做什么：**
```
新增 src/train/distill.h + distill.c

蒸馏损失函数:
  - kd_mse_loss(student_logits, teacher_logits):           MSE 蒸馏
  - kd_kl_loss(student_logits, teacher_logits, temp):      KL 散度蒸馏
  - kd_cosine_loss(student_feat, teacher_feat):            特征层蒸馏
  - kd_combined_loss(student_logits, teacher_logits, y_true, α, temp):
      = α·CrossEntropy(logits, y_true) + (1-α)·KL(logits/T, teacher_logits/T)·T²

Profiler 侧支持:
  - 网络定义中声明 teacher_model 路径和蒸馏配置
  - 自动生成 teacher 前向传播 + student loss 计算代码
```

**工作量：** 中（损失函数实现简单，profiler 侧支持需要设计）
**受益 demo：** nested_nav, road_graph_nav, snake, cs（统一蒸馏实现）

---

#### [P2-4] INT8 后训练量化 — MCU 推理部署

**为什么现在需要：**
- 训练在桌面 CPU 完成，但推理二进制 (`src/infer/`) 设计为独立编译部署
- 推理端可能需要运行在 MCU 级别的硬件上（flash/内存受限）
- INT8 量化将模型缩小 4×，同时保持 1-3% 精度损失
- 与你的编译分离架构天然契合：训练用 FP32，部署用量化 INT8

**做什么：**
```
新增 src/infer/quantize.h + quantize.c (推理端模块, 不依赖训练)

支持格式:
  - W8A8 (INT8 权重 + INT8 激活):  4× 压缩, 通用兼容, 1-3% 精度损失
  - W8A32 (INT8 权重 + FP32 激活):  4× 权重压缩, ~0% 精度损失, 适合 MCU 浮点单元

量化流程:
  Step 1: 校准 → 遍历少量样本, 收集每层激活的 min/max
  Step 2: 量化权重 → weight_int8[i] = round(weight_fp32[i] / scale_w)
  Step 3: 推理 → dequant → compute → requantize (或直接用 FP32 accumulator)

逐通道量化 (Per-Channel):
  - 每个输出通道独立 scale (2025 工业标准, 精度显著优于逐张量)
  - 对 CNN depthwise conv 尤其重要

推理侧接入:
  - 每个网络类型 (MLP/CNN/RNN/GNN/Transformer) 增加 infer_quantized 入口
  - 或统一在 infer_runtime 层做 dispatch (检测权重是否为 INT8)
  - 编译开关: ACTION_C_ENABLE_INFER_QUANTIZE=ON/OFF

Profiler 侧:
  - 新增 quantize 步骤: generate → train → calibrate → quantize → build infer
  - 量化后的权重以 .c/.h 数组形式生成 (int8_t 类型)

不做的:
  - 不做量化感知训练 (QAT) — 当前 demo 规模下 W8A8 后训练量化已足够；模型更大时可重新评估 QAT
  - 不做 INT4 — 纯 INT4 精度损失严重, MCU 不是那么极端受限
```

**工作量：** 中（校准 + 逐层量化核心 ~200 行 C，各网络类型的量化推理适配工作量较大）
**受益 demo：** edge_video_preprocess (摄像头端部署), 所有 demo 的 MCU/Wasm 推理端

---

#### [P2-5] 轻量超参数搜索器 — 自动化网络调优

**为什么现在需要：**
- 当前每个 demo 手动试网络结构（"8→6→4 不好，改成 8→12→4 试试"）
- snake 的 82 参数 MLP 试 20 种配置仅需几分钟，但不自动化就是反复手工劳动
- 非完整 NAS — 不发明新算子，只在你已定义的搜索空间内找最佳参数组合

**做什么：**
```
新增 tools/hyper_search/ (Python 脚本, 不进入 C 源码)

搜索空间定义 (每个 demo 的 .yaml 或 .json 配置):
  snake:
    mlp:
      hidden_layers: [2, 3, 4]
      hidden_dim:    [4, 8, 12, 16, 24]
      activation:    ["relu", "swiglu", "leaky_relu"]
    train:
      learning_rate: [0.001, 0.003, 0.01, 0.03]
      optimizer:     ["adam", "muon"]

搜索算法: 贝叶斯优化 (推荐) 或 简单随机搜索
  - 贝叶斯优化: 20-50 次试验找到近最优配置
  - 高斯过程 surrogate model → 选下一个最有潜力的配置 → 调 profiler + train → 记录得分
  - 或用更简单的 Hyperband (early-stop 差配置, 节省算力)

工作流:
  hyper_search.py snake
    → 生成配置 → 调 profiler → 调 train → 跑 eval → 记录得分
    → 选下一个配置 → ... → 重复 30 次 → 输出最优配置

刻意不做的事:
  - 不做 DARTS (可微分, 太重)
  - 不做 RL NAS (搜索成本太高)
  - 不搜索网络类型 (MLP vs CNN — 这个由开发者指定)
  - 只在用户指定的网络类型内搜索超参数
```

**工作量：** 中（Python 脚本 ~500 行，依赖 scikit-learn 做高斯过程）
**受益 demo：** 所有 demo（自动化代替手工试错）

---

### Phase 3 — 新能力

---

#### [P3-1] Embedding Layer + RoPE

**为什么现在需要：**
- Transformer 当前使用字符级 tokenizer，限制词汇表达能力
- 在 hybrid_route 中需要对上游 cue 做更好的语义编码
- RoPE 是位置编码的工业标准，支持长序列外推

**做什么：**
```
新增 src/nn/embedding/embedding.h + embedding.c

Token Embedding:
  - embed_lookup(indices, vocab_size, emb_dim, weights) → embedded_sequence
  - 可学习查找表，嵌入维度 ≤ 隐藏维度

RoPE 位置编码:
  - rope_apply(x, position, base=10000) → x_rotated
  - x_rotated = x ⊙ cos(pos·θ) + rotate(x) ⊙ sin(pos·θ)
  - 预计算 cos/sin 表

Transformer 升级:
  - embedding_mode: CHARACTER (当前, 零参数) | EMBEDDING (可学习查找表 + RoPE)
```

**工作量：** 中（embedding lookup 简单，RoPE 旋转实现需要注意奇偶维配对）
**受益 demo：** transformer, hybrid_route

---

#### [P3-2] Seq2Seq Encoder-Decoder

**为什么现在需要：**
- 目前所有网络都是 "输入→输出" 单次映射
- weather 等多步时序预测、snake 等多轮决策可以从 encoder-decoder 结构获益
- 使用 minGRU 作为 encoder/decoder 单元，代码复用

**做什么：**
```
Profiler 侧支持:
  网络中声明 encoder 子网络和 decoder 子网络
  decoder 接收 encoder 的输出作为 cross-attention 的 K/V

Seq2Seq minGRU 架构:
  Encoder: minGRU 读取输入序列 → 输出 final hidden state
  Decoder: minGRU + cross-attention → 逐 token 生成输出序列

实现方式:
  不新增独立网络类型
  在 profiler 层将两个已注册的网络（encoder + decoder）组装为 Seq2Seq
  encoder 输出 → 通过 profiler 生成的连接代码传入 decoder
```

**工作量：** 大（需要 profiler 层支持子网络间数据流动，cross-attention 实现，decoder 的 autoregressive 循环）
**受益 demo：** weather（多步预测）, snake（多轮决策）, transformer（seq2seq 翻译）

---

### Phase 4 — 远期按需

---

#### [P4-1] MobileNet 风格轻量 CNN 配置

**为什么未来需要：**
- edge_video_preprocess 已有深度可分离卷积
- 需要系统化的轻量配置：Inverted Residual Block + 线性瓶颈 + 高效激活
- 当前按需做即可

---

#### [P4-2] U-Net — 像素级预测

**为什么未来需要：**
- 仅在 pixel-level 任务需要（运动分割、语义分割）
- 前置依赖残差连接和跳跃连接（P0-2 完成后才可能）
- 当日仅 edge_video_preprocess 可能需要

---

#### [P4-3] Mamba-2 / SSD — 长序列状态空间

**为什么远期可能需要：**
- 当前序列长度（time series ≤3653, game frames ≤100s）不需要线性注意力
- 当 cs 或 weather 扩展到极长序列时再考虑
- 实现复杂度高（segsum 稳定化是主要难点）

---

## 不纳入的项目及其原因

以下方向的调研很深入，但基于项目场景，明确**不纳入当前路线图**：

| 方向 | 排除原因 |
|---|---|
| **联邦学习** | 单设备代码生成工具，无分布式训练场景 |
| **完整 NAS / DARTS** | 搜索成本过高；P2-5 的轻量超参数搜索器已足够覆盖需求 |
| **MoE (混合专家)** | 当前应用场景以小型网络为主，MoE 的路由开销和通信成本远超小模型自身的计算量；若未来出现需要 MoE 的大模型场景再评估 |
| **TTT (推理时训练)** | 与"先生成后编译"的代码生成管道冲突，推理时不应修改权重 |
| **脉冲神经网络 (SNN)** | 需要特殊的脉冲编码和神经形态硬件，超出项目硬件范围 |
| **INT4 / 量化感知训练** | 当前场景 W8A8 INT8 即可覆盖典型需求；INT4 和 QAT 的额外精度收益不足以弥补复杂度增加（模型更大时可重新评估） |
| **片上训练 (MCU 级训练)** | 训练在桌面/服务器完成，MCU 只跑推理二进制；MCU 上的完整 BP 训练不在范围 |
| **持续学习** | 训练管道是"生成-训练-部署"一次性流程，不是在线学习 |
| **KV Cache 量化** | 当前 Transformer 上下文极短（字符级对话），无长序列 KV cache 压力 |
| **联邦蒸馏** | 无多设备场景 |
| **超维计算 (HDC)** | 与现有网络类型架构不兼容 |

---

## 完整依赖关系图

```
Phase 0 (三项零依赖, 可同时开工)
═══
P0-1 RMSNorm     ────独立模块 (新增 src/nn/norm/)
P0-2 LAuReL-RW   ────独立模块 (各类型 forward/backward 适配)
P0-3 Dropout     ────独立模块 (新增 src/nn/dropout/)

      │
      ▼
Phase 1 (依赖 Phase 0 基础设施)
═══
P1-1 minGRU ◄──────── 需要 P0-1 (RMSNorm) + P0-2 (skip)
P1-2 MHA+GQA ◄─────── 需要 P0-1 (RMSNorm) + P0-2 (skip)
P1-3 SwiGLU ───────── 依赖 sigmoid (已有), 几乎零依赖
P1-4 Muon ◄────────── 独立优化器模块

      │
      ▼
Phase 2 (依赖 Phase 1 的部分模块)
═══
P2-1 EC-GATv2 ◄────── 需要 P0-1 + P1-2 (attention 模块)
P2-2 Data Aug ◄────── 需要 P0-3 (Dropout rng) + P1-4 (Muon 协同)
P2-3 Distillation ─── 独立训练端模块
P2-4 INT8 Quant ◄──── 独立推理端模块 (只在 infer/ 侧, 不依赖 Phase 0/1)
P2-5 Hyper Search ─── Python 工具, 调用 profiler + train 二进制

      │
      ▼
Phase 3 (依赖 Phase 0+1 多项)
═══
P3-1 Embed+RoPE ◄──── 需要 P1-2 (MHA) + P0-1 (RMSNorm)
P3-2 Seq2Seq ◄─────── 需要 P1-1 (minGRU) + P1-2 (cross-att) + P0-1 + P0-2

      │
      ▼
Phase 4 (远期, 按需)
═══
P4-1 MobileNet Config ─ 需要 P0-1 + P0-2
P4-2 U-Net ◄─────────── 需要 P0-1 + P0-2 + P1-3 (SwiGLU)
P4-3 Mamba-2 ◄───────── 需要 P0-1 + parallel scan (同 P1-1)
```

---

## 实施顺序总览

```
现在 ────────────────────────────────────────────────────────→ 未来

│ Phase 0 (已完成✅)      │ Phase 1              │ Phase 2          │ Phase 3       │ Phase 4
│ (基础设施)              │ (能力升级)            │ (领域特化)        │ (新能力)       │ (按需)
│                        │                      │                  │               │
│ ┌──────────────┐      │ ┌──────────────┐    │ ┌─────────────┐  │ ┌───────────┐ │ ┌─────────┐
│ │ RMSNorm      │ ───→ │ │ minGRU       │    │ │ EC-GATv2    │  │ │ Embed+RoPE │ │ │ MobileNet│
│ │ (3天)        │      │ │ (1周)         │    │ │ (1周)        │  │ │ (1周)      │ │ │ (按需)   │
│ └──────────────┘      │ └──────────────┘    │ └─────────────┘  │ └───────────┘ │ └─────────┘
│ ┌──────────────┐      │ ┌──────────────┐    │ ┌─────────────┐  │ ┌───────────┐ │ ┌─────────┐
│ │ LAuReL-RW    │ ───→ │ │ MHA+GQA      │    │ │ Data Aug    │  │ │ Seq2Seq   │ │ │ U-Net    │
│ │ (1周)        │      │ │ (1.5周)       │    │ │ (5天)        │  │ │ (2周)      │ │ │ (按需)   │
│ └──────────────┘      │ └──────────────┘    │ └─────────────┘  │ └───────────┘ │ └─────────┘
│ ┌──────────────┐      │ ┌──────────────┐    │ ┌─────────────┐  │               │ ┌─────────┐
│ │ Dropout 通用 │ ───→ │ │ SwiGLU       │    │ │ Distill     │  │               │ │ Mamba-2 │
│ │ (3天)        │      │ │ (3天)         │    │ │ (1周)        │  │               │ │ (按需)   │
│ └──────────────┘      │ └──────────────┘    │ └─────────────┘  │               │ └─────────┘
│                       │ ┌──────────────┐    │ ┌─────────────┐  │               │
│                       │ │ Muon         │    │ │ INT8 Quant  │  │               │
│                       │ │ (1.5周)       │    │ │ (1周)        │  │               │
│                       │ └──────────────┘    │ └─────────────┘  │               │
│                       │                     │ ┌─────────────┐  │               │
│                       │                     │ │ HyperSearch │  │               │
│                       │                     │ │ (5天)        │  │               │
│                       │                     │ └─────────────┘  │               │
```

---

## 关键参考文献

| 领域 | 核心论文 |
|---|---|
| 残差连接 | He et al., ResNet (2015); LAuReL, ICML 2025; SCORE, arXiv 2603.10544 |
| 序列模型 | minLSTM/minGRU, ICLR 2025; Mamba-2, Dao & Gu, ICML 2024 |
| 注意力机制 | Vaswani et al., Transformer (2017); GQA, Ainslie et al. 2023; KHA, arXiv 2510.23052 |
| 归一化 | RMSNorm (2019); Peri-LN, ICML 2025; FlashNorm, arXiv 2407.09577v4 |
| 激活函数 | SwiGLU, Shazeer 2020; ReLU², Nemotron-4 2024 |
| 优化器 | Muon, Keller Jordan 2024; Sophia, Liu et al., ICLR 2024 |
| 图神经网络 | GATv2, Brody et al. 2022; EC-GATv2, Zhang et al. 2025 |
| 知识蒸馏 | Hinton et al. 2015; LRC, arXiv 2025; EA-KD, ICCV 2025 |
| 嵌入 | RoPE, Su et al. 2023 |
| 数据增强 | Mixup, Zhang et al. 2018; CutMix, Yun et al. 2019; RandAugment, Cubuk et al. 2020 |
| 量化 | INT8 Post-Training Quantization, Jacob et al. 2018; Per-Channel Quantization, Krishnamoorthi 2018 |
| 超参数搜索 | Bayesian Optimization, Snoek et al. 2012; Hyperband, Li et al. 2018; PrototypeNAS, arXiv 2603.15106 |
