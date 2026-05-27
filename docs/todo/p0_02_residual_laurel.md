# P0-02: LAuReL-RW 残差连接

> 阶段: Phase 0 | 工作量: 中 | 前置依赖: 无 | 可与 P0-01、P0-03 并行

---

## 1. 是什么

残差连接让网络层输出跳过自身，直接加到后续层。LAuReL-RW 是最轻量的可学习残差：

```
标准 ResNet:  y = F(x) + x
LAuReL-RW:    y = α · F(x) + β · x    (α, β 通过 sigmoid/softmax 归一化)
```

仅 +2 个标量参数/层（<0.003%），获得接近多一整层的收益（ICML 2025）。

## 2. 为什么需要

- 当前所有 6 种网络类型均为纯前馈，无跨层连接
- edge_video_preprocess (5层), nested_nav (嵌套子网) 的深度受限 — 加深则梯度退化
- LAuReL-RW 的 ROI 极高：2 参数换一层收益，特别适合本项目的小型网络
- 是后续 U-Net (P4-02) 等深层网络的前提条件

## 3. 详细子任务

### 3.1 通用 skip 接口设计

- [x] **定义 skip_mode 枚举** (已创建 `src/nn/residual/skip_connection.h`)
  ```c
  typedef enum {
      SKIP_NONE     = 0,  // y = F(x)
      SKIP_IDENTITY = 1,  // y = F(x) + x
      SKIP_LAUREL_RW = 2, // y = α·F(x) + β·x, α=sigmoid(α_raw), β=1-α
  } SkipMode;
  ```
- [x] **定义 skip 参数结构体 + 前向/反向 API**
  ```c
  typedef struct {
      SkipMode mode;
      float alpha_raw;      // 可学习参数，初始化为 0 → α≈0.5, β≈0.5
      float alpha_cache;    // sigmoid(alpha_raw) 缓存，forward 时更新
      float beta_cache;     // 1 - alpha_cache
      float grad_alpha_raw; // backward 累积
  } SkipConnection;
  void skip_init(SkipConnection *skip, SkipMode mode);
  void skip_forward(float *restrict out, const float *restrict module_out,
                     const float *restrict input, size_t n, SkipConnection *skip);
  void skip_backward(float *restrict d_module_out, float *restrict d_input,
                      const float *restrict module_out, const float *restrict input,
                      const float *restrict dout, size_t n, SkipConnection *skip);
  ```
- [x] **已实现 `src/nn/residual/skip_connection.c`**，集成到 `src/nn/CMakeLists.txt`

### 3.2 MLP 接入

- [ ] `mlp_config.h` — `MlpDenseLayer` 增加 `SkipConnection skip;`
- [ ] `mlp_infer_ops.c` — 前向传播：dense 输出后，若 `skip.mode != NONE`，计算 `y = α·F(x) + β·x`
  - 注意：skip 要求 F(x) 和 x 形状一致。若维度变化，暂时跳过 (mode=NONE) 或使用 LAuReL-LR (后续)
- [ ] `mlp_train_ops.c` — 反向传播：skip 参数的梯度计算
  - `d_alpha_raw = ∂L/∂y · (F(x) - x) · sigmoid'(α_raw)`

### 3.3 CNN 接入

- [ ] `cnn_config.h` — `CnnConfig` 增加 `SkipMode skip_mode; int skip_every_n;` (每 N 层插一个 skip)
- [ ] `cnn_infer_ops.c` — conv 输出后，若 shape 匹配 (通道数、空间尺寸一致)，应用 skip
  - 典型场景：`stride=1, channels_in==channels_out` 的同形残差块
- [ ] `cnn_train_ops.c` — 对应的 backward

### 3.4 Transformer 接入

- [ ] `transformer_config.h` — 增加 `SkipMode attn_skip_mode; SkipMode ffn_skip_mode;`
- [ ] `transformer_forward.c` — attention sublayer 后的 skip：
  ```
  attn_out = Attention(RMSNorm(x))  →  x + skip(attn_out, x)  (Peri-LN 模式)
  ```
  FFN sublayer 后的 skip：同上
- [ ] `transformer_train_ops.c` — backward 适配

### 3.5 RNN 接入

- [ ] `rnn_config.h` — 增加 skip 配置
  - RNN 场景特殊：skip 跨时间步而非跨层（SCORE 式共享块迭代）
  - 初步实现简化为：在时间维度的输入上做 skip（如第 t 帧输入连接 x_t-1）
- [ ] `rnn_infer_ops.c` / `rnn_train_ops.c` — 适配

### 3.6 GNN 接入

- [ ] `gnn_config.h` — 消息传递轮次间增加 skip
- [ ] `gnn_infer_ops.c` / `gnn_train_ops.c` — 适配

### 3.7 测试

- [x] **基本单元测试** (`tests/nn/test_skip_connection.c`) — 15 tests, 51 assertions
  - 三种模式 (NONE/IDENTITY/LAUREL_RW) 前向输出验证
  - LAuReL-RW sigmoid 缓存及 α+β=1 验证
  - backward d_fx/dx/grad_alpha_raw 正确性
  - FD 梯度验证
  - grad_alpha_raw 累积正确

- [x] **数值验证测试** (`tests/nn/test_skip_connection_numerical.c`) — 16 tests, 1,362 assertions
  - **前向精确性**: n∈{1,2,4,8,16,32,64,128} 跨维度、跨模式逐元素验证
  - **alpha_raw 扫查**: 11 个值 (-10 ~ +10)，验证 sigmoid 缓存和输出
  - **α+β=1**: 21 个 alpha_raw 值验证数学恒等式
  - **F(x)=x → y=x**: 对任意 α，验证 y ≡ x（5 个 α 值 × 128 维）
  - **渐近行为**: α→1→y≈F(x), α→0→y≈x, α=0.5 等权重
  - **FD d_fx/dx**: 16 维分解，每个元素逐一 FD 验证
  - **FD alpha_raw 蒙特卡洛**: 15 组随机配置，随机 alpha_raw∈[-3,3]
  - **梯度符号一致性**: F>>x→grad>0, F<<x→grad<0
  - **复合梯度流**: forward→loss→backward→GD on alpha_raw，loss 下降
  - **大维度**: n=1024 前向反向 spot-check 验证
  - **数值稳定性**: ±1e8 量级前向反向无 NaN/Inf
  - **模式一致性**: SKIP_NONE ≈ LAuReL(α≈1)

- [ ] **MLP mnist** — 同深度，有 skip vs 无 skip，验证 loss 更低/收敛更快
- [ ] **CNN edge_video_preprocess** — 加深网络 (加入 skip) 后精度提升
- [ ] **Transformer** — 加入 Peri-LN (Pre-Norm + skip) 后训练更稳定

## 4. API 设计

```c
// skip_connection.h

typedef enum { SKIP_NONE, SKIP_IDENTITY, SKIP_LAUREL_RW } SkipMode;

typedef struct {
    SkipMode mode;
    float alpha_raw;
    float alpha_cache;      // sigmoid(alpha_raw)
    float beta_cache;       // 1 - alpha_cache
} SkipConnection;

// 前向: 对已计算的模块输出和原始输入做 skip 融合
// out[i] = (mode==LAUREL)? alpha*module_out[i] + beta*input[i]
//        : (mode==IDENTITY)? module_out[i] + input[i]
//        : module_out[i]
void skip_forward(float *out, const float *module_out, const float *input,
                   int n, SkipConnection *skip);

// 反向: 计算 d_module_out, d_input, 并更新 skip->alpha_raw 梯度
void skip_backward(float *d_module_out, float *d_input,
                    const float *module_out, const float *input,
                    const float *dout, int n, SkipConnection *skip);
```

## 5. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `src/nn/skip_connection.h` |
| **新建** | `src/nn/skip_connection.c` |
| **修改** | `src/nn/types/mlp/mlp_config.h`, `mlp_layers.h`, `mlp_infer_ops.c`, `mlp_train_ops.c` |
| **修改** | `src/nn/types/cnn/cnn_config.h`, `cnn_infer_ops.c`, `cnn_train_ops.c` |
| **修改** | `src/nn/types/rnn/rnn_config.h`, `rnn_infer_ops.c`, `rnn_train_ops.c` |
| **修改** | `src/nn/types/gnn/gnn_config.h`, `gnn_infer_ops.c`, `gnn_train_ops.c` |
| **修改** | `src/nn/types/transformer/transformer_config.h`, `transformer_forward.c`, `transformer_infer_ops.c`, `transformer_train_ops.c` |

## 6. 验收标准

1. LAuReL-RW forward 与参考实现一致（误差 < 1e-5）
2. alpha_raw 梯度正确（与数值梯度比较 < 1e-4）
3. MLP mnist: skip vs no-skip, 同深度下 loss 更低
4. CNN edge_video_preprocess: 加深至 6-8 层 + skip，训练不退化
5. Transformer: Pre-Norm + skip 训练 loss 曲线稳定下降
6. shape 不匹配时自动 fallback 到 NONE，不崩溃

---

**参考论文:** LAuReL, Menghani et al., ICML 2025; ResNet, He et al. 2015; SCORE, arXiv 2603.10544
