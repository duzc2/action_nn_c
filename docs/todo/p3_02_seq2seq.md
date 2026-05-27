# P3-02: Seq2Seq Encoder-Decoder

> 阶段: Phase 3 | 工作量: 大 | 前置依赖: P0-01 (RMSNorm), P0-02 (LAuReL-RW), P1-01 (minGRU), P1-02 (MHA+GQA/cross-attention)

---

## 1. 是什么

在 profiler 层支持将两个独立注册的网络组装为 Encoder-Decoder 架构，实现"输入序列 → encoder 编码 → decoder 逐步生成 → 输出序列"的能力。

```
Encoder-Decoder 架构:

  Input Sequence [x₁, x₂, ..., x_T]
        │
   ┌────▼─────────────────┐
   │   Encoder Network    │  (例如 minGRU 或 Transformer)
   │   编码为 fixed-size  │
   │   encoder_output     │
   └────┬─────────────────┘
        │  encoder_output = [h₁, h₂, ..., h_T]  (或 final state)
        │
   ┌────▼─────────────────────────────┐
   │   Decoder Network                │
   │   - Self-Attention (causal)      │
   │   - Cross-Attention(Q=dec, K/V=enc)  ← 关键创新
   │   - Autoregressive 生成         │
   └────┬─────────────────────────────┘
        │
   Output Sequence [y₁, y₂, ..., y_M]  (逐 token 生成)
```

## 2. 为什么需要

- 目前所有网络都是"输入→输出"单次映射，无序列到序列的能力
- weather 等多步时序预测天然是 seq2seq (历史序列 → 未来多步)
- snake 等多轮决策可以 encoder 理解环境 → decoder 生成动作序列
- 不新增独立网络类型 — 复用现有的 minGRU/Transformer 作为 encoder/decoder 单元
- cross-attention 是 Transformer 的标准组件，P1-02 的 MHA 模块已预留 cross-attention 接口

## 3. 详细子任务

### 3.1 Cross-Attention 实现

- [ ] **Cross-Attention 前向**
  ```c
  // Decoder 的 cross-attention: Q 来自 decoder hidden state
  //                              K, V 来自 encoder output
  void cross_attention_forward(
      float *output,                        // [tgt_len × d_model]
      const float *Q_dec,                   // [tgt_len × d_model] decoder queries
      const float *K_enc,                   // [src_len × d_model] encoder keys
      const float *V_enc,                   // [src_len × d_model] encoder values
      const float *W_q, const float *W_k, const float *W_v,  // 投影权重
      const float *W_o, const float *b_o,   // 输出投影
      int tgt_len, int src_len, int d_model,
      int num_heads, int num_kv_heads,
      float *attn_weights);                 // [num_heads × tgt_len × src_len] 缓存
  ```

- [ ] **Cross-Attention 反向**
  ```c
  void cross_attention_backward(
      float *dQ_dec, float *dK_enc, float *dV_enc,
      float *dW_q, float *dW_k, float *dW_v, float *dW_o, float *db_o,
      const float *dout,
      const float *Q_dec, const float *K_enc, const float *V_enc,
      const float *attn_weights,
      int tgt_len, int src_len, int d_model,
      int num_heads, int num_kv_heads);
  ```

- [ ] **复用 P1-02 的 scaled_dot_product 模块**
  - Self-attention: Q, K, V 都来自同一源 → 现有 `attention_forward`
  - Cross-attention: Q 来自 decoder, K/V 来自 encoder → `cross_attention_forward`
  - 共享底层 `softmax(QK^T/√d)·V` 计算，仅输入来源不同

### 3.2 Encoder 实现

- [ ] **Encoder 前向**
  ```c
  // Encoder 处理整个输入序列，输出编码表示
  // 支持 minGRU encoder 或 Transformer encoder

  typedef struct {
      void *encoder_net;       // 指向已注册的 encoder 网络
      char *encoder_type;      // "rnn_min_gru" 或 "transformer"
      float *hidden_states;    // [src_len × d_model] 可选, 最后保留全部 hidden states
      float *final_state;      // [d_model] encoder 最终状态 (可选, 用于 decoder 初始化)
      int src_len;
      int d_model;
  } EncoderState;

  void encoder_forward(EncoderState *enc,
                        const float *input,    // [src_len × input_dim]
                        int src_len);
  ```

- [ ] **Encoder 配置**
  ```c
  // 在 profiler 中指定 encoder 子网络
  // encoder 网络可以是任意已注册类型 (minGRU, Transformer, MLP)
  ```

### 3.3 Decoder 实现

- [ ] **Decoder 单步推理 (autoregressive)**
  ```c
  // 推理时逐 token 生成
  void decoder_step(
      float *output_token,                   // [output_dim] 当前步输出
      const float *prev_token,               // [output_dim] 上一步的 token
      void *decoder_state,                   // decoder 内部状态 (minGRU hidden)
      const float *encoder_output,           // [src_len × d_model] encoder 输出
      int src_len, int d_model,
      int step);                             // 当前步数
  ```

- [ ] **Decoder 训练前向 (teacher forcing)**
  ```c
  // 训练时使用 teacher forcing: 给定完整目标序列, 一次性并行计算
  void decoder_train_forward(
      float *output,                         // [tgt_len × output_dim]
      const float *target_input,             // [tgt_len × output_dim] 右移一位的目标
      const float *encoder_output,           // [src_len × d_model] encoder 输出
      int src_len, int tgt_len, int d_model,
      int num_heads, int num_kv_heads,
      int causal_mask);                     // decoder 自注意力必须是 causal
  ```

- [ ] **Decoder 层级结构**
  ```
  for each decoder layer:
    1. Self-Attention (causal):
       Q/K/V 来自 decoder 自己的 hidden states
       注意: 必须是 causal mask (只能向前看)
       → 输出 after_self_attn

    2. Cross-Attention:
       Q 来自 after_self_attn (decoder)
       K/V 来自 encoder_output
       → 输出 after_cross_attn

    3. FFN (SwiGLU 或 MLP):
       after_cross_attn → FFN → output

    4. RMSNorm + Residual: 每步归一化 + 残差连接
  ```

### 3.4 Profiler 层支持

- [ ] **网络组装接口**
  ```c
  // profiler 中声明 Seq2Seq 架构
  typedef struct {
      char *encoder_network_def;   // encoder 子网络的 JSON/配置
      char *decoder_network_def;   // decoder 子网络的 JSON/配置
      int src_max_len;             // encoder 输入最大长度
      int tgt_max_len;             // decoder 输出最大长度
      int d_model;                 // encoder/decoder 共享的模型维度
      int share_embedding;         // 是否共享 encoder/decoder 的 embedding (1/0)
      int bos_token_id;            // 开始 token ID
      int eos_token_id;            // 结束 token ID
  } Seq2SeqConfig;
  ```

- [ ] **代码生成逻辑**
  ```
  profiler 收到 Seq2Seq 定义后:
    1. 分别生成 encoder 和 decoder 的 C 代码
    2. 生成连接代码: encoder_output → decoder cross-attention
    3. 生成 decoder autoregressive 循环 (推理) 和 teacher forcing 代码 (训练)
    4. 组装为完整的 seq2seq_train.c 和 seq2seq_infer.c
  ```

### 3.5 注意力融合 (可选优化)

- [ ] **MQA (Multi-Query Attention) 支持**
  - `num_kv_heads = 1` → 所有 query head 共享一对 K/V
  - 极大加速 decoder 推理 (每步只需 1 组 K/V 计算)
  - 是 cross-attention 的常见加速手段

- [ ] **KV Cache (可选)**
  - decoder 推理时缓存已生成的 K/V 投影
  - 每步只在当前 token 上计算 K/V 投影, 历史 K/V 复用
  - 将 decoder 自注意力复杂度从 O(T²) 降到 O(T)

### 3.6 测试

- [ ] **单元测试** — cross-attention: Q(i)·K(j) 得分矩阵正确
- [ ] **单元测试** — cross-attention: 当 K_enc = V_enc = input 时, 输出合理
- [ ] **单元测试** — decoder causal mask: 位置 i 不能 attend 到 j>i
- [ ] **单元测试** — teacher forcing 训练时 loss 随 epoch 递减
- [ ] **单元测试** — encoder-decoder 反向传播: encoder 和 decoder 的梯度都非零
- [ ] **集成测试** — weather: encoder 编码历史 30 天, decoder 预测未来 7 天
- [ ] **集成测试** — snake: encoder 编码游戏状态, decoder 生成动作序列
- [ ] **集成测试** — 简单 copy task: input="abc", output="abc" (验证 encoder-decoder 能力)

## 4. API 设计

```c
// seq2seq.h (generated by profiler, 非手写模块)
// 以下为 profiler 生成的代码结构示例:

// ==========================================
// 训练 API
// ==========================================
typedef struct {
    EncoderState encoder;
    DecoderState decoder;
    Seq2SeqConfig config;
    // 损失函数
    float train_loss;  // cross-entropy over all target tokens
} Seq2SeqTrainState;

// 一次完整的前向+反向
void seq2seq_train_step(Seq2SeqTrainState *state,
                         const float *src_seq,   // [src_len × input_dim]
                         const float *tgt_seq,   // [tgt_len × output_dim]
                         const float *tgt_label, // [tgt_len] token ids for loss
                         float learning_rate);

// ==========================================
// 推理 API
// ==========================================
typedef struct {
    EncoderState encoder;
    DecoderInferState decoder;  // 包含 KV cache
    Seq2SeqConfig config;
} Seq2SeqInferState;

// 推理: 给定源序列, autoregressive 生成目标序列
int seq2seq_infer_generate(Seq2SeqInferState *state,
                            const float *src_seq,  // [src_len × input_dim]
                            int src_len,
                            float *output_seq,      // [max_tgt_len × output_dim]
                            int max_tgt_len,
                            int *actual_tgt_len);   // 实际生成长度 (遇到 EOS 停止)
```

## 5. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `src/nn/attention/scaled_dot_product_cross.c` — cross-attention 变体 |
| **修改** | `src/nn/attention/scaled_dot_product.h` — 增加 cross-attention 接口 |
| **修改** | `src/nn/attention/scaled_dot_product.c` — cross-attention 实现 |
| **修改** | `src/profiler/` — Seq2Seq 组装逻辑 (核心改动, ~500 行新增) |
| **修改** | `src/nn/types/transformer/transformer_forward.c` — decoder 层支持 cross-attn |
| **修改** | `src/nn/types/transformer/transformer_train_ops.c` — cross-attn backward |
| 各 seq2seq demo 的 profiler 配置文件 (per-demo) |

## 6. 验收标准

1. Cross-attention: Q·K^T 维度 [tgt_len × src_len], softmax 在 src_len 维度上归一化
2. Decoder causal mask: attn_weights[i][j] = -inf for j > i
3. Copy task: 输入序列经 encoder-decoder 能准确复制输出
4. Teacher forcing 训练: 训练 loss 随 epoch 递减
5. Autoregressive 推理: 遇到 EOS 时停止, 输出序列长度正确
6. weather: 30 天历史 → 7 天预测, MAE < 直接预测的基准
7. Backward: encoder 和 decoder 的权重梯度均非零, 梯度检查通过 (< 1e-4)

---

**参考论文:** Encoder-Decoder, Sutskever et al. 2014 (Seq2Seq); Transformer, Vaswani et al. 2017; minGRU, Feng et al. ICLR 2025
