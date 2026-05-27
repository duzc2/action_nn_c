# P1-02: 多头注意力 + GQA

> 阶段: Phase 1 | 工作量: 中 | 前置依赖: P0-01 (RMSNorm), P0-02 (skip)

---

## 1. 是什么

将当前单头自注意力升级为可配置多头的注意力模块，支持 MHA/GQA 两种模式。

```
MHA (Multi-Head Attention):
  把 Q/K/V 沿维度拆为 num_heads 组，每组独立做 scaled dot-product attention，
  最后拼接并通过 O_proj 投影。

GQA (Grouped-Query Attention):
  若 num_kv_heads < num_heads，多个 query head 共享一组 K/V。
  KV cache 减少 (num_heads - num_kv_heads) / num_heads。
```

## 2. 为什么需要

- 当前 Transformer 仅单头注意力，hybrid_route 的语义理解受限
- 独立 `src/nn/attention/` 模块可供 Transformer (self-attn)、GNN (EC-GATv2，P2-01)、Seq2Seq (cross-attn，P3-02) 复用
- GQA 是工业标准 (Llama/Gemma/Mistral)，KV cache 减少 87.5%

## 3. 详细子任务

### 3.1 独立注意力模块

- [ ] **创建 `src/nn/attention/scaled_dot_product.h`**
  ```c
  // MHA: 标准多头注意力
  void mha_forward(float *output,           // [seq_len × d_model]
                   const float *Q, const float *K, const float *V,
                   int seq_len, int d_model,
                   int num_heads,            // 头数
                   const float *W_o,         // 输出投影 [d_model × d_model]
                   const float *b_o,
                   int causal_mask);        // 1=因果遮罩, 0=双向

  // GQA: 分组查询注意力
  void gqa_forward(float *output,
                   const float *Q, const float *K, const float *V,
                   int seq_len, int d_model,
                   int num_heads, int num_kv_heads,  // num_kv_heads ≤ num_heads
                   const float *W_o, const float *b_o,
                   int causal_mask);

  // 反向传播 (用于训练)
  void attention_backward(float *dQ, float *dK, float *dV, float *dW_o,
                           const float *dout,
                           const float *Q, const float *K, const float *V,
                           const float *attn_weights,   // forward 缓存
                           int seq_len, int d_model,
                           int num_heads, int num_kv_heads);
  ```

- [ ] **创建 `src/nn/attention/scaled_dot_product.c`**
  - 实现 softmax(Q·K^T / √d_k) · V
  - GQA 的 KV 共享逻辑：`K/V` 投影后 broadcast 到所有 query head
  - 支持 causal_mask（上三角置 -inf）
  - 训练 forward 缓存 attention_weights 供 backward 使用

- [ ] **创建 `src/nn/attention/CMakeLists.txt`** — `ACTION_C_ENABLE_ATTENTION` 开关

### 3.2 Transformer 接入

- [ ] **`transformer_config.h` — 增加配置**
  ```c
  int num_heads;        // 默认 1, 向后兼容
  int num_kv_heads;     // 默认 = num_heads (MHA), <num_heads (GQA)
  int head_dim;         // d_model / num_heads
  ```
- [ ] **`transformer_forward.c` — 替换单头为 MHA/GQA**
  - 根据 `num_heads` 分配并拆分 Q/K/V
  - 若 `num_kv_heads < num_heads`，调用 `gqa_forward`
  - 否则调用 `mha_forward`
- [ ] **`transformer_train_ops.c` — 替换训练反向传播**

### 3.3 KHA (可选增强)

- [ ] **若后期加入 KHA (跨头交互)**
  - 在 Q/K/V 投影后乘共享变换矩阵 T_Q/T_K/T_V
  - 初始化为单位矩阵 (训练初期行为 = 标准 MHA)
  - 推理时将 T 吸收进投影权重 (零推理开销)
  - **不纳入当前阶段，仅预留接口**

### 3.4 测试

- [ ] **单元测试** — MHA 与 PyTorch `nn.MultiheadAttention` 对比
  - test: seq=8, d_model=16, num_heads=4。输出误差 < 1e-5
- [ ] **单元测试** — GQA 与 MHA 比较
  - 当 num_kv_heads=num_heads 时，GQA 输出 == MHA 输出
- [ ] **单元测试** — causal_mask: 位置 i 不能 attend 到 j>i
- [ ] **单元测试** — 反向传播梯度正确
- [ ] **集成测试** — transformer demo 用多头注意力后 loss 更低
- [ ] **集成测试** — hybrid_route demo 用多头后判断准确率提升

## 4. API 设计

```c
// scaled_dot_product.h

// 注意力 forward, 根据 num_kv_heads 自动选择 MHA 或 GQA
void attention_forward(
    float *output,                       // [seq_len × d_model]
    const float *Q, const float *K, const float *V,
    const float *W_q, const float *W_k, const float *W_v, // 输入投影
    const float *W_o, const float *b_o,  // 输出投影
    int seq_len, int d_model,
    int num_heads, int num_kv_heads,     // num_kv_heads=0 时自动用 num_heads
    int causal_mask,
    // 缓存 (训练时需要, 推理时传 NULL)
    float **cached_attn_weights);        // [num_heads × seq_len × seq_len]

void attention_backward(
    float *dQ, float *dK, float *dV,
    float *dW_q, float *dW_k, float *dW_v, float *dW_o, float *db_o,
    const float *dout,
    const float *Q, const float *K, const float *V,
    const float *W_q, const float *W_k, const float *W_v,
    const float *attn_weights,           // forward 时缓存的
    int seq_len, int d_model,
    int num_heads, int num_kv_heads);
```

## 5. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `src/nn/attention/scaled_dot_product.h` |
| **新建** | `src/nn/attention/scaled_dot_product.c` |
| **新建** | `src/nn/attention/CMakeLists.txt` |
| **修改** | `src/nn/CMakeLists.txt` |
| **修改** | `src/nn/types/transformer/transformer_config.h` |
| **修改** | `src/nn/types/transformer/transformer_forward.c` |
| **修改** | `src/nn/types/transformer/transformer_infer_ops.c` |
| **修改** | `src/nn/types/transformer/transformer_train_ops.c` |

## 6. 验收标准

1. MHA forward 与 PyTorch 一致 (< 1e-5)
2. GQA (num_kv_heads=num_heads) 输出 == MHA 输出
3. causal_mask 正确：attend_weights[i][j] == -inf for j>i
4. backward 梯度正确（与数值梯度比较）
5. transformer demo 使用 num_heads=4 后 loss 显著低于 num_heads=1
6. num_heads=1 时完全向后兼容 (输出 == 当前单头)

---

**参考论文:** Transformer, Vaswani et al. 2017; GQA, Ainslie et al. 2023; KHA, Zhou et al., arXiv 2510.23052
