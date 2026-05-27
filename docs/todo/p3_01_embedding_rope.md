# P3-01: Embedding Layer + RoPE 位置编码

> 阶段: Phase 3 | 工作量: 中 | 前置依赖: P0-01 (RMSNorm), P1-02 (多头注意力)

---

## 1. 是什么

为 Transformer 增加可学习的 token embedding 替代当前的字符级 tokenizer，并引入 RoPE (Rotary Position Embedding) 位置编码。

```
Token Embedding:
  input:  token indices [seq_len]  (整数 token ID)
  output: embeddings [seq_len × emb_dim]  (通过查找表)

RoPE 位置编码:
  将位置信息编码为 Q/K 的旋转，使注意力得分自然依赖相对位置
  Q_rope[pos] = Q[pos] ⊙ cos(pos·θ) + rotate(Q[pos]) ⊙ sin(pos·θ)
  K_rope[pos] = K[pos] ⊙ cos(pos·θ) + rotate(K[pos]) ⊙ sin(pos·θ)

  关键属性:
    Q^T_rope[pos_q] · K_rope[pos_k] = f(pos_q - pos_k)  ← 仅依赖相对距离
```

## 2. 为什么需要

- 当前 Transformer 使用字符级 tokenizer，每个字符是独立 one-hot 编码（表达能力受限）
- hybrid_route 中需要更好的语义编码来理解上游 cue 词汇的语义关系
- RoPE 是工业标准位置编码（Llama/Mistral/Gemma/Qwen），支持长序列外推
- Embedding 层是当今所有 Transformer 的标准组件
- 与现有字符级 tokenizer 模式并存（embedding_mode 配置项控制）

## 3. 详细子任务

### 3.1 Token Embedding 层 (`src/nn/embedding/embedding.h` + `embedding.c`)

- [ ] **Embedding 数据结构**
  ```c
  typedef struct {
      float *weight;         // [vocab_size × emb_dim] 可学习查找表
      int vocab_size;        // 词汇表大小
      int emb_dim;           // 嵌入维度
      int padding_idx;       // padding token ID (-1 = 无)
  } EmbeddingLayer;

  void embedding_init(EmbeddingLayer *emb, int vocab_size, int emb_dim);
  void embedding_free(EmbeddingLayer *emb);
  ```

- [ ] **Embedding 前向传播 (查表)**
  ```c
  // 对每个 token index 查表取出对应的 emb_dim 维向量
  // output[i] = weight[token_ids[i]]  (逐行复制)
  void embedding_forward(float *output,             // [seq_len × emb_dim]
                          const int *token_ids,      // [seq_len]
                          int seq_len,
                          const EmbeddingLayer *emb);

  // 反向传播: 梯度累积到 weight 的对应行
  void embedding_backward(float *dweight,           // 累积梯度
                           const float *dout,        // [seq_len × emb_dim] 上游梯度
                           const int *token_ids,     // [seq_len]
                           int seq_len,
                           const EmbeddingLayer *emb);
  ```

- [ ] **可选的 scale 缩放**
  ```c
  // 嵌入后乘以 sqrt(emb_dim) — 标准 Transformer 实践
  void embedding_forward_scaled(float *output,
                                 const int *token_ids, int seq_len,
                                 const EmbeddingLayer *emb);
  // output = weight[ids] * sqrt(emb_dim)
  ```

### 3.2 RoPE 位置编码

- [ ] **预计算 cos/sin 表**
  ```c
  typedef struct {
      float *cos_table;  // [max_seq_len × head_dim/2]
      float *sin_table;  // [max_seq_len × head_dim/2]
      int max_seq_len;
      int head_dim;
      float base;        // 频率基数 (默认 10000)
  } RoPECache;

  void rope_cache_init(RoPECache *cache, int max_seq_len, int head_dim, float base);
  void rope_cache_free(RoPECache *cache);
  ```

- [ ] **预计算实现**
  ```c
  void rope_cache_init(RoPECache *cache, int max_seq_len, int head_dim, float base) {
      cache->max_seq_len = max_seq_len;
      cache->head_dim = head_dim;
      cache->base = base;
      int half = head_dim / 2;
      cache->cos_table = malloc(max_seq_len * half * sizeof(float));
      cache->sin_table = malloc(max_seq_len * half * sizeof(float));

      for (int pos = 0; pos < max_seq_len; pos++) {
          for (int i = 0; i < half; i++) {
              float theta = pos / powf(base, (2.0f * i) / head_dim);
              cache->cos_table[pos * half + i] = cosf(theta);
              cache->sin_table[pos * half + i] = sinf(theta);
          }
      }
  }
  ```

- [ ] **RoPE 前向 (应用到 Q 或 K)**
  ```c
  // 对 [seq_len × head_dim] 的 Q (或 K) 应用 RoPE
  // 旋转在 head_dim 维度上进行: (x0,x1) 配对旋转, (x2,x3) 配对旋转, ...
  void rope_apply(float *x,                         // [seq_len × head_dim] 输入+输出
                   int seq_len, int head_dim,
                   const RoPECache *cache,
                   int start_pos);                   // 起始位置 (用于 KV cache)

  // 实现:
  // 对每个位置 pos, 对 head_dim 内的每对相邻维度:
  //   x0' = x0 * cos - x1 * sin
  //   x1' = x0 * sin + x1 * cos
  void rope_apply_single(float *x, int head_dim,
                          int pos, const RoPECache *cache);
  ```

- [ ] **RoPE 反向传播**
  ```c
  void rope_backward(float *dx,                       // 输出梯度
                      const float *dout,              // 上游梯度
                      const float *x,                 // forward 输入
                      int seq_len, int head_dim,
                      const RoPECache *cache,
                      int start_pos);
  // RoPE 是线性操作 (旋转), 反向即反向旋转
  // dx0' = dout_x0 * cos + dout_x1 * sin   ← 注意符号
  // dx1' = dout_x1 * cos - dout_x0 * sin
  ```

### 3.3 Transformer 接入

- [ ] **`transformer_config.h` — 新增配置**
  ```c
  typedef enum {
      TRANSFORMER_EMBED_CHARACTER = 0,   // 当前: 字符级 one-hot (零参数)
      TRANSFORMER_EMBED_LEARNABLE,       // 可学习 Embedding + RoPE
  } TransformerEmbedMode;

  typedef struct {
      // ... 现有字段 ...
      TransformerEmbedMode embed_mode;   // 默认 CHARACTER (向后兼容)
      int vocab_size;                    // embed_mode==LEARNABLE 时有效
      int emb_dim;                       // 嵌入维度 (默认 = d_model)
      float rope_base;                   // RoPE 基数 (默认 10000)
      int max_seq_len;                   // 最大序列长度
  } TransformerConfig;
  ```

- [ ] **`transformer_forward.c` — 嵌入阶段分支**
  ```c
  if (cfg->embed_mode == TRANSFORMER_EMBED_CHARACTER) {
      // 现有逻辑: 字符 → one-hot → 投影
  } else {
      // 新逻辑: token_ids → embedding_lookup → scale → RoPE → 后续
      embedding_forward_scaled(x, token_ids, seq_len, emb);
      // 对每个 head 的 Q 和 K 应用 RoPE
      for (int h = 0; h < num_heads; h++) {
          rope_apply(Q + h * head_dim, seq_len, head_dim, rope_cache, 0);
          rope_apply(K + h * head_dim, seq_len, head_dim, rope_cache, 0);
      }
      // 注意: V 不应用 RoPE
  }
  ```

- [ ] **`transformer_train_ops.c` — embedding backward**
  - token embedding 权重梯度
  - RoPE backward 梯度

### 3.4 词表构建

- [ ] **简单 BPE 或固定词表**
  ```c
  // 方案 A: 固定词表 (字符 + 常见词) — 最简单, 适合小型 demo
  // 方案 B: 简单 BPE tokenizer — 基于训练数据统计

  // tokenizer.h (可选)
  void tokenizer_build_vocab(const char *text_corpus, int vocab_size,
                              char **vocab_table, int *vocab_len);
  int tokenizer_encode(const char *text, const char **vocab, int vocab_size,
                        int *token_ids, int max_len);
  void tokenizer_decode(const int *token_ids, int len,
                         const char **vocab, char *text, int max_text_len);
  ```

### 3.5 测试

- [ ] **单元测试** — embedding 查表: 已知 weight, 验证 output = weight[ids]
- [ ] **单元测试** — embedding backward: 梯度只在对应 token 行非零
- [ ] **单元测试** — RoPE 预计算: cos(0)=1, sin(0)=0
- [ ] **单元测试** — RoPE: 旋转 90° 后 (x, y) → (-y, x)
- [ ] **单元测试** — RoPE 相对性: `rope(Q, pos_a)·rope(K, pos_b)` 仅依赖 `pos_a - pos_b`
- [ ] **单元测试** — RoPE backward 与数值梯度一致
- [ ] **集成测试** — transformer (embed_mode=CHARACTER) 输出不变 (向后兼容)
- [ ] **集成测试** — transformer (embed_mode=LEARNABLE) 训练 loss 收敛

## 4. API 设计

```c
// embedding.h

// ==========================================
// Token Embedding
// ==========================================
typedef struct {
    float *weight;
    int vocab_size, emb_dim;
    int padding_idx;
} EmbeddingLayer;

void embedding_init(EmbeddingLayer *emb, int vocab_size, int emb_dim);
void embedding_free(EmbeddingLayer *emb);

// 保存/加载 embedding 权重
void embedding_save(const EmbeddingLayer *emb, FILE *fp);
void embedding_load(EmbeddingLayer *emb, FILE *fp);

// 前向: token_ids → embeddings
void embedding_forward(float *output, const int *token_ids,
                        int seq_len, const EmbeddingLayer *emb);
void embedding_forward_scaled(float *output, const int *token_ids,
                               int seq_len, const EmbeddingLayer *emb);

// 反向: 梯度传播
void embedding_backward(float *dweight, const float *dout,
                         const int *token_ids, int seq_len,
                         const EmbeddingLayer *emb);

// ==========================================
// RoPE 位置编码
// ==========================================
typedef struct {
    float *cos_table, *sin_table;
    int max_seq_len, head_dim;
    float base;
} RoPECache;

void rope_cache_init(RoPECache *cache, int max_seq_len,
                      int head_dim, float base);
void rope_cache_free(RoPECache *cache);

// 应用 RoPE 到 [seq_len × head_dim] 的张量
// x 为输入+输出, 就地修改
void rope_apply(float *x, int seq_len, int head_dim,
                 const RoPECache *cache, int start_pos);

// 反向传播
void rope_backward(float *dx, const float *dout, const float *x,
                    int seq_len, int head_dim,
                    const RoPECache *cache, int start_pos);
```

## 5. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `src/nn/embedding/embedding.h` |
| **新建** | `src/nn/embedding/embedding.c` |
| **新建** | `src/nn/embedding/CMakeLists.txt` |
| **修改** | `src/nn/CMakeLists.txt` — 增加 embedding 子目录 |
| **修改** | `src/nn/types/transformer/transformer_config.h` — 新增 embed_mode |
| **修改** | `src/nn/types/transformer/transformer_forward.c` — 嵌入分支 |
| **修改** | `src/nn/types/transformer/transformer_train_ops.c` — embedding backward |

## 6. 验收标准

1. Embedding 查表: `output_row_i == weight[token_id_i]` 逐元素一致
2. RoPE: `rope(Q, pos_a)^T · rope(K, pos_b) = g(pos_a - pos_b)` (仅相对距离)
3. RoPE 反向与数值梯度误差 < 1e-4
4. `embed_mode=CHARACTER` 时输出与旧版完全一致 (向后兼容)
5. `embed_mode=LEARNABLE` 时 loss 在训练中收敛
6. RoPE 预计算 cos/sin 表在 `base=10000, head_dim=64, max_len=128` 时值域正确

---

**参考论文:** RoPE, Su et al. 2023 (RoFormer); Scaling RoPE (NTK-aware), bloc97 2023
