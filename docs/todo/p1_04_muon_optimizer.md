# P1-04: Muon 优化器

> 阶段: Phase 1 | 工作量: 中 | 前置依赖: 无 (训练端独立模块) | 仅适用于 2D 矩阵参数，1D 参数用 AdamW

---

## 1. 是什么

Muon (MomentUm Orthogonalized by Newton-Schulz) 是一种矩阵级别的优化器。与 AdamW (逐标量) 不同，Muon 将每个权重矩阵作为一个线性变换整体来优化。

```
1. 动量:    M = β·M + G                   // β=0.95 (标准 Nesterov)
2. NS 正交化: X = NewtonSchulz(M, steps=5) // 使 X 的奇异值趋向 1
3. 更新:    θ = θ - lr·X - lr·wd·θ        // weight decay

Newton-Schulz 迭代 (5 次, 五阶多项式):
  X = G / ||G||F
  for i=1..5:
      A = X @ X^T
      B = b·A + c·(A@A)           // a≈3.44, b≈-4.78, c≈2.03
      X = a·X + B@X
```

对几十到几百维的小矩阵 (本项目的典型规模)，NS 迭代代价极低。

## 2. 为什么需要

- 当前仅 SGD + Adam。Muon 用 1 份动量替代 Adam 的 m+v (2 份) → 33% 优化器内存节省
- Moonshot AI 数据: 同一 loss 仅需 Adam 52% 的迭代步数
- 矩阵参数占本项目模型绝大部分 (MLP dense, CNN kernel, Transformer Q/K/V/O, GNN edge)
- NS 迭代对小型矩阵 (几十到几百维) 计算代价小，适合本项目

## 3. 详细子任务

### 3.1 核心实现

- [ ] **创建 `src/train/optimizer_muon.h`**
  ```c
  typedef struct {
      float lr;             // 学习率 (通常 0.02, 比 Adam 大)
      float momentum;       // β (默认 0.95)
      float weight_decay;   // (默认 0.01)
      int nesterov;         // 1=使用 Nesterov 加速
      int ns_steps;         // Newton-Schulz 迭代步数 (默认 5)
  } MuonConfig;

  typedef struct {
      float *momentum_buffer;  // 与参数同形状
      int rows, cols;          // 矩阵维度
  } MuonState;

  void muon_init(MuonState *state, int rows, int cols);
  void muon_update(float *params, float *grads,
                    MuonState *state, const MuonConfig *cfg);
  ```

- [ ] **创建 `src/train/optimizer_muon.c`**
  - `muon_update`:
    1. 动量: `M += G` (若 nesterov，先 `M = β*M + G` 再 `G += β*M`)
    2. NS: 若 rows > cols，转置矩阵 (NS 迭代对宽矩阵更快)
    3. 矩阵乘 `A = X @ X^T` (NS 迭代的核心操作)
    4. 多项式 `B = b·A + c·(A@A)`，然后 `X = a·X + B@X`
    5. 更新: `θ -= lr * X + lr * wd * θ`

- [ ] **矩阵乘助手** — 极小矩阵可直接循环，或引入简单 BLAS-3 循环
  ```c
  // 简单的 C = A @ B^T  (A: m×k, B: n×k, C: m×n)
  void matmul_transb(float *C, const float *A, const float *B,
                      int m, int n, int k);
  ```

### 3.2 参数判别 — 哪些用 Muon, 哪些用 AdamW

- [ ] **矩阵 vs 向量判别逻辑**
  ```c
  // 判别规则: params 在内存中是 [rows × cols] 二维布局 → Muon
  //           params 在内存中是 [size] 一维布局 → AdamW
  int is_matrix_param(int param_size, int known_rows, int known_cols);
  ```
- [ ] **各网络类型的判别**
  - MLP dense weight → Muon
  - MLP bias → AdamW
  - CNN conv kernel (展平为 2D) → Muon
  - CNN BN gamma/beta → AdamW
  - Transformer Q/K/V/O weight → Muon
  - Transformer bias, norm gamma → AdamW
  - Embedding table → AdamW (1D 查找表)

### 3.3 接入各网络类型

- [ ] **MLP 训练** — `mlp_train_ops.c`: 判别每个参数类型，混合调用 Muon/AdamW
- [ ] **CNN 训练** — `cnn_train_ops.c`: Conv kernel → Muon, bias/BN → AdamW
- [ ] **Transformer 训练** — `transformer_train_ops.c`: 同上
- [ ] **RNN/GNN 训练** — 同上

### 3.4 测试

- [ ] **单元测试** — NS 迭代：随机输入 X，验证 NS(X) 的奇异值接近 1
- [ ] **单元测试** — Muon update: 对简单二次函数，验证 loss 下降速度
- [ ] **集成测试** — snake: Muon vs Adam，相同 iterations 后比较 loss
- [ ] **集成测试** — mnist MLP: Muon vs Adam，相同 epochs 后比较验证集准确率
- [ ] **内存测试** — 验证 Muon 使用的优化器内存比 Adam 少 ~33%

## 4. API 设计

```c
// optimizer_muon.h

typedef struct {
    float lr;
    float momentum;
    float weight_decay;
    int nesterov;
    int ns_steps;
} MuonConfig;

void muon_config_default(MuonConfig *cfg);
// 设置: lr=0.02, momentum=0.95, wd=0.01, nesterov=1, ns_steps=5

typedef struct { float *buffer; int rows, cols; } MuonState;
void muon_init(MuonState *state, int rows, int cols);
void muon_free(MuonState *state);
void muon_update(float *params, float *grads, MuonState *state,
                  const MuonConfig *cfg);

// Newton-Schulz 正交化 (单独暴露, 可测试)
void newton_schulz(float *X, int rows, int cols, int steps);
```

## 5. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `src/train/optimizer_muon.h` |
| **新建** | `src/train/optimizer_muon.c` |
| **修改** | `src/train/CMakeLists.txt` |
| **修改** | `src/nn/types/mlp/mlp_train_ops.c` |
| **修改** | `src/nn/types/cnn/cnn_train_ops.c` |
| **修改** | `src/nn/types/transformer/transformer_train_ops.c` |
| **修改** | `src/nn/types/rnn/rnn_train_ops.c` (minGRU 接入时) |
| **修改** | `src/nn/types/gnn/gnn_train_ops.c` |

## 6. 验收标准

1. NS(X) 输出矩阵的所有奇异值在 [0.5, 1.5] 范围内
2. 对简单的二次优化问题，Muon loss 下降速度 ≥ Adam
3. snake 训练: Muon 在相同 iteration 数后 loss 更低
4. 优化器内存: Muon (1 份缓冲区) < Adam (2 份缓冲区)
5. 1D 参数 (bias) 自动 fallback 到 AdamW，不报错
6. rows > cols 时 NS 迭代正确转置，输出形状不变

---

**参考论文:** Muon, Keller Jordan 2024 (NanoGPT modded); Moonshot AI, arXiv 2509.24406; Flash-Muon, nil0x9 2025
