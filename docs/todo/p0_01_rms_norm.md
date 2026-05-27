# P0-01: RMSNorm 归一化层

> 阶段: Phase 0 | 工作量: 小 | 前置依赖: 无 | 可与 P0-02、P0-03 并行

---

## 1. 是什么

RMSNorm 是一种归一化层，仅用输入的均方根（Root Mean Square）做缩放，不做均值减法。

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) · γ

其中 γ 是逐元素可学习缩放参数，ε 是小常数防除零。
```

2026 年研究证明 LayerNorm 的均值减法步骤在神经网络中冗余（被减去分量平均接近 0），所以 RMSNorm 是更简洁高效的等价替代。

## 2. 为什么需要

- 当前归一化只有 BN，且耦合在 CNN 内部，MLP/RNN/GNN/Transformer 训练时无归一化可用
- 深层训练时 Peri-LN 论文证明的 Pre-LN 指数方差增长会导致 FP32 溢出 → 训练崩溃
- RMSNorm 核心仅 ~30 行 C，接入工作量远小于 LayerNorm

## 3. 详细子任务

### 3.1 新增共享模块

- [ ] **创建 `src/nn/norm/rms_norm.h`**
  ```c
  // 前向传播
  void rms_norm_forward(float *y, const float *x, const float *gamma,
                         int n, float epsilon);
  // 反向传播
  void rms_norm_backward(float *dx, float *dgamma, const float *dy,
                          const float *x, const float *gamma,
                          int n, float epsilon);
  ```
- [ ] **创建 `src/nn/norm/rms_norm.c`** — 实现上述函数
  - forward: `rms = sqrt(mean(x²)+ε)` → `y[i] = x[i] / rms * gamma[i % g]`
  - backward: 对 x 和 gamma 分别求导
- [ ] **创建 `src/nn/norm/CMakeLists.txt`** — `ACTION_C_ENABLE_NN_NORM` 开关
- [ ] **更新 `src/nn/CMakeLists.txt`** — 添加 `add_subdirectory(norm)`

### 3.2 各网络类型接入

- [ ] **MLP 接入** (`src/nn/types/mlp/`)
  - `mlp_config.h`: `MlpConfig` 增加 `int use_rms_norm;`
  - `mlp_layers.h`: `MlpDenseLayer` 增加 `float *norm_gamma;`
  - `mlp_infer_ops.c`: dense 前向传播后，若 `use_rms_norm` 则调用 `rms_norm_forward`
  - `mlp_train_ops.c`: 反向传播时调用 `rms_norm_backward`

- [ ] **CNN 接入** (`src/nn/types/cnn/`)
  - `cnn_config.h`: `CnnConfig` 增加 `int norm_type;` (0=NONE, 1=BN, 2=RMSNorm)
  - `cnn_infer_ops.c`: conv 输出后，根据 `norm_type` 选择 BN 或 RMSNorm
  - `cnn_train_ops.c`: 对应的 backward 分支

- [ ] **RNN 接入** (`src/nn/types/rnn/`)
  - 在输入→隐藏投影后、隐藏→输出投影后可加 RMSNorm
  - `rnn_config.h`: 增加 `int use_rms_norm;`

- [ ] **GNN 接入** (`src/nn/types/gnn/`)
  - 消息聚合后加 RMSNorm
  - `gnn_config.h`: 增加 `int use_rms_norm;`

- [ ] **Transformer 接入** (`src/nn/types/transformer/`)
  - Attention 子层输出后 + RMSNorm（Pre-Norm 模式）
  - FFN 子层输出后 + RMSNorm
  - 即 Peri-LN 模式：`x + Norm(Module(Norm(x)))`

### 3.3 测试

- [ ] **单元测试** — 验证 RMSNorm forward/backward
  - 对已知输入验证输出（与 PyTorch/FLAX 参考实现对比）
  - 验证 gamma 梯度正确性
  - 验证数值稳定性（极值输入不产生 NaN）

- [ ] **集成测试** — 在现有 demo 上验证
  - 选择 transformer demo：加 RMSNorm 后训练 loss 曲线应更稳定
  - 选择 cnn_rnn_react demo：加 RMSNorm 后训练不崩溃

## 4. API 设计

```c
// 分组归一化 (group_norm=1 即逐元素，group_norm=n 即标准 RMSNorm)
// 实际实现时 group_norm 固定为输入维度 n（标准 RMSNorm）
void rms_norm_forward(float *y, const float *x, const float *gamma,
                       int n, float epsilon);

// backward 输出 dx (梯度对输入), dgamma (梯度对 gamma)
void rms_norm_backward(float *dx, float *dgamma, const float *dy,
                        const float *x, const float *gamma,
                        int n, float epsilon);
```

## 5. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `src/nn/norm/rms_norm.h` |
| **新建** | `src/nn/norm/rms_norm.c` |
| **新建** | `src/nn/norm/CMakeLists.txt` |
| **修改** | `src/nn/CMakeLists.txt` |
| **修改** | `src/nn/types/mlp/mlp_config.h` |
| **修改** | `src/nn/types/mlp/mlp_layers.h` |
| **修改** | `src/nn/types/mlp/mlp_infer_ops.c` |
| **修改** | `src/nn/types/mlp/mlp_train_ops.c` |
| **修改** | `src/nn/types/cnn/cnn_config.h` |
| **修改** | `src/nn/types/cnn/cnn_infer_ops.c` |
| **修改** | `src/nn/types/cnn/cnn_train_ops.c` |
| **修改** | `src/nn/types/rnn/rnn_config.h` |
| **修改** | `src/nn/types/rnn/rnn_infer_ops.c` |
| **修改** | `src/nn/types/rnn/rnn_train_ops.c` |
| **修改** | `src/nn/types/gnn/gnn_config.h` |
| **修改** | `src/nn/types/gnn/gnn_infer_ops.c` |
| **修改** | `src/nn/types/gnn/gnn_train_ops.c` |
| **修改** | `src/nn/types/transformer/transformer_config.h` |
| **修改** | `src/nn/types/transformer/transformer_forward.c` |
| **修改** | `src/nn/types/transformer/transformer_infer_ops.c` |
| **修改** | `src/nn/types/transformer/transformer_train_ops.c` |

## 6. 验收标准

1. RMSNorm 前向传播与 PyTorch `nn.RMSNorm` 输出一致（误差 < 1e-5）
2. RMSNorm 反向传播梯度与 PyTorch autograd 一致（误差 < 1e-5）
3. MLP + RMSNorm 在 mnist demo 上可正常训练（loss 下降）
4. Transformer + RMSNorm 在 transformer demo 上训练 loss 曲线比不加更稳定
5. 极端输入 (x 全零、x 极大值) 不产生 NaN 或 inf

---

**参考论文:** RMSNorm, Zhang & Sennrich 2019; Peri-LN, Kim et al., ICML 2025
