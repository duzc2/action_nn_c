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

- [x] **创建 `src/nn/norm/rms_norm.h`**
  ```c
  int rms_norm_forward(float *restrict output, const float *restrict input,
                       const float *restrict gamma, size_t n, float epsilon);
  int rms_norm_backward(float *restrict d_input, float *restrict d_gamma,
                        const float *restrict d_output, const float *restrict input,
                        const float *restrict gamma, size_t n, float epsilon);
  ```
- [x] **创建 `src/nn/norm/rms_norm.c`** — 实现上述函数
  - forward: `r = 1/sqrt(mean(x²)+eps)` → `y[i] = x[i] * r * gamma[i]`
  - backward: 对 x 和 gamma 分别求导，正确处理 gamma=NULL/d_gamma=NULL 情况
- [x] **已集成到 `src/nn/CMakeLists.txt`**（未创建独立 CMakeLists.txt，直接作为 NN_LAYER_SOURCES 的一部分）
  - 文件列表位于 `src/nn/CMakeLists.txt`，通过 `nn_infer_core` 库导出给各网络类型

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

- [x] **基本单元测试** (`tests/nn/test_rms_norm.c`) — 16 tests, 43 assertions
  - 对各维度 (n=1~16) 验证 forward 输出正确
  - 验证 backward d_gamma 累积正确
  - 验证 gamma=NULL 和 d_gamma=NULL 处理
  - 验证输入/输出 aliasing 安全
  - FD 梯度验证（与解析梯度对比）
  - 前向-反向 roundtrip 验证

- [x] **数值验证测试** (`tests/nn/test_rms_norm_numerical.c`) — 28 tests, 2,814 assertions
  - **前向跨维度**: n∈{1,2,4,8,16,32,64,128,256}，逐元素与手算对比
  - **epsilon 扫查**: 6 种 eps (1e-8 ~ 1e10)，验证范数单调性和趋零行为
  - **尺度不变性**: 验证 RMSNorm(αx, γ, eps=0) == RMSNorm(x, γ, eps=0) 精确成立
  - **符号保持**: positive→positive, negative→negative, zero→zero
  - **蒙特卡洛 FD 梯度**: 6 维度 × (5~20) 随机配置，每一组每个元素逐一 FD 验证
  - **d_gamma FD**: 逐一元素 FD 验证
  - **数值稳定性**: ±1e10、±1e-30、混合量级、eps=1e-15、eps=0、全相同值
  - **复合梯度流**: forward→loss→backward→GD，验证 loss 严格下降
  - **输出 RMS≈1**: n∈{1,4,8,16,32,64} 验证
  - **确定性**: 同输入 → 同输出

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
