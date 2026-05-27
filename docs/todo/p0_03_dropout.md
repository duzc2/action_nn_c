# P0-03: Dropout + Stochastic Depth 正则化

> 阶段: Phase 0 | 工作量: 小 | 前置依赖: 无 | 可与 P0-01、P0-02 并行

---

## 1. 是什么

**Dropout:** 训练时随机将一部分神经元输出置零，推理时关闭。强制网络学习冗余表示，防过拟合。

**Stochastic Depth (DropPath):** 随机跳过整层（输出 = 输入），比逐神经元 Dropout 对深层网络更有效。

```
Dropout:     train: y = mask ⊙ x / keep_prob     eval: y = x
DropPath:    train: y = (random<p)? x : F(x)/p    eval: y = F(x)
```

## 2. 为什么需要

- 当前 Dropout 仅在 CNN 内部 ad hoc 实现，MLP/Transformer/GNN 无可用正则化
- snake (82 params)、sevenseg 等小模型在样本不足时过拟合 → 需要 Dropout
- edge_video_preprocess 等深层网络 → Stochastic Depth 比 Dropout 更有效
- 共享模块化后，所有网络类型一次接入

## 3. 详细子任务

### 3.1 新增共享模块

- [ ] **创建 `src/nn/dropout/dropout.h`**
  ```c
  typedef struct {
      float keep_prob;      // 保留概率 (0.0~1.0)
      uint32_t rng_state;   // 随机数状态 (PCG 或 xoshiro)
      int training;         // 1=训练模式, 0=推理模式
  } DropoutLayer;

  void dropout_forward(float *y, const float *x, int n, DropoutLayer *dp);
  void dropout_backward(float *dx, const float *dy, const float *x,
                         int n, const DropoutLayer *dp);
  // DropPath: 整层跳过
  void droppath_forward(float *y, const float *fx, const float *x,
                         int n, DropoutLayer *dp);
  void droppath_backward(float *dfx, float *dx, const float *dy,
                          int n, const DropoutLayer *dp);
  ```

- [ ] **创建 `src/nn/dropout/dropout.c`**
  - 实现 PCG 随机数生成器（轻量、周期长、无 libc 依赖）
  - `dropout_forward`: 对每个元素 `random() < keep_prob ? x[i]/keep_prob : 0`
  - `droppath_forward`: 一次性决定整层通断
  - backward: 根据 forward 时的 mask 回传梯度

- [ ] **创建 `src/nn/dropout/CMakeLists.txt`** — `ACTION_C_ENABLE_DROPOUT` 开关

### 3.2 各网络类型接入

- [ ] **MLP 接入**
  - `mlp_config.h`: `MlpDenseLayer` 增加 `DropoutLayer dropout;`（dense 输出后应用）
  - `mlp_infer_ops.c`: 训练模式调用 `dropout_forward`，推理模式直通
  - `mlp_train_ops.c`: 反向传播调用 `dropout_backward`

- [ ] **CNN 接入**
  - 现有 CNN 的 dropout 改为调用共享模块（接口统一）
  - `cnn_config.h`: `CnnConfig` 的 dropout 参数保留，内部改用 `DropoutLayer`
  - `cnn_infer_ops.c` / `cnn_train_ops.c`: 替换实现

- [ ] **Transformer 接入**
  - `transformer_config.h`: attention output 和 FFN output 各加 `float dropout_keep_prob;`
  - `transformer_forward.c`: 子层输出后调用 `dropout_forward`
  - `transformer_train_ops.c`: 适配 backward

- [ ] **RNN / GNN 接入**
  - `rnn_config.h`: 隐藏状态后可选 Dropout
  - `gnn_config.h`: 消息聚合后可选 Dropout

- [ ] **DropPath 接入**
  - 所有接入 skip 连接的网络类型（P0-02 完成后），在 skip 路径上支持 DropPath

### 3.3 测试

- [ ] **单元测试** — Dropout forward 的 mask 比例正确
  - 多次调用后统计，被保留的元素比例 ≈ keep_prob（误差 < 5%）
  - backward 梯度根据 mask 正确缩放
- [ ] **集成测试** — snake + Dropout 过拟合程度比无 Dropout 更低
- [ ] **集成测试** — mnist MLP + Dropout 验证集准确率更高

## 4. API 设计

```c
// dropout.h

typedef struct {
    float keep_prob;
    uint32_t rng_state;
    int training;        // 由训练/推理上下文设置
} DropoutLayer;

void dropout_init(DropoutLayer *dp, float keep_prob, uint32_t seed);
void dropout_set_training(DropoutLayer *dp, int training);

void dropout_forward(float *y, const float *x, int n, DropoutLayer *dp);
void dropout_backward(float *dx, const float *dy, const float *x,
                       int n, const DropoutLayer *dp);

// DropPath 专门接口 (keep_prob 含义略有不同: p_drop = 1-keep_prob)
void droppath_forward(float *y, const float *fx, const float *x,
                       int n, DropoutLayer *dp);
void droppath_backward(float *dfx, float *dx, const float *dy,
                        int n, const DropoutLayer *dp);
```

## 5. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `src/nn/dropout/dropout.h` |
| **新建** | `src/nn/dropout/dropout.c` |
| **新建** | `src/nn/dropout/CMakeLists.txt` |
| **修改** | `src/nn/CMakeLists.txt` |
| **修改** | `src/nn/types/mlp/mlp_config.h`, `mlp_layers.h`, `mlp_infer_ops.c`, `mlp_train_ops.c` |
| **修改** | `src/nn/types/cnn/cnn_config.h`, `cnn_infer_ops.c`, `cnn_train_ops.c` (改为调用共享模块) |
| **修改** | `src/nn/types/transformer/transformer_config.h`, `transformer_forward.c`, `transformer_train_ops.c` |
| **修改** | `src/nn/types/rnn/rnn_config.h`, `rnn_infer_ops.c`, `rnn_train_ops.c` |
| **修改** | `src/nn/types/gnn/gnn_config.h`, `gnn_infer_ops.c`, `gnn_train_ops.c` |

## 6. 验收标准

1. Dropout forward 保留比例 ≈ keep_prob（1000 次试验统计，误差 < 5%）
2. Dropout backward 梯度正确（与手动 mask 后的 BP 一致）
3. DropPath 整层跳过概率正确
4. training=0 时输出 == 输入（无任何 mask）
5. snake + Dropout: 训练集 vs 验证集 gap 缩小
6. CNN 接入共享模块后，现有 mnist_cnn 训练结果不变（功能等价替换）
7. RNG 确定性：相同 seed 产生相同 mask 序列

---

**参考论文:** Dropout, Srivastava et al. 2014; Stochastic Depth, Huang et al. 2016; Lipschitz-Guided, Nayal et al., arXiv 2509.10298
