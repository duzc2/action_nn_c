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

- [x] **创建 `src/nn/dropout/dropout.h`**
  ```c
  typedef struct {
      float keep_prob;           // 保留概率 (0.0~1.0)
      uint32_t rng_state;        // xorshift32 状态 (零 libc 依赖)
      int training;              // 1=训练模式, 0=推理模式
      uint8_t *mask;             // forward 时缓存的 mask 数组
      size_t mask_capacity;       // mask 缓冲区大小
      int drop_path_flag;        // DropPath: 当前层是否被保留
  } DropoutLayer;

  void dropout_init(DropoutLayer *dp, float keep_prob, uint32_t seed);
  void dropout_free(DropoutLayer *dp);
  void dropout_set_training(DropoutLayer *dp, int training);

  void dropout_forward(float *restrict y, const float *restrict x,
                        size_t n, DropoutLayer *dp);
  void dropout_backward(float *restrict dx, const float *restrict dy,
                         size_t n, const DropoutLayer *dp);
  void droppath_forward(float *restrict y, const float *restrict fx,
                         const float *restrict x, size_t n, DropoutLayer *dp);
  void droppath_backward(float *restrict d_fx, float *restrict dx,
                          const float *restrict dy, size_t n,
                          const DropoutLayer *dp);
  ```

- [x] **创建 `src/nn/dropout/dropout.c`**
  - 使用 xorshift32 PRNG（轻量、零 libc 依赖、确定性）
  - `dropout_forward`: inverted dropout, `mask[i] ? x[i]/keep_prob : 0`
  - `droppath_forward`: 单次 Bernoulli 决定整层通断，缩放同 inverted
  - `dropout_backward`: 根据 forward 时缓存的 mask 精确回传梯度
  - `droppath_backward`: 根据 drop_path_flag 决定梯度流向
  - training=0 或 keep_prob≥1 → 恒等映射；keep_prob≤0 → 全零

- [x] **已集成到 `src/nn/CMakeLists.txt`**（未创建独立 CMakeLists.txt）

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

- [x] **基本单元测试** (`tests/nn/test_dropout.c`) — 18 tests, 189 assertions
  - 生命周期 (init/free/training 切换)
  - inference 模式 passthrough
  - keep_prob 边界 (1.0, 0.0)
  - 统计性质 (1000 samples, keep rate ±5%)
  - 确定性 (同 seed → 同 mask)
  - backward mask 复现验证
  - DropPath 训练行为

- [x] **数值验证测试** (`tests/nn/test_dropout_numerical.c`) — 15 tests, 3,346 assertions
  - **大规模 keep rate**: 9 种 keep_prob (0.1~0.9)，各 1000 elements，验证率 ±2%
  - **DropPath keep rate**: 5 种 prob，各 5000 trials，验证概率准确
  - **无偏估计**: kp=0.5, 50000 trials × 100 elements，验证 E[y]=x
  - **方差理论**: Var(dropout) = x²·(1-p)/p，10000 trials 统计验证
  - **精确反向**: 已知 mask 下 backward 精确对应 (1e-7 精度)
  - **随机输入反向**: 64 dim × 5 seeds，用 dp.mask[i] 直接判断保留/丢弃
  - **DropPath 精确反向**: flag=0/1 手动设置，验证两种路径的精确梯度
  - **前向-反向闭环**: forward→loss→backward，验证链式法则 dx=x/kp²
  - **确定性**: 500 elements 全序列，同 seed → bit-level 一致
  - **不同 seed → 不同 mask**: 10 pairs × 200 elements 统计验证
  - **极端 kp=0.001**: scale=1000，不溢出
  - **极端 kp=0.999**: 验证几乎全保留且缩放正确
  - **大输入 (x=1e6)**: kp=0.3，前向反向后均无 NaN/Inf
  - **RNG 长序列**: 100k draws，无异常
  - **inference/kp=1**: 256 dim，前向反向均为恒等映射

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
