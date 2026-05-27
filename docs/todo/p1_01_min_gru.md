# P1-01: minGRU 序列模型

> 阶段: Phase 1 | 工作量: 中 | 前置依赖: P0-01 (RMSNorm), P0-02 (skip)

---

## 1. 是什么

minGRU 是最简化的 GRU 变体：去掉 reset gate，gate 不依赖 h_{t-1}，因此可以用 parallel scan 并行训练。

```
每时间步:
  z_t   = sigmoid(W_z · x_t + b_z)           // update gate (仅依赖 x_t)
  h̃_t   = W_h · x_t + b_h                    // candidate (无 tanh)
  h_t   = (1 - z_t) · h_{t-1} + z_t · h̃_t    // 标准 GRU 循环

并行训练:
  a_t = 1 - z_t,   b_t = z_t · h̃_t
  v_t = a_t · v_{t-1} + b_t    ← parallel prefix scan
```

相比 LSTM：参数少 62-85%，训练快 175-1361x（ICLR 2025）。

## 2. 为什么需要

- 当前 Elman RNN 无法处理长序列依赖
- cnn_rnn_react、cs 的行为记忆能力受限
- minGRU 的 C 实现极简（element-wise + scan），比 LSTM 干净很多
- parallel scan 可在任意 CPU 上高效执行（无需 GPU）

## 3. 详细子任务

### 3.1 核心算法实现

- [ ] **创建 `src/nn/types/rnn/rnn_min_gru.h`**
  ```c
  typedef struct {
      int input_dim;
      int hidden_dim;
      float *W_z;      // [hidden_dim × input_dim]
      float *b_z;      // [hidden_dim]
      float *W_h;      // [hidden_dim × input_dim]
      float *b_h;      // [hidden_dim]
      // 可选: 多层堆叠或 skip
  } MinGRUConfig;

  // 推理: 逐时间步 (serial, 简单)
  void mingru_infer(float *h_out, const float *x_seq,
                     int seq_len, const MinGRUConfig *cfg, float *h_init);

  // 训练: parallel scan (log-space 数值稳定)
  void mingru_train_forward(float *h_seq, const float *x_seq,
                              int seq_len, const MinGRUConfig *cfg);
  void mingru_train_backward(float *dW_z, float *db_z, float *dW_h,
                               float *db_h, const float *x_seq,
                               const float *h_seq, const float *dh_seq,
                               int seq_len, const MinGRUConfig *cfg);
  ```

- [ ] **创建 `src/nn/types/rnn/rnn_min_gru.c`**
  - `mingru_infer`: 简单的 for 循环逐时间步
  - `mingru_train_forward`:
    1. 对所有时间步并行计算 `z_t`, `h̃_t`
    2. `a_t = 1 - z_t`, `b_t = z_t * h̃_t`
    3. parallel scan over `(a_t, b_t)` 得到 `h_t` 序列
  - `mingru_train_backward`: scan 的反向自动微分 (可复用 scan 原语)

- [ ] **实现 parallel scan 原语** (可复用于 Mamba-2)
  ```c
  // 关联操作符: (A₂,B₂) ⊕ (A₁,B₁) = (A₂·A₁, A₂·B₁+B₂)
  // 仅需 element-wise 乘法和加法，适合在 C 中实现
  void parallel_scan(float *a, float *b, int n);  // in-place
  ```

- [ ] **log-space 数值稳定实现**
  - 当 a_t 接近 0 或 1 时，线性 scan 可能溢出
  - 实现 `log_parallel_scan` 版本，全部操作在 log 空间

### 3.2 注册接入

- [ ] **创建 `src/nn/types/rnn/nn_type_rnn_min_gru_infer.c`** — 注册为 `"rnn_min_gru"`
- [ ] **创建 `src/nn/types/rnn/nn_type_rnn_min_gru_train.c`** — 训练端注册
- [ ] **更新 `src/nn/types/rnn/CMakeLists.txt`** — 添加编译目标
- [ ] **更新 `src/nn/CMakeLists.txt`** — `ACTION_C_ENABLE_NN_RNN_MIN_GRU` 开关

### 3.3 与现有 RNN 的关系

- [ ] **独立注册 vs 扩展**
  - 方案 A: 注册为新类型 `"rnn_min_gru"`（推荐，清晰独立）
  - 方案 B: 作为现有 RNN 的 `activation_type` 扩展
  - 选择方案 A，保留 Elman RNN 不变

### 3.4 测试

- [ ] **单元测试** — forward 与 PyTorch 参考实现对比
  - test: 小序列(seq=8, d_in=4, d_hid=3)，对比最后一帧 h_T 误差 < 1e-5
- [ ] **单元测试** — parallel scan 正确性
  - 对比 serial scan 和 parallel scan 的输出，每时间步误差 < 1e-6
- [ ] **单元测试** — backward 梯度正确
  - W_z, b_z, W_h, b_h 梯度与数值梯度比较 < 1e-4
- [ ] **集成测试** — 替换 cnn_rnn_react 的 Elman RNN 为 minGRU
  - 相同参数规模，训练后行为记忆能力更强
- [ ] **基准测试** — 长序列(seq=1024)下 serial vs parallel 速度比

## 4. API 设计

```c
// rnn_min_gru.h

// 初始化 minGRU 的参数
void mingru_init(MinGRUConfig *cfg, int input_dim, int hidden_dim);

// 推理模式: 逐时间步 (MCU 友好, 仅需 O(hidden_dim) 额外内存)
int mingru_step(float *h, const float *x, const MinGRUConfig *cfg);

// 序列推理: 自动调用 mingru_step
void mingru_infer_sequence(float *h_final, const float *x_seq,
                            int seq_len, const MinGRUConfig *cfg,
                            const float *h_init);

// 训练前向: parallel scan, O(log T) 深度
void mingru_train_forward(float *h_seq, const float *x_seq,
                           int seq_len, const MinGRUConfig *cfg);

// 训练反向
void mingru_train_backward(
    float *dW_z, float *db_z, float *dW_h, float *db_h,
    float *dx_seq,
    const float *x_seq, const float *h_seq, const float *dh_seq,
    int seq_len, const MinGRUConfig *cfg);
```

## 5. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `src/nn/types/rnn/rnn_min_gru.h` |
| **新建** | `src/nn/types/rnn/rnn_min_gru.c` |
| **新建** | `src/nn/types/rnn/nn_type_rnn_min_gru_infer.c` |
| **新建** | `src/nn/types/rnn/nn_type_rnn_min_gru_train.c` |
| **修改** | `src/nn/types/rnn/CMakeLists.txt` |
| **修改** | `src/nn/CMakeLists.txt` (新增 ACTION_C_ENABLE_NN_RNN_MIN_GRU) |

## 6. 验收标准

1. forward (serial scan) 与 PyTorch 参考实现一致 (< 1e-5)
2. parallel scan 与 serial scan 输出一致 (< 1e-6)
3. backward 梯度正确 (与数值梯度比较 < 1e-4)
4. 长序列 (seq=1024) 不产生 NaN
5. cnn_rnn_react 用 minGRU 替换后行为记忆显著优于 Elman RNN

---

**参考论文:** minGRU/minLSTM, Feng et al., ICLR 2025 (arXiv:2410.01201)
