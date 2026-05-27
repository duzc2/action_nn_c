# P2-04: INT8 后训练量化

> 阶段: Phase 2 | 工作量: 中 | 前置依赖: 无 (独立推理端模块, 可立即启动)

---

## 1. 是什么

在训练完成后（桌面 FP32 训练），对权重进行 INT8 量化后生成独立的量化推理二进制。与现有 FP32 推理管道完全独立——FP32 推理保留不变，INT8 推理作为可选部署路径。

```
量化流程:
  FP32 训练 → 校准(收集激活范围) → 量化权重 → 生成 INT8 .c/.h → 编译 MCU 推理二进制

量化粒度: 逐通道 (Per-Channel)
  每个输出通道独立拥有 scale 和 zero_point → 精度显著优于逐张量量化
量化格式: W8A8 (权重 INT8 + 激活 INT8)
```

## 2. 为什么需要

- 推理二进制 (`src/infer/`) 设计为独立编译部署，可部署到 MCU 级别硬件
- 训练在桌面 CPU 完成 FP32，推理端内存/flash 受限时 INT8 将模型缩小 4×
- 逐通道量化是 2025 年工业标准（Google/Qualcomm/ARM NN），精度损失仅 1-3%
- 与编译分离架构天然契合：训练用 FP32 全精度，部署用量化 INT8
- 只做后训练量化 (PTQ)，不做量化感知训练 (QAT) — 当前 demo 规模下 PTQ 足矣

## 3. 详细子任务

### 3.1 校准模块 (`src/infer/quantize.h` + `quantize.c`)

- [ ] **校准数据结构**
  ```c
  typedef struct {
      float min_val;
      float max_val;
      float scale;       // (max - min) / 255
      float zero_point;  // round(-min / scale)
  } QuantRange;

  typedef struct {
      int n_layers;
      QuantRange *weight_ranges;   // 每层权重范围 (逐通道: [n_layers × out_channels])
      QuantRange *input_ranges;    // 每层输入激活范围
      int *per_channel_dims;       // 每层权重输出通道数 (0 = 逐张量)
  } QuantCalibration;
  ```

- [ ] **校准数据收集**
  ```c
  // 跑少量代表性样本, 收集每层激活的 min/max
  // 默认使用 KL 散度校准 (比 min-max 更精确)：
  //   1. 收集激活的直方图 (2048 bins)
  //   2. 尝试不同阈值, 选择使量化前后分布 KL 散度最小的阈值
  void quant_calibrate_activation(const float *activations, int n,
                                   int num_bins, QuantRange *range);
  ```

- [ ] **权重范围计算**
  ```c
  // 逐通道: 对每个输出通道独立计算 min/max
  void quant_calibrate_weight_per_channel(const float *weights,
                                           int out_channels, int in_size,
                                           QuantRange *ranges); // [out_channels] 个
  // 逐张量: 整个权重矩阵统一 min/max
  void quant_calibrate_weight_per_tensor(const float *weights,
                                          int n, QuantRange *range);
  ```

### 3.2 量化/反量化

- [ ] **量化函数**
  ```c
  // 将 FP32 权重/激活量化为 INT8
  // q = clamp(round(x / scale) + zero_point, -128, 127)
  void quantize_to_int8(const float *x, int8_t *qx, int n,
                         float scale, float zero_point);

  // 反量化: x_fp = (q - zero_point) * scale
  void dequantize_from_int8(const int8_t *qx, float *x, int n,
                             float scale, float zero_point);
  ```

- [ ] **逐通道量化**
  ```c
  // 权重矩阵 W[out_channels × in_features]
  // 每个输出通道独立量化 → [out_channels] 个 scale
  void quantize_weight_per_channel(const float *W,
                                    int8_t *qW,           // 量化后权重
                                    float *scales,        // [out_channels]
                                    int out_channels, int in_features);
  ```

### 3.3 各网络类型的 INT8 推理

- [ ] **INT8 矩阵乘核心**
  ```c
  // INT8 权重 × INT8 输入 → INT32 accumulator → FP32 output
  // output[k] = scale_w[k] * scale_x * Σ_j (qW[k,j] * qx[j])
  //            - scale_w[k] * scale_x * zp_x * Σ_j qW[k,j]
  //            - scale_w[k] * scale_x * zp_w[k] * Σ_j qx[j]
  //            + scale_w[k] * scale_x * zp_w[k] * zp_x * n
  void matmul_int8(float *output,          // [out_dim]
                   const int8_t *qW,       // [out_dim × in_dim] INT8 权重
                   const int8_t *qx,       // [in_dim] INT8 输入
                   const float *scale_w,    // [out_dim] 逐通道 scale
                   float scale_x,
                   int out_dim, int in_dim);
  ```

- [ ] **MLP INT8 推理** — `mlp_infer_ops.c` 新增 `mlp_infer_int8()`
- [ ] **CNN INT8 推理** — `cnn_infer_ops.c` 新增 `cnn_infer_int8()`
  - conv2d 的 INT8 实现: im2col → INT8 matmul
  - depthwise conv 的 INT8 实现
  - BN 的 fold + quantize
- [ ] **Transformer INT8 推理** — `transformer_infer_ops.c` 新增
- [ ] **RNN INT8 推理** — `rnn_infer_ops.c` 新增
- [ ] **GNN INT8 推理** — `gnn_infer_ops.c` 新增

### 3.4 代码生成支持

- [ ] **Profiler 侧新增量化步骤**
  ```
  generate → train → calibrate → quantize → build(int8_infer)
  ```
- [ ] **量化后的权重以 `int8_t` 数组生成 `.c/.h`**
  ```c
  // 生成文件示例: build/model_quant_weights.c
  static const int8_t layer0_weight[128 * 64] = { ... };
  static const float layer0_scale[128] = { ... };
  ```
- [ ] **编译开关** — `ACTION_C_ENABLE_INFER_QUANTIZE=ON/OFF` 控制是否编译量化推理路径

### 3.5 支持格式

- [ ] **W8A8** (权重 INT8 + 激活 INT8) — 主推格式, 4× 压缩
- [ ] **W8A32** (权重 INT8 + 激活 FP32) — 可选, 精度无损, 适合有 FPU 的 MCU

### 3.6 测试

- [ ] **单元测试** — 量化+反量化: `dequant(quant(x)) ≈ x` 误差 < 1e-2 (INT8 理论精度)
- [ ] **单元测试** — 逐通道量化 vs 逐张量: 前者精度更高
- [ ] **单元测试** — INT8 矩阵乘 vs FP32: 相对误差 < 1%
- [ ] **集成测试** — mnist: INT8 推理 vs FP32 推理准确率差异 < 2%
- [ ] **集成测试** — snake: INT8 推理得分 vs FP32 推理得分差异忽略不计
- [ ] **内存测试** — INT8 模型文件大小 ≈ FP32 模型文件大小的 1/4 (不含 scale 表)

## 4. API 设计

```c
// quantize.h

// ==========================================
// 校准 (训练后运行)
// ==========================================

typedef struct {
    float min_val, max_val;
    float scale, zero_point;
} QuantRange;

typedef struct {
    QuantRange *weight_ranges;     // 每层权重量化范围
    QuantRange *input_ranges;      // 每层输入激活范围
    int *per_channel_dims;         // 每层输出通道数 (0=逐张量)
    int n_layers;
} QuantCalibration;

void quant_calibration_init(QuantCalibration *calib, int n_layers);
void quant_calibration_free(QuantCalibration *calib);

// 使用 KL 散度校准激活范围 (推荐)
void quant_calibrate_activation(const float *activations, int n,
                                 int num_bins, QuantRange *range);

// 权重校准
void quant_calibrate_weight_per_channel(const float *W,
                                         int out_channels, int in_size,
                                         QuantRange *ranges);
void quant_calibrate_weight_per_tensor(const float *W, int n,
                                        QuantRange *range);

// ==========================================
// 量化/反量化
// ==========================================
void quantize_to_int8(const float *x, int8_t *qx, int n,
                       float scale, float zero_point);
void dequantize_from_int8(const int8_t *qx, float *x, int n,
                           float scale, float zero_point);

// ==========================================
// INT8 推理 (与 FP32 推理平行的独立路径)
// ==========================================

// INT8 矩阵乘 (核心算子)
void matmul_int8(float *output,
                 const int8_t *qW, const int8_t *qx,
                 const float *scale_w, float scale_x,
                 int out_dim, int in_dim);

// INT8 卷积 (im2col + matmul)
void conv2d_int8(float *output,
                 const float *input,     // FP32 输入 (激活先量化)
                 const int8_t *qW,       // INT8 卷积核
                 const float *scale_w,    // 逐通道 scale
                 int H, int W, int C_in,
                 int C_out, int K, int stride, int pad);
```

## 5. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `src/infer/quantize.h` |
| **新建** | `src/infer/quantize.c` |
| **修改** | `src/infer/CMakeLists.txt` — 增加 QUANTIZE 编译开关 |
| **修改** | `src/nn/types/mlp/mlp_infer_ops.c` — 新增 INT8 推理 |
| **修改** | `src/nn/types/cnn/cnn_infer_ops.c` — 新增 INT8 推理 |
| **修改** | `src/nn/types/transformer/transformer_infer_ops.c` |
| **修改** | `src/nn/types/rnn/rnn_infer_ops.c` |
| **修改** | `src/nn/types/gnn/gnn_infer_ops.c` |
| **修改** | `src/profiler/` — 增加 calibrate + quantize + build(int8) 步骤 |

## 6. 验收标准

1. 量化+反量化误差 ≤ INT8 理论精度 (1/256 ≈ 0.4%)
2. 逐通道量化精度 ≥ 逐张量量化（CNN depthwise 层尤其明显）
3. mnist: INT8 推理准确率 vs FP32 ≤ 2% 差异
4. INT8 模型文件(含 scale 表) ≤ FP32 模型文件的 35%
5. snake/small_mlp: INT8 推理结果与 FP32 肉眼不可区分
6. `ACTION_C_ENABLE_INFER_QUANTIZE=OFF` 时零编译开销 (量化代码完全排除)

---

**参考论文:** Jacob et al., Quantization and Training of Neural Networks (2018); Per-Channel Quantization, Krishnamoorthi 2018; QServe, arXiv 2025; DartQuant, arXiv 2025
