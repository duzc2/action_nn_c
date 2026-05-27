# P1-03: SwiGLU 激活函数

> 阶段: Phase 1 | 工作量: 小 | 前置依赖: sigmoid (已有) | 几乎零依赖, 最早可完工

---

## 1. 是什么

SwiGLU 是一种门控激活，由 Swish 门 + 线性值投影 + 逐元素乘法组成。

```
Swish(x) = x · sigmoid(x)

SwiGLU(x):
  gate   = Swish(W_gate · x)       // W_gate 是可学习权重
  value  = V · x                   // V 是单独的值投影权重
  hidden = gate ⊙ value           // ⊙ = 逐元素乘
  output = W_out · hidden
```

通过 2/3 规则，总参数量与标准 GELU/ReLU MLP 相同：
```
hidden_dim_swiglu = int(2/3 * hidden_dim_standard)
// 3 个权重矩阵 × 2/3 dim ≈ 2 个权重矩阵 × 1 dim
```

SwiGLU 是 Llama/Mistral/DeepSeek/Palm/Gemini 的标配。

## 2. 为什么需要

- 现有 MLP 激活都是逐点函数 (ReLU/Sigmoid/Tanh/LeakyReLU)
- SwiGLU 的门控机制表达力更强，同样的参数量获得更好的训练结果
- 所有使用 MLP 的 demo 都能受益（snake, nested_nav, weather, move, target, sevenseg, hybrid_route, mnist = 8 个 demo）
- sigmoid 已有，Swish 实现仅需 1 行：`x * sigmoid(x)`

## 3. 详细子任务

### 3.1 Swish 激活实现

- [ ] **`src/nn/types/mlp/mlp_layers.h` — 增加 swish 实现**
  ```c
  static inline float swish(float x) {
      return x * sigmoid(x);    // sigmoid 已有
  }
  static inline float swish_derivative(float x) {
      float sx = sigmoid(x);
      return sx + x * sx * (1.0f - sx);  // d(x·σ(x))/dx
  }
  ```

- [ ] **`mlp_layers.c` — 扩展前向传播**
  ```c
  // 当前: MLP activation 有 NONE/RELU/SIGMOID/TANH/SOFTMAX/LEAKY_RELU
  // 新增: SWIGLU
  ```

### 3.2 SwiGLU FFN 实现

- [ ] **`mlp_config.h` — 增加 SWIGLU 枚举值**
  ```c
  typedef enum {
      MLP_ACTIVATION_NONE = 0,
      MLP_ACTIVATION_RELU,
      MLP_ACTIVATION_SIGMOID,
      MLP_ACTIVATION_TANH,
      MLP_ACTIVATION_SOFTMAX,
      MLP_ACTIVATION_LEAKY_RELU,
      MLP_ACTIVATION_SWIGLU,     // 新增
  } MlpActivationType;
  ```

- [ ] **`mlp_layers.h` — SwiGLU 层需要 3 个权重矩阵**
  ```c
  typedef struct {
      // ... 现有字段 ...
      // 当 activation == SWIGLU 时:
      float *W_gate;    // [hidden_dim × input_dim]  gate 投影
      float *V;        // [hidden_dim × input_dim]  value 投影
      float *W_out;    // [output_dim × hidden_dim]  输出投影
      // 注意: 此处的 hidden_dim = int(2/3 * normal_hidden_dim)
  } MlpDenseLayer;
  ```

- [ ] **`mlp_infer_ops.c` — SwiGLU forward**
  ```c
  if (layer->activation == MLP_ACTIVATION_SWIGLU) {
      // gate[i] = swish(Σ_j W_gate[i,j] * x[j])
      // value[i] = Σ_j V[i,j] * x[j]
      // hidden[i] = gate[i] * value[i]
      // output[k] = Σ_i W_out[k,i] * hidden[i]
  }
  ```

- [ ] **`mlp_train_ops.c` — SwiGLU backward**
  - 三个权重矩阵各自的梯度
  - swish 导数的链式法则

### 3.3 Transformer FFN 接入

- [ ] **`transformer_config.h` — FFN type 增加 `ffn_swiglu`**
- [ ] **`transformer_forward.c` — 替换 FFN 为 SwiGLU**
  - 复用 MLP 的 SwiGLU 前向逻辑
- [ ] **`transformer_train_ops.c` — SwiGLU backward**

### 3.4 2/3 规则

- [ ] **profiler 或 init 函数中实现 2/3 规则**
  ```c
  int swiglu_hidden_dim(int standard_hidden_dim) {
      int d = (2 * standard_hidden_dim) / 3;
      d = (d + 7) & ~7;  // 上取整到 8 的倍数 (缓存行对齐)
      return d;
  }
  ```

### 3.5 测试

- [ ] **单元测试** — swish(x) 输出与参考实现一致
- [ ] **单元测试** — SwiGLU 前向传播与 PyTorch `F.silu` + 手动矩阵乘一致
- [ ] **单元测试** — SwiGLU 反向传播梯度正确
- [ ] **集成测试** — snake: ReLU vs SwiGLU，相同参数量，SwiGLU 平均得分更高
- [ ] **集成测试** — mnist: GELU MLP vs SwiGLU MLP，SwiGLU 验证集准确率更高

## 4. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **修改** | `src/nn/types/mlp/mlp_config.h` |
| **修改** | `src/nn/types/mlp/mlp_layers.h` |
| **修改** | `src/nn/types/mlp/mlp_layers.c` |
| **修改** | `src/nn/types/mlp/mlp_infer_ops.c` |
| **修改** | `src/nn/types/mlp/mlp_train_ops.c` |
| **修改** | `src/nn/types/transformer/transformer_config.h` |
| **修改** | `src/nn/types/transformer/transformer_forward.c` |
| **修改** | `src/nn/types/transformer/transformer_train_ops.c` |

## 5. 验收标准

1. swish(x) 输出正确 (与 PyTorch `F.silu` 一致, <1e-6)
2. SwiGLU forward 与参考实现一致 (< 1e-5)
3. SwiGLU backward 梯度正确 (< 1e-4)
4. snake: SwiGLU 平均得分 ≥ ReLU (相同总参数量)
5. mnist: SwiGLU 验证集准确率 ≥ GELU
6. 不使用 SwiGLU 时零开销（代码路径不进入 SwiGLU 分支）

---

**参考论文:** SwiGLU, Shazeer 2020; GLU Variants, Dauphin et al. 2017
