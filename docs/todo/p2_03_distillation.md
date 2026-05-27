# P2-03: 知识蒸馏框架

> 阶段: Phase 2 | 工作量: 中 | 前置依赖: 无 (训练端独立模块, 可提前启动)

---

## 1. 是什么

将当前每个 demo 手工实现的 teacher-student 蒸馏逻辑系统化为统一的蒸馏损失函数模块。提供 MSE 蒸馏、KL 散度蒸馏、特征层蒸馏三种损失，以及组合损失（任务损失 + 蒸馏损失）。

```
基础概念:
  teacher_model: 已训练好的大模型（输出 logits 或中间特征）
  student_model: 正在训练的小模型

  蒸馏损失 L_distill = D(student_logits, teacher_logits)
  总损失   L_total   = α · L_task + (1-α) · L_distill
```

## 2. 为什么需要

- 项目核心范式就是"大模型做策略，小网络做执行"——每个 demo 都涉及 teacher-student
- 目前 snake、nested_nav、road_graph_nav 等 demo 各自手工实现了蒸馏逻辑，代码重复
- 系统化后统一接口，新 demo 只需配置即可启用蒸馏，减少重复代码
- 2025 年蒸馏研究（EA-KD 自适应加权、DTS 动态温度、多教师蒸馏）提供了更丰富的能力

## 3. 详细子任务

### 3.1 蒸馏损失函数 (`src/train/distill.h` + `distill.c`)

- [ ] **MSE 蒸馏损失** — 最简单的 logit matching
  ```c
  // L = (1/N) · Σ_i (s_i - t_i)²
  float kd_mse_loss(const float *student_logits, const float *teacher_logits, int n);
  // 返回 loss 值, dloss/dstudent 写入 dstudent (调用方提供)
  void kd_mse_loss_backward(float *dstudent, const float *student_logits,
                             const float *teacher_logits, int n);
  ```

- [ ] **KL 散度蒸馏损失**
  ```c
  // L = T² · Σ_i softmax(t_i/T) · log(softmax(t_i/T) / softmax(s_i/T))
  // T 是温度参数 (默认 3.0-5.0)
  float kd_kl_loss(const float *student_logits, const float *teacher_logits,
                   int n, float temperature);
  void kd_kl_loss_backward(float *dstudent, const float *student_logits,
                            const float *teacher_logits, int n, float temperature);
  ```

- [ ] **特征层蒸馏损失**
  ```c
  // L = ||student_feat - teacher_feat||²  (或 cosine distance)
  float kd_cosine_loss(const float *student_feat, const float *teacher_feat, int n);
  void kd_cosine_loss_backward(float *dstudent_feat, const float *student_feat,
                                const float *teacher_feat, int n);
  ```

- [ ] **组合损失**
  ```c
  // L_total = α·CrossEntropy(logits, y_true) + (1-α)·KL(logits/T, teacher_logits/T)·T²
  float kd_combined_loss(float *dlogits,
                          const float *student_logits,
                          const float *teacher_logits,
                          const float *y_true,       // one-hot 标签
                          int n, int n_classes,
                          float alpha,               // 任务损失权重 (默认 0.5)
                          float temperature);        // KD 温度 (默认 4.0)
  ```

### 3.2 自适应增强（可选, 基于 2025 论文）

- [ ] **动态温度 — DTS**
  ```c
  // temperature = base_temp + (init_temp - base_temp) * exp(-decay * step)
  float kd_dynamic_temperature(float base_temp, float init_temp, float decay, int step);
  ```

- [ ] **熵自适应加权 — EA-KD** (低优先级)
  ```c
  // 基于 teacher 对样本的置信度动态调整蒸馏权重
  // 高置信度样本: 蒸馏权重↓; 低置信度样本: 蒸馏权重↑
  ```

### 3.3 多教师蒸馏接口 (预留)

- [ ] **多教师加权蒸馏**
  ```c
  // L = Σ_k w_k · L_distill(student, teacher_k)
  // 权重 w_k 可固定或基于验证集性能动态调整
  float kd_multi_teacher_loss(float *dlogits,
                               const float *student_logits,
                               const float **teacher_logits,  // [num_teachers × n_outputs]
                               const float *teacher_weights,   // [num_teachers]
                               int n, int num_teachers,
                               float temperature);
  ```

### 3.4 Profiler 侧支持

- [ ] **profiler 中增加 teacher 配置**
  ```c
  // 网络定义中声明:
  // teacher_model_path: "build/snake_teacher/model.bin"
  // distill_config { type: KL, temperature: 4.0, alpha: 0.5 }
  ```
- [ ] **自动生成 teacher forward + student loss 组合代码**
  - teacher 权重在训练代码中只读（不参与反向传播）
  - student 的 forward 和 teacher 的 forward 在训练图中并列

### 3.5 测试

- [ ] **单元测试** — MSE 蒸馏: 两个相同向量 loss=0, 不同向量 loss>0
- [ ] **单元测试** — KL 蒸馏: T=1 时与 CrossEntropy 一致, T→∞ 时梯度趋于 0
- [ ] **单元测试** — 组合损失: α=0 时 L = T²·KL, α=1 时 L = CrossEntropy
- [ ] **集成测试** — snake: 蒸馏训练 vs 无蒸馏训练, 蒸馏组收敛更快
- [ ] **集成测试** — nested_nav: 蒸馏训练 vs 直接训练, 蒸馏组平均得分更高

## 4. API 设计

```c
// distill.h

typedef enum {
    KD_LOSS_MSE = 0,
    KD_LOSS_KL,
    KD_LOSS_COSINE,        // 特征层余弦距离
    KD_LOSS_COMBINED,      // CrossEntropy + KL
    KD_LOSS_MULTI_TEACHER, // 多教师加权
} KdLossType;

typedef struct {
    KdLossType type;
    float temperature;     // KL/COMBINED 的温度 (默认 4.0)
    float alpha;           // COMBINED 的任务损失权重 (默认 0.5)
    float init_temp;       // 动态温度的初始温度 (0 = 不使用)
    float base_temp;       // 动态温度的最终温度 (默认 1.0)
    float temp_decay;      // 动态温度的衰减率 (默认 0.001)
} DistillConfig;

void distill_config_default(DistillConfig *cfg);
// 设置: type=KL, temperature=4.0, alpha=0.5, 动态温度关闭

// 核心 API
float distill_loss(float *dstudent_logits,
                   const float *student_logits,
                   const float *teacher_logits,
                   const float *y_true,          // NULL 时仅计算蒸馏损失
                   int n, int n_classes,
                   const DistillConfig *cfg);

// 特征蒸馏 API
float distill_feature_loss(float *dstudent_feat,
                            const float *student_feat,
                            const float *teacher_feat,
                            int n);

// 动态温度更新 (每步调用)
void distill_update_temperature(DistillConfig *cfg, int step);

// 多教师接口
float distill_multi_teacher_loss(
    float *dstudent_logits,
    const float *student_logits,
    const float *const *teacher_logits,  // [num_teachers × n]
    const float *teacher_weights,        // [num_teachers]
    int n, int num_teachers,
    const DistillConfig *cfg);
```

## 5. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `src/train/distill.h` |
| **新建** | `src/train/distill.c` |
| **修改** | `src/train/CMakeLists.txt` |
| **修改** | `src/profiler/` — 增加 teacher 配置项和代码生成逻辑 |
| — | 各 demo 的蒸馏逻辑逐步迁移到统一接口 (非一次性) |

## 6. 验收标准

1. MSE 蒸馏: student == teacher 时 `loss == 0`
2. KL 蒸馏: T=1 且 student==teacher 时 `KL == 0`；T→∞ 时梯度趋近于 0
3. 组合损失: `α=1.0` 时 `L == CrossEntropy`；`α=0.0` 时 `L == T²·KL`
4. 动态温度: 随 step 增加 t 从 init_temp 单调递减到 base_temp
5. snake 蒸馏训练: loss 收敛速度 ≥ 直接训练
6. nested_nav 蒸馏训练: 平均得分 ≥ 直接训练

---

**参考论文:** Hinton et al., Distilling the Knowledge in a Neural Network (2015); EA-KD, ICCV 2025; DTS, IEEE 2026; LRC, arXiv 2025; MoD, IJCAI 2025
