# P2-01: GNN 边特征 + Edge-Conditioned GATv2

> 阶段: Phase 2 | 工作量: 中 | 前置依赖: P0-01 (RMSNorm), P1-02 (attention 模块)

---

## 1. 是什么

在现有 GNN 基础上增加三个能力：
1. **边特征支持** — 邻接关系附加大于 0/1 的权重信息（如距离、容量）
2. **SUM/MAX 聚合器** — 替代仅有 MEAN 的单一选择
3. **Edge-Conditioned GATv2** — 当聚合器为 ATTENTION 时，注意力计算纳入边特征

## 2. 为什么需要

- road_graph_nav 仅用邻接关系 (0/1)，road 的距离、权重完全忽略
- 边特征对路径规划质量有直接提升
- EC-GATv2 是 2025 年 GNN 注意力标准 (三方注意力: 源+目标+边)

## 3. 详细子任务

### 3.1 GNN 配置扩展

- [ ] **`gnn_config.h` — 新增配置**
  ```c
  typedef enum {
      GNN_AGG_MEAN = 0,
      GNN_AGG_SUM,
      GNN_AGG_MAX,
      GNN_AGG_ATTENTION,   // 新增
  } GnnAggregatorType;

  typedef struct {
      // ... 现有字段 ...
      int edge_dim;                    // 边特征维度 (0=无边特征, 向后兼容)
      GnnAggregatorType aggregator;    // 默认 GNN_AGG_MEAN
      // 当 aggregator==ATTENTION 时:
      float *We;   // 边投影矩阵 [head_dim × edge_dim] (多头的平铺)
      float *a;     // 注意力向量 [3 * head_dim]
  } GnnConfig;
  ```

### 3.2 新增聚合器实现

- [ ] **SUM 聚合器** (`gnn_infer_ops.c`)
  ```c
  h_i' = Σ_{j∈N(i)} W·h_j      // 不做邻居数归一化
  ```
- [ ] **MAX 聚合器**
  ```c
  h_i' = max_{j∈N(i)} (W·h_j)   // 逐元素取 max
  ```
- [ ] **ATTENTION 聚合器 (无边缘)**
  ```c
  α_ij = softmax_j(LeakyReLU(a^T·[W·h_i ∥ W·h_j]))
  h_i' = Σ α_ij · W·h_j
  ```

### 3.3 Edge-Conditioned GATv2

- [ ] **有边特征时的注意力**
  ```c
  // 对每个邻居 j:
  h_i_hat = W * h_i          // 源节点投影
  h_j_hat = W * h_j          // 目标节点投影
  e_hat   = We * e_ij        // 边特征投影 ← 核心创新
  score   = LeakyReLU(a^T * concat(h_i_hat, h_j_hat, e_hat))
  α_ij    = softmax_j(score)
  h_i'    = Σ α_ij * h_j_hat + h_i  // + 残差连接
  ```

- [ ] **实现方式：**
  - `edge_dim=0` 且 `aggregator=ATTENTION` → 标准 GAT (无边的三方拼接用 [h_i ∥ h_j])
  - `edge_dim>0` 且 `aggregator=ATTENTION` → EC-GATv2
  - 两步 `forward`: 先对所有节点对计算线性投影 (1 次矩阵乘)，再逐邻居聚合

- [ ] **`gnn_train_ops.c` — EC-GATv2 反向传播**
  - We, a, W 各自的梯度
  - 注意：LeakyReLU 反向、softmax 的反向、concat 的反向拆解

### 3.4 测试

- [ ] **单元测试** — SUM/MAX/ATTENTION 聚合器分别与参考实现对比
- [ ] **单元测试** — EC-GATv2: 固定 We 为单位阵，验证边特征流入注意力
- [ ] **集成测试** — road_graph_nav: 边特征使用 weight 后路径规划质量提升
- [ ] **向后兼容测试** — edge_dim=0, aggregator=MEAN 时行为不变

## 4. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **修改** | `src/nn/types/gnn/gnn_config.h` |
| **修改** | `src/nn/types/gnn/gnn_infer_ops.c` |
| **修改** | `src/nn/types/gnn/gnn_train_ops.c` |
| 无新建文件（现有 GNN 目录内扩展） |

## 5. 验收标准

1. SUM aggregator forward 与手动 Σ 一致
2. MAX aggregator forward 与逐元素 max 一致
3. ATTENTION (edge_dim=0) 输出与 PyTorch GATv2 一致 (< 1e-5)
4. EC-GATv2 边特征在前向中有非零贡献
5. backward 梯度正确
6. edge_dim=0 且 aggregator=MEAN 时输出与旧版完全一致

---

**参考论文:** GATv2, Brody et al. 2022; EC-GATv2, Zhang et al. 2025
