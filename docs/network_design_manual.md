# 网络设计手册（图拓扑版）

## 1. 设计原则

- 网络是**有向拓扑图**，不是固定层序模板
- 图是单一事实来源，spec 由图推导
- 同一输入可扇出到多个子网络，允许重叠输入选择
- 训练/推理共享同一 spec，不存在双轨接口

## 2. 数据结构

核心结构定义于 `src/include/network_spec.h`：

- `NetworkGraphNode`
- `NetworkGraphEdge`
- `NetworkGraph`
- `NetworkSpec`

图节点包含：

- `id`
- `type`
- `selector_offset`
- `selector_size`

## 3. 支持的语义节点类型

- 输入与路由：`INPUT`、`SELECT`、`MERGE`
- 计算：`LINEAR`、`TRANSFORMER_BLOCK`、`ATTENTION_HEAD`、`CNN`、`RNN`、`KNN`

## 4. 从图到规格

流程：

1. `network_graph_build`
2. `network_graph_validate`
3. 拓扑排序
4. `network_spec_build_from_graph`

校验项包含：

- 节点 id 唯一
- 边引用节点存在
- 输入/输出节点存在
- 图无环
- `SELECT` 节点参数合法

## 5. 执行语义

- 训练和推理按 `NetworkSpec.layers` 顺序执行
- 当前后端对 `ATTENTION_HEAD/CNN/RNN/KNN` 已接入统一分发路径
- 复杂路由（如重叠 SELECT + MERGE）的维度传播可在执行器继续细化

## 6. profiler 的职责

- 读取 `config_user.h` 图配置
- 估算参数量、峰值激活、FLOPs
- 生成 `network_def.h`
- 输出 `network_def_build_spec`，供运行时直接构建 spec

## 7. 推荐配置模式

- 先做最小可运行图：`INPUT -> LINEAR -> OUTPUT`
- 再增加语义节点（RNN/CNN/Attention）
- 每次变更后执行：
  - 重新生成 `network_def.h`
  - 重新构建
  - 运行测试与 demo
