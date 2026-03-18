# 任务清单（当前版本）

## 已完成

- 图拓扑配置引入并替代旧模型分类配置
- `network_graph_build / validate / toposort / build_spec` 实现
- 训练与推理接口统一要求 `NetworkSpec`
- profiler 改造为图配置驱动
- demo 与主要测试迁移到新流程

## 进行中

- `SELECT/MERGE` 执行语义细化
- 异构节点（CNN/RNN/KNN）执行器专有实现

## 待办

- 增加复杂分支拓扑示例
- 增加拓扑维度冲突的负测试
- 增加文档示例配置片段自动校验
