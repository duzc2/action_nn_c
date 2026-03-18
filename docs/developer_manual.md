# 开发手册

## 1. 架构总览

项目分为三层：

- 配置层：`config_user.h` 提供图拓扑配置与 IO 定义
- 规格层：`network_spec.*` 完成图校验、拓扑排序、规格构建
- 执行层：`workflow_train.*` 与 `workflow_runtime.*` 消费 `NetworkSpec`

## 2. 关键模块

- `src/core/network_spec.c`  
  提供 `network_graph_build`、`network_graph_validate`、`network_spec_build_from_graph`
- `src/tools/profiler.c`  
  读取配置，生成 `network_def.h`，并输出参数/激活/FLOPs 估算
- `src/train/workflow_train.c`  
  训练与权重导出
- `src/infer/workflow_runtime.c`  
  二进制权重加载与在线推理

## 3. 统一调用规范

- 训练：`WorkflowTrainOptions.network_spec` / `WorkflowTrainMemoryOptions.network_spec` 必填
- 推理：`workflow_runtime_init(..., spec)` 必填
- 权重校验：`workflow_weights_count(spec)`

## 4. 构建与测试

```bash
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

## 5. sevenseg 子工程

- 数据生成：`sevenseg_train --export-only <dir>`
- 推理入口：`sevenseg_infer_bin / sevenseg_infer_c_array / sevenseg_infer_c_func`
- 基准入口：`sevenseg_benchmark`
- 所有入口都支持自动解析数据目录

## 6. 新增节点类型的开发流程

1. 在 `NetworkNodeType` / `NetworkLayerKind` 增加枚举
2. 在 `network_spec_build_from_graph` 增加映射规则
3. 在 `profiler_run` 增加估算模型
4. 在训练与推理执行路径增加算子分发
5. 补充 demo 与测试

## 7. 禁止事项

- 不新增旧兼容接口
- 不引入模型类型二选一开关
- 不在文档和代码中保留过时构建入口说明
