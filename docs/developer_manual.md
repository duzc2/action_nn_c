# 开发手册

## 1. 目标

本文档说明代码结构、扩展方式、测试策略和交付标准。  
目标读者是维护者和二次开发者。

## 2. 系统边界

本仓库提供：
- C99 推理链路
- 最小训练闭环
- 权重导出与回加载
- 测试框架和模型专项测试

本仓库不提供：
- 通用训练框架
- 自动微分
- GPU 运行时

## 3. 架构与职责

### 3.1 模块目录

- `src/include/`：稳定接口和配置
- `src/core/`：核心算子与兼容入口
- `src/tokenizer/`：词表与文本编码实现
- `src/model/`：模型层实现
- `src/train/`：CSV 数据加载
- `src/tools/`：工具程序与端到端示例
- `test/`：测试框架和测试用例

### 3.2 核心数据对象

- `Tensor`：只保存视图，不持有生命周期  
  [tensor.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/tensor.h#L21-L34)
- `Vocabulary` / `Tokenizer`：文本到 token id 映射  
  [tokenizer.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/tokenizer.h#L26-L43)
- `ProtocolFrame`：RAW/TOK 统一帧  
  [protocol.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/protocol.h#L31-L36)
- `WorkflowLoopOptions`：推理循环配置与动作回调  
  [workflow.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/workflow.h#L26-L34)

### 3.3 执行链路

固定链路：
1. `tokenizer_encode`
2. 模型前向
3. `op_actuator`
4. `action_callback`

链路约束：
- 控制循环由外部维护
- 模型只做单帧决策

## 4. 配置管理

配置入口：  
[config_user.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/config_user.h)

配置分三类：
- 结构参数：`EMBED_DIM / NUM_LAYERS / NUM_HEADS / FFN_DIM`
- I/O 参数：`VOCAB_SIZE / STATE_DIM / OUTPUT_DIM / MAX_SEQ_LEN`
- 输出映射：`IO_MAPPING_ACTIVATIONS / IO_MAPPING_NAMES`

变更规则：
- 先改配置，再跑 profiler，再跑全量测试
- 训练后不要变更输出通道语义

## 5. 工具链

### 5.1 profiler

用途：
- 估算参数量、激活内存、FLOPs
- 输出 `network_def.h`

实现：  
[profiler.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/profiler.c)

### 5.2 min_train_loop

用途：
- 验证最小训练闭环
- 验证协议 RAW/TOK 两种输入

实现：  
[min_train_loop.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/min_train_loop.c)

### 5.3 c99_full_demo

用途：
- 端到端链路验证
- 外部循环多步推理示例

实现：  
[c99_full_demo.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c)

## 6. 开发流程

标准顺序：
1. 定义需求和 I/O
2. 修改配置
3. 跑 profiler
4. 开发实现
5. 增加测试
6. 跑全量测试
7. 更新文档

构建命令：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build
```

## 7. 扩展指南

### 7.1 新增模型算子

步骤：
1. 在 `src/include` 增加接口
2. 在 `src/core` 或 `src/model` 实现
3. 在 `CMakeLists.txt` 接入源码
4. 增加单元测试和负向测试

### 7.2 新增协议字段

步骤：
1. 修改 `protocol_encode_*`
2. 修改 `protocol_decode_packet`
3. 增加格式错误测试
4. 校验旧格式兼容性

### 7.3 新增执行层回调

步骤：
1. 在业务入口实现 `on_action` 回调
2. 对接目标设备 SDK
3. 将回调赋值给 `WorkflowLoopOptions.action_callback`
4. 增加执行层回归测试

## 8. 测试体系

测试入口：  
[test_suite.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/test/src/test_suite.c)

当前分组：
- 单元
- 正确性
- 错误
- 边界
- 压力
- 集成
- 模型专项

模型专项覆盖：
- 泛化
- OOD
- 对抗扰动
- 长时序极限
- 预期错误路径

模型专项文件：  
[test_cases_model_special.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/test/src/test_cases_model_special.c)

## 9. 代码质量要求

- 保持 C99
- 返回码必须可判定
- 内存释放路径完整
- 新增能力必须有测试
- 禁止提交运行产物

## 10. 交付清单

合并前必须完成：
- clang 构建通过
- `full_test_suite` 全绿
- 日志可追踪关键指标
- 文档同步更新
