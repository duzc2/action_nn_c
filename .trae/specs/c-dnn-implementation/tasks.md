# 任务列表 (Tasks)

- [x] 任务 1: 资源评估与 IO 配置工具
    - [x] 实现 `profiler.c`。
    - [x] 实现 IO 映射配置解析。

- [x] 任务 2: 核心算子与数据结构
    - [x] 实现 `Tensor` 与静态内存池。
    - [x] 实现 `MatMul`, `Softmax`, `RMSNorm`, `Gelu`, `Actuator`。

- [x] 任务 3: 独立 Tokenizer 模块
    - [x] 实现 `tokenizer.c`。
    - [x] 确保无依赖独立编译。

- [x] 任务 4: 类 Transformer/MoE 组件
    - [x] 实现 `Embedding`, `Attention`, `MoE`, `TransformerBlock`。

- [x] 任务 5: 序列化与权重内嵌 **(关键)**
    - [x] 实现标准二进制保存/加载。
    - [x] **新增**：实现 `export_weights_to_c_source()`，生成可编译的 `weights.c`。

- [x] 任务 6: 平台适配与协议
    - [x] 协议设计 (Raw/Token)。
    - [x] PC/ESP32 驱动实现。

- [x] 任务 7: 训练引擎与模拟器
    - [x] 实现 CSV 加载、反向传播。
    - [x] 实现 PC 端闭环训练。

# 任务依赖
- 任务 5 依赖 任务 2 (Tensor 结构)。
- 任务 7 依赖 任务 4, 5, 6。
