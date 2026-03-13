# 任务列表 (Tasks)

- [ ] 任务 1: 资源评估与 IO 配置工具
    - [ ] 实现 `profiler.c`。
    - [ ] 实现 IO 映射配置解析。

- [ ] 任务 2: 核心算子与数据结构
    - [ ] 实现 `Tensor` 与静态内存池。
    - [ ] 实现 `MatMul`, `Softmax`, `RMSNorm`, `Gelu`, `Actuator`。

- [ ] 任务 3: 独立 Tokenizer 模块
    - [ ] 实现 `tokenizer.c`。
    - [ ] 确保无依赖独立编译。

- [ ] 任务 4: 类 Transformer/MoE 组件
    - [ ] 实现 `Embedding`, `Attention`, `MoE`, `TransformerBlock`。

- [ ] 任务 5: 序列化与权重内嵌 **(关键)**
    - [ ] 实现标准二进制保存/加载。
    - [ ] **新增**：实现 `export_weights_to_c_source()`，生成可编译的 `weights.c`。

- [ ] 任务 6: 平台适配与协议
    - [ ] 协议设计 (Raw/Token)。
    - [ ] PC/ESP32 驱动实现。

- [ ] 任务 7: 训练引擎与模拟器
    - [ ] 实现 CSV 加载、反向传播。
    - [ ] 实现 PC 端闭环训练。

# 任务依赖
- 任务 5 依赖 任务 2 (Tensor 结构)。
- 任务 7 依赖 任务 4, 5, 6。
