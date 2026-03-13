# C语言类大模型神经网络 - 技术开发与架构设计文档

## 1. 核心数据结构设计 (Data Structures)

为了满足 C99 标准、零依赖和单线程高性能的要求，所有数据结构设计遵循“扁平化”和“静态化”原则。

### 1.1 张量 (Tensor)
核心运算单元。在训练模式下包含梯度，在推理模式下仅包含数值。

```c
typedef struct {
    float* data;      // 指向数值数组的指针
    float* grad;      // 指向梯度数组的指针 (推理模式下为 NULL)
    int shape[4];     // 支持最高4维 [Batch, Seq, Head, Dim]
    int ndim;         // 有效维度数
    int size;         // 元素总数
    bool requires_grad;
} Tensor;
```

### 1.2 全局配置 (Global Config)
由 Profiler 生成的 `network_def.h` 中的宏驱动。

```c
// network_def.h (Auto-generated)
#define MAX_SEQ_LEN 32
#define EMBED_DIM   64
#define NUM_LAYERS  4
#define NUM_HEADS   4
#define FFN_DIM     256
#define VOCAB_SIZE  128
```

### 1.3 模型组件 (Model Components)

**Attention 层结构：**
```c
typedef struct {
    Tensor W_Q, W_K, W_V, W_O; // 权重
    Tensor B_Q, B_K, B_V, B_O; // 偏置
    int num_heads;
    int head_dim;
} AttentionLayer;
```

**MoE 层结构：**
```c
typedef struct {
    Tensor W_Gate;      // 路由门控权重
    Expert* experts;    // 专家数组
    int num_experts;
    int k_top;          // 激活专家数
} MoELayer;
```

## 2. 内存管理策略 (Memory Management)

为了适应 ESP32 的 SRAM 限制，杜绝运行时 `malloc` 碎片。

### 2.1 静态内存池 (Static Memory Pool)
系统启动时分配两大块连续内存：
1.  **Parameter Arena**: 存储所有权重 (Weights & Biases)。
    *   *PC 端*：`malloc` 分配。
    *   *ESP32 (High-Perf Mode)*：**直接映射到 .rodata 段 (Flash)**，通过 `weights.c` 编译生成，零 RAM 占用。
    *   *ESP32 (Debug Mode)*：从 SD 卡/Flash 文件系统加载到 RAM。
2.  **Activation Arena**: 存储推理/训练过程中的中间结果。
    *   采用 **Ping-Pong Buffer** 或 **Scratchpad** 复用机制。前一层的输出覆盖再前一层的输入（需拓扑分析）。

### 2.2 零拷贝张量初始化
`Tensor` 结构体本身不拥有内存，只通过指针指向 Arena 中的特定偏移量。

## 3. 关键模块详解

### 3.1 Tokenizer (独立无依赖模块)
该模块必须设计为**零依赖库**（不依赖 `tensor.h` 或 `dnn` 核心），以便在任何主机（PC/Server）上独立编译运行。

- **接口**：
```c
// tokenizer.h
typedef struct {
    int* ids;
    int length;
    int capacity;
} TokenSequence;

// 输入: "walk(3,3)" -> 输出: [ID_WALK, ID_LPAREN, ID_3, ...]
void tokenize(const char* input, TokenSequence* out);
```

### 3.2 权重内嵌机制 (Weight Embedding)
为了极致性能，`serializer` 模块需支持生成 C 源码。

- **工作流**：
    1.  训练完成，内存中持有所有 Tensor 的 float 数据。
    2.  调用 `export_weights_to_c(network, "weights.c")`。
    3.  生成如下代码：
```c
// weights.c (Auto-generated)
#include "network_def.h"

// 对齐到 16 字节以利于 SIMD 加载
__attribute__((aligned(16))) 
const float WEIGHT_L0_ATTN_Q[] = { 0.12f, -0.5f, ... };

__attribute__((aligned(16))) 
const float WEIGHT_L0_ATTN_K[] = { ... };

// 指针表，供 Network 初始化时索引
const float* const WEIGHT_TABLE[] = {
    WEIGHT_L0_ATTN_Q,
    WEIGHT_L0_ATTN_K,
    // ...
};
```

### 3.3 Profiler 工具 (资源预测)
Profiler 是一个在编译前运行的 CLI 工具，用于**静态分析**网络架构的资源需求。它不执行实际的矩阵运算，而是进行一次“虚拟前向传播” (Dry Run)。

**核心输出**：
1.  `network_def.h`: 定义层数、维度宏。
2.  **IO 映射报告**：打印每个输出神经元的物理含义。
3.  **内存报告**：静态 Flash 占用 + 动态 RAM 峰值。

## 4. 目录与文件职责详解

```text
src/
├── core/
│   ├── tensor.c       // 张量操作与内存池管理
│   ├── ops_math.c     // MatMul (含 SIMD 优化桩), Scale, Add
│   └── ops_act.c      // Relu, Gelu, Sigmoid, Tanh, Softmax
├── model/
│   ├── attention.c    // Multi-Head Attention 实现
│   ├── moe.c          // Router 与 Expert 选择逻辑
│   ├── layers.c       // Linear, LayerNorm, Embedding
│   └── transformer.c  // 骨干网络组装
├── tokenizer/
│   ├── tokenizer.c    // [独立] 词法分析与 ID 映射
│   └── vocab.c        // 词表管理
├── io_map/
│   └── actuator.c     // 输出层 Sigmoid/Tanh 映射逻辑
├── tools/
│   └── profiler.c     // [核心] 资源计算与 network_def.h 生成器
├── platform/
│   ├── pc/
│   │   ├── driver.c   // Windows/Linux 键鼠模拟
│   │   └── main.c     // 训练与模拟主循环
│   └── esp32/
│       ├── driver.c   // GPIO/PWM 驱动
│       └── main.c     // 嵌入式推理主循环 (支持 Token流 输入)
└── utils/
    └── serializer.c   // 权重保存 (Bin) 与 导出 (C Source)
```

## 5. 构建与部署流程

1.  **配置阶段**：用户编辑 `config_user.h`。
2.  **评估阶段**：运行 `make profiler` -> 生成 `network_def.h`。
3.  **训练阶段**：运行 `make train` (PC) -> 生成 `weights.c` (高性能模式)。
4.  **部署阶段**：
    *   *PC 模拟*：`make sim` (编译链接 `weights.c`)。
    *   *ESP32*：IDF Build (将 `weights.c` 加入 SRCS 列表)。
