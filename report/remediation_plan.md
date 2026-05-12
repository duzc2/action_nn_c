# action_c 架构根本性修复方案

**关联报告**: `report/static_analysis_report.md`
**生成日期**: 2026-05-12
**设计原则**: 流程驱动 · 根因消除 · 零向后兼容 · 理想目标架构

---

# 第一部分：根因分析

> 156 个表面问题可归约为 6 个根因。消除 1 个根因 → 消灭 10~40 个表面症状。以下逐一展开。

## 根因全景图

```
6 个根因                             156 个表面症状消失比例

RC1: 无统一内存模型 (Arena)           PRF-C02 C03 C04, PRF-H06, PRF-M05 M06 M11 M12,
                                       STB-M01 M08, STB-M09 等 ~25 项

RC2: Profiler 无流水线抽象             PRF-C01, PRF-H01 H02 H03, PRF-M01 M02 M03,
                                       STB-M03 M04, SEC-M11 等 ~15 项

RC3: 后端注册无类型安全接口             SEC-C03, SEC-M01 M02, MNT-M01 M02 M03,
                                       USA-M01 M04, FUN-C01 等 ~20 项

RC4: 构建是脚本，不是配置               STB-H01 H02 H03 H04 H05 H06, SEC-H01 H02 H03 H04,
                                       PRF-M16 M17, FUN-C02 等 ~35 项

RC5: 代码复制被视为架构常态             MNT-H01 H02 H03, FUN-M04 M05, PRF-H04 H05,
                                       PRF-M09 M10 M13 等 ~25 项

RC6: 无错误处理与诊断基础设施           STB-H07, USA-M01, MNT-M03, SEC-M11,
                                       STB-L01~L05 等 ~20 项

剩余无法归约的独立问题                                                                  ~16 项
  (主要是 wasm 前端 JS bug、硬编码路径、.gitignore 格式等，按独立方案处理)
```

---

# 第二部分：根本性修复方案

> 每个方案从**理想架构**出发，描述目标状态。然后给出从现状到目标的**转换动作**。不需要向后兼容，以最终架构的正确性为唯一目标。

---

## 方案 RC1：统一内存模型 — Arena Allocator

### 根因描述

当前每个后端独立管理临时缓冲区。有的预分配（部分字段），有的每步 `calloc/free`。根源在于：**工程中没有"临时分配的归属和生命周期"这一概念**。`malloc` 被当作唯一的分配手段，而 `malloc` 的语义（"谁申请谁释放"）天然要求调用方记得释放——这正是热路径堆分配问题的根源。

### 理想架构

```
Context 创建时:
  Context.arena = arena_create(max_step_scratch_size)
  // arena 内所有分配在 step 结束时统一回收，不产生 free 调用

Step 执行时:
  arena_mark = arena_snapshot(ctx->arena)          // 记录水位线
  float* delta = ARENA_ALLOC(ctx->arena, float, n)  // bump 分配，O(1)
  float* grad  = ARENA_ALLOC(ctx->arena, float, m)
  // ... 使用 delta, grad ...
  arena_restore(ctx->arena, arena_mark)             // 一步回卷，O(1)

Context 销毁时:
  arena_destroy(ctx->arena)
```

**核心语义**: Arena 不跟踪单个分配。所有分配在一个 step 内有效。step 结束时回卷水位线，一步释放所有临时内存。不存在泄漏，不需要 free，不存在分配/释放的不对称。

### 转换动作

| 步骤 | 内容 | 消灭的问题 |
|------|------|-----------|
| RC1.1 | 新建 `src/utils/arena.h` + `arena.c`，实现 bump allocator + mark/restore | — |
| RC1.2 | 在所有 Context 结构体（MlpTrainContext, GNNInferContext 等 8 个）中添加 `Arena* arena` 字段 | — |
| RC1.3 | Context `create` 时计算 max_step_scratch_size 并创建 arena | PRF-M14 (gnn config 不再每步验证) |
| RC1.4 | 修改所有后端的 `forward_pass` / `backward_pass` / `train_step`：用 `ARENA_ALLOC` 替代所有 `calloc` + 移除所有对应 `free` | PRF-C02 C03 C04, PRF-H06, PRF-M05 M06 M11 M12, PRF-M13 |
| RC1.5 | 在每个 step 入口 `arena_snapshot`，出口 `arena_restore` | STB-M01 (realloc 泄漏路径消失), STB-M08 M09 |
| RC1.6 | 将所有后端的 `destroy` 函数统一为 `arena_destroy(ctx->arena) + free(ctx)` | PRF-M07 M08 (契约缓存初始化可走 arena) |
| RC1.7 | 添加 `arena_overflow_handler`：arena 耗尽时自动 `realloc` 扩展底层内存（O(log n) 次扩展） | — |

### 为什么这是根本性解决

不是给每个 `calloc` 加检查，而是**改变分配范式**。Arena 内部没有"忘记释放"的概念——回卷是必然的。没有"分配失败"——arena 预先保证容量充足。代码从"申请-判断-使用-释放"的 4 步防御模式变为"申请-使用"的 2 步流模式。

---

## 方案 RC2：Profiler 流水线架构

### 根因描述

Profiler 的 validate → flatten → hash → codegen 各阶段被实现为互相不知道对方存在的独立函数。codegen 不知道 flatten 已经做过了，于是自己做一遍。validate 内部多次 flatten——三次。每个阶段都要自己从头构建叶子图。根源在于：**没有"流水线中间结果"这一概念**。每个函数是孤岛。

### 理想架构

```
网络定义 (NNNetworkDef*)
    │
    ▼
[Validate ──► ValidationResult]     不可变
    │
    ▼
[Flatten  ──► FlatNetwork]          不可变
    │  (叶子列表 + 邻接表 + ID→索引映射)
    ▼
[Hash     ──► NetworkHashes]        不可变
    │  ({network_hash, layout_hash, abi_version})
    ▼
[Codegen  ──► GeneratedFiles]       不可变
    │  (7 个 .c/.h 文件的内容)
    ▼
写入磁盘
```

**核心语义**: 每个阶段输出一个不可变结果对象。下游阶段只读上游的输出。没有阶段重复执行上游的工作。管道在 `generate_all` 中一次性构建，每个阶段的结果在整个管道生命周期内复用。

### 转换动作

| 步骤 | 内容 | 消灭的问题 |
|------|------|-----------|
| RC2.1 | 定义 `ValidationResult`、`FlatNetwork`、`NetworkHashes`、`GeneratedFiles` 四个不可变结果类型 | — |
| RC2.2 | `FlatNetwork` 中包含：`SubnetList`（容量字段 + 几何增长）+ `CSRAdjacencyList` + `StringHashMap(id→index)` | PRF-H01 H02 H03, STB-M01 |
| RC2.3 | `FlatNetwork` 中新增 `prof_string_hash_map` 作为 `prof_hash.c` 的一部分 | PRF-M01 (O(n²) ID检测 → O(n)) |
| RC2.4 | 重构 `profiler_generate_v2` 为一个管道：4 个阶段按序执行，结果沿管道流动 | PRF-C01, PRF-M02 M03 |
| RC2.5 | 验证阶段扁平化为：`validate_network_def(network)` → `validate_connections(network, flat)` → `validate_dag(flat)`，flat 只构建一次 | — |
| RC2.6 | Codegen 阶段接收 `const FlatNetwork*`，7 个发射器不再各自构建图 | PRF-C01（彻底消除）, STB-M03 M04 |
| RC2.7 | `write_file` 改为返回 `ProfStatus` 而非 `void`，在管道末端集中处理 I/O 错误 | SEC-M11 |

### 为什么这是根本性解决

不是"缓存结果避免重复计算"，而是**重新定义计算模型**。管道保证了每个信息只被推导一次，就像编译器不会对同一个 AST 做 7 次类型检查。信息流是单向的、不可变的、非冗余的。

---

## 方案 RC3：后端类型安全接口 — VTable 注册模型

### 根因描述

当前每个后端通过 `nn_type_mlp_infer.c` 等 bridge 文件注册。bridge 文件中大量使用 `(void*)` 强制类型转换和函数指针类型转换（CERT EXP37-C UB）。六个后端各有自己的上下文结构体，但彼此之间没有形式化的共性接口。根源在于：**没有"后端接口协议"这一类型层面的抽象**。bridge 文件是对缺失抽象的补丁。

### 理想架构

```c
// ── 在 nn_codegen_hooks.h 中定义 ──

typedef struct NNInferBackend {
    const char* type_name;                           // "mlp", "transformer", ...

    // 生命周期
    void* (*create) (const void* config_blob, size_t config_size, Arena* arena);
    void  (*destroy)(void* context);

    // 推理
    int   (*step)     (void* context);
    int   (*get_output)(const void* context, float* out, size_t out_size);

    // 持久化
    int   (*save_weights)(const void* context, FILE* fp);
    int   (*load_weights)(void* context, FILE* fp);

    // 元信息
    uint64_t (*get_network_hash)(const void* context);
    uint64_t (*get_layout_hash) (const void* context);
    uint32_t (*get_abi_version) (void);
} NNInferBackend;

typedef struct NNTrainBackend {
    const char* type_name;

    void* (*create) (const void* config_blob, size_t config_size,
                     const void* infer_config_blob, size_t infer_config_size,
                     Arena* arena);
    void  (*destroy)(void* context);

    int   (*step)           (void* context, const float* input, const float* target);
    int   (*step_with_data) (void* context, const float* input, const float* target, float* out_grad);
    int   (*save_checkpoint)(void* context, FILE* fp);
    int   (*load_checkpoint)(void* context, FILE* fp);
} NNTrainBackend;

// ── 注册 ──
void nn_register_infer_backend (const NNInferBackend* backend);
void nn_register_train_backend (const NNTrainBackend* backend);

// ── 每个后端实现 vtable ──
// 在 mlp/mlp_infer_ops.c:
static const NNInferBackend g_mlp_infer_backend = {
    .type_name    = "mlp",
    .create       = mlp_infer_create_impl,
    .destroy      = mlp_infer_destroy_impl,
    .step         = mlp_infer_step_impl,
    // ...
};

// ── 在 bridge 文件中（编译期注册） ──
// nn_type_mlp_infer.c:
__attribute__((constructor)) static void _register_mlp_infer(void) {
    nn_register_infer_backend(&g_mlp_infer_backend);
}
```

**核心语义**: VTable 中所有函数签名已对齐——`void*` 是后端间的约定，vtable 内的函数**接受 void* 并在实现体第一行转换为具体类型**，这是类型安全的（与当前"函数指针类型转换"不同：这里转换的是**参数值**而非**函数类型**，不违反 EXP37-C）。

注册表不再存储孤立的函数指针，而是存储 `const NNInferBackend*`。查找时返回 vtable 指针，调用方通过 vtable 调用。

### 转换动作

| 步骤 | 内容 | 消灭的问题 |
|------|------|-----------|
| RC3.1 | 在 `nn_codegen_hooks.h` 中定义 `NNInferBackend`、`NNTrainBackend` vtable | SEC-C03 |
| RC3.2 | 重写 `nn_train_registry.c` / `nn_infer_registry.c`：slot 存储 vtable 指针而非孤立的函数指针组 | MNT-M01 M02 |
| RC3.3 | 删除 `nn_graph_*_contract.c` 中的桥接调度层（vtable 内部已完成调度） | PRF-M07 M08, STB-M06 |
| RC3.4 | 在 `nn_weight_header.h` 中统一 `NNWeightHeader` + `nn_validate_weight_header()` 函数，所有后端通过 vtable 的 `get_network_hash`/`get_layout_hash` 自动获取期望值 | SEC-C02, FUN-C01, FUN-M06 |
| RC3.5 | 每个后端实现 vtable：MLP、Transformer、CNN、CNN-Dual-Pool、RNN、GNN | SEC-M01 M02, FUN-M04 M05 |
| RC3.6 | 桥接文件 `nn_type_*_*.c` 仅剩编译期 `constructor` 注册调用 | USA-L01 L04 |
| RC3.7 | `copy_type_name` 合并为单一实现，使用 `strncpy` + 静态长度上限 | MNT-M02 |

### 为什么这是根本性解决

不是给每个桥接文件修正类型转换，而是**让类型转换不再需要**。VTable 将"类型差异"封装在 vtable 实现内部，调用方看到的始终是统一的 `NNInferBackend*` 接口。不再有孤立的函数指针组需要手工维护一致性。

---

## 方案 RC4：构建系统 — 配置驱动，消除脚本

### 根因描述

30+ 个 CMakeLists.txt 互相不知道对方的存在。`file(GLOB)` 不跟踪新文件。`demo_common.cmake` 中的函数与 `src/CMakeLists.txt` 重复定义。生成文件的依赖关系靠约定（"请先跑 generate 再跑 train"）而非 CMake 依赖图。根源在于：**构建被当作胶水脚本，而非声明式配置**。

### 理想架构

```
CMakeLists.txt (根)
    ├── cmake/
    │   ├── compiler_flags.cmake    (action_c_apply_hardening + strict_warnings, 唯一定义)
    │   ├── network_types.cmake     (action_c_enable_network_types)
    │   └── version.cmake           (统一版本号 → 生成头文件)
    │
    ├── src/CMakeLists.txt          (include cmake/*, 不再定义函数)
    ├── demo/
    │   └── CMakeLists.txt           (单入口, add_subdirectory 各 demo)
    │       ├── move/CMakeLists.txt  (generate → train → infer 三段构建)
    │       │   └── add_dependencies(move_train move_generate)
    │       │   └── add_dependencies(move_infer  move_generate)
    │       └── ...
    │
    ├── wasm/CMakeLists.txt          (Emscripten, 从 wasm_exports.c 自动提取导出列表)
    │
    └── CMakePresets.json            (Debug / Release / Hardened / Sanitized 四种预设)
```

**核心语义**:
- 所有 CMake 函数唯一定义于 `cmake/`，通过 `include()` 共享
- 源文件显式列出或使用 `CONFIGURE_DEPENDS`
- demo 的 generate → train → infer 有 `add_dependencies` 的显式构建顺序
- Wasm 导出列表通过 CMake 脚本从 `wasm_exports.c` 自动提取，永不同步
- 一个 `CMakePresets.json` 替代所有 `.bat`/`.sh`/`.ps1` 脚本中的 `-D` 参数

### 转换动作

| 步骤 | 内容 | 消灭的问题 |
|------|------|-----------|
| RC4.1 | 创建 `cmake/compiler_flags.cmake`：合并 `action_c_apply_strict_warnings` + `action_c_apply_hardening` + `action_c_enable_lto` | STB-H02, SEC-H01, PRF-M16 |
| RC4.2 | 创建 `cmake/network_types.cmake`：合并 `action_c_enable_network_types` | — |
| RC4.3 | 创建 `cmake/version.cmake`：从单一 `VERSION` 变量生成 `wasm_config.h` 中的宏 | FUN-M10 |
| RC4.4 | `src/CMakeLists.txt` + `demo/demo_common.cmake`：删除函数定义，改为 `include(cmake/...)` | STB-H02, STB-M11 |
| RC4.5 | `src/nn/CMakeLists.txt`：`file(GLOB ... CONFIGURE_DEPENDS)` | STB-H01 |
| RC4.6 | 每个 demo 的 `train/CMakeLists.txt` + `infer/CMakeLists.txt`：添加 `add_dependencies(xxx_train xxx_generate)` | STB-H04 |
| RC4.7 | `demo/demo_common.cmake`：`action_c_get_demo_generated_dir` 使用 CMake cache 变量 | STB-H03 |
| RC4.8 | 重写 `build_demos.ps1`：调用 `cmake --preset demo-all` 而非手工循环 | STB-H06 |
| RC4.9 | 创建 `CMakePresets.json`：debug / release / hardened / sanitized 四种预设 | STB-H05, SEC-H01 |
| RC4.10 | `wasm/CMakeLists.txt`：自动从 `wasm_exports.c` 提取导出列表 | STB-C01, USA-H01, FUN-H04 |
| RC4.11 | 修复 `build_wasm.sh` 中的 `$EMSDK` 展开 + 添加未设置检测 | FUN-C02, SEC-H04 |
| RC4.12 | `demo/cs/tools/CMakeLists.txt`：移除硬编码 VS 路径；非 Windows 编译跳过 probe | SEC-H02 H03, FUN-H05 |
| RC4.13 | 统一所有 demo CMakeLists.txt 中 `CMAKE_C_EXTENSIONS OFF` | USA-M02 |

### 为什么这是根本性解决

不是修每个脚本中的每个硬编码路径，而是**让硬编码路径不再存在**。CMakePresets.json 集中管理所有构建变体。Wasm 导出列表自动生成而非手工维护。依赖关系由 CMake 图计算而非约定。脚本从"编排者"退化（提升）为"预设调用者"。

---

## 方案 RC5：实现去重 — 参数化替代复制

### 根因描述

CNN 和 CNN_Dual_Pool 是两个独立目录，差异仅在于后者存储 `max_index`。Transformer 推理和训练前向传播是两个函数，差异仅在是否存储中间结果。MNIST dataset 有两个副本。根源在于：**代码复制是当前架构鼓励的扩展方式**。添加新网络类型 = 复制整个目录然后修改几行。这是架构纵容的。

### 理想架构

```
src/nn/types/cnn/
    cnn_ops.c           ← 唯一实现，通过 CnnPoolingMode 控制行为
    cnn_config.h        ← enum CnnPoolingMode { AVG, MAX, DUAL }
    nn_type_cnn_infer.c
    nn_type_cnn_train.c

src/nn/types/transformer/
    transformer_forward.c  ← 唯一前向实现，通过 flags 控制中间结果存储
    transformer_infer.c
    transformer_train.c

demo/mnist_common/
    mnist_idx_reader.c     ← 唯一 IDX 读取实现，参数化 MNISTDatasetConfig
```

**核心语义**:
- CNN 使用 `CnnPoolingMode` 枚举区分 AVG / MAX / DUAL 三种池化策略。编译期 `if (mode == DUAL)` 分支的额外存储开销仅在 dual 模式生效。
- Transformer 前向传播是一个函数，定义一个 `flags` 参数控制是否存储 `pooled` 等训练时才需要的中间结果，替代两个几乎相同的函数。
- MNIST 读取器通过 `normalize_to_01` 和 `target_channels` 两个配置字段适配 MLP 和 CNN 的不同需求。

### 转换动作

| 步骤 | 内容 | 消灭的问题 |
|------|------|-----------|
| RC5.1 | 在 `cnn_config.h` 中添加 `CnnPoolingMode` 枚举；`cnn_forward_pass` 中用一个 `if(mode==DUAL)` 替代两个函数；删除 `cnn_dual_pool/` 整个目录 | MNT-H01, FUN-L11 |
| RC5.2 | `cnn_backpropagate` 合并 CNN 和 Dual-Pool 的反向传播（dual 额外的 `max_index` 参数在 mode!=DUAL 时为 NULL） | PRF-H04 H05 |
| RC5.3 | `transformer_forward.c`：提取为一个函数，接受 `flags` (NONE / STORE_POOLED / STORE_ALL)，用 `if(flags & ...)` 控制存储 | MNT-H02, FUN-M02 M03 |
| RC5.4 | 在 `mlp_layers.h` 中为激活函数定义 `typedef float (*ActivationFn)(float)`，在推理上下文创建时预解析为函数指针 | PRF-M10 |
| RC5.5 | `demo/mnist_common/mnist_idx_reader.c`：统一 MNIST 读取器 | MNT-H03 |
| RC5.6 | `pcg_rand` 改为纯 32 位版本 | PRF-M09 |
| RC5.7 | 为所有张量操作添加 `float* restrict` | PRF-M15, MNT-L02 |
| RC5.8 | CNN im2col + GEMM 范式（替代深度嵌套循环，同时受益于 SIMD） | PRF-M15 |
| RC5.9 | RNN BPTT：指针交换替代 memcpy | PRF-M13 |

### 为什么这是根本性解决

不是对两段相似代码各自优化，而是**让两段代码不存在**。"两份相似代码"这个事实本身是 bug。消除重复后，修复 CNN 的一个 bug 不会再需要"另一份也记得修"的人类记忆。

---

## 方案 RC6：统一错误处理与诊断基础设施

### 根因描述

返回错误码 `-1`、`0`、`1`、`-2` 散布全工程。函数静默失败。无日志。无断言。根源在于：**错误处理没有被当作一个横切关注点来设计**。它是事后在代码中随机插入的。

### 理想架构

```c
// ── src/utils/error.h ──
typedef enum {
    ACTION_C_OK = 0,

    ACTION_C_ERR_NULL_POINTER   = -1,
    ACTION_C_ERR_INVALID_ARG    = -2,
    ACTION_C_ERR_NO_MEMORY      = -3,
    ACTION_C_ERR_DIM_MISMATCH   = -4,
    ACTION_C_ERR_IO_FAILED      = -5,
    ACTION_C_ERR_VERSION_MISMATCH = -6,
    ACTION_C_ERR_CYCLE_DETECTED = -7,
    ACTION_C_ERR_CONFIG_INVALID = -8,
    ACTION_C_ERR_NOT_FOUND      = -9,
    ACTION_C_ERR_INTERNAL       = -99,
} ActionCError;

// ── src/utils/log.h ──
#ifndef ACTION_C_LOG_LEVEL
#define ACTION_C_LOG_LEVEL 2   // 0=OFF, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG
#endif

#define LOG_ERROR(fmt, ...)  _action_c_log(1, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)   _action_c_log(2, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)   _action_c_log(3, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

// Debug 断言 —— Release 完全消除
#ifndef NDEBUG
#define ASSERT(cond, err, msg) do { \
    if (!(cond)) { LOG_ERROR("ASSERT: %s - %s", #cond, msg); return (err); } \
} while(0)
#else
#define ASSERT(cond, err, msg) ((void)0)
#endif
```

**核心语义**:
- `ActionCError` 是工程中所有函数的唯一返回值类型
- `LOG_ERROR` 在函数失败时自动记录文件名和行号（不是调用方手工写 `fprintf`）
- `ASSERT` 在 Debug 模式验证不变量，Release 编译时完全移除（零开销）
- 日志级别由编译宏控制，可完全禁用

### 转换动作

| 步骤 | 内容 | 消灭的问题 |
|------|------|-----------|
| RC6.1 | 新建 `src/utils/error.h` + `src/utils/log.h` | MNT-M03, USA-M01 |
| RC6.2 | 全工程搜索 `return -1; return -2; return 1;`，替换为 `ActionCError` 枚举值 | 全部 MNT-M03 |
| RC6.3 | 在所有后端 `create` 和 `step` 函数中添加 `ASSERT(ctx != NULL, ...)` | STB-L05, FUN-L03 L04 |
| RC6.4 | `sevenseg/train_main.c`：将 `train_step(...)` / `infer_auto_run(...)` 调用改为 `if(rc != ACTION_C_OK) { LOG_ERROR(...); goto cleanup; }` | STB-H07 |
| RC6.5 | 所有 demo 的 `train_main.c`：统一训练失败日志 + early exit 模式 | SEC-L01 |
| RC6.6 | `prof_error_set` 改为使用 `LOG_ERROR` + `ActionCError`（淘汰异构的 `ProfStatus` 单独错误体系，或不淘汰但映射到 ActionCError） | FUN-L01 L02 |
| RC6.7 | `safe_realloc` 等工具函数使用 `ASSERT` 验证前置条件 + 返回 `ACTION_C_ERR_NO_MEMORY` | STB-M01 |

### 为什么这是根本性解决

不是给每个 `return -1` 加 printf，而是**让 `return -1` 不再存在**。`ActionCError` 枚举让错误码自文档化。`LOG_ERROR` 让诊断输出零人工成本。`ASSERT` 让不变量在 Debug 中自动验证。错误处理从"人工记忆"变为"编译器强制执行"。

---

# 第三部分：方案间的依赖与合并策略

> 六个方案不是独立的。它们之间存在"基础设施→上层消费"的供给关系。

## 依赖拓扑

```
Layer 0 (无依赖，先行):
  RC6: 错误处理与诊断基础设施
        ↓ (所有上层代码使用 ActionCError + LOG_ERROR + ASSERT)

Layer 1 (依赖 RC6):
  RC1: Arena 内存模型
  RC4: 构建系统配置化

Layer 2 (依赖 RC1 + RC6):
  RC2: Profiler 流水线
  RC3: 后端 VTable 注册
  RC5: 实现去重

Layer 3 (依赖 RC1 + RC2 + RC3 + RC5):
  RC3.5~RC3.7: 各后端 vtable 实现 + CNN 合并 + Transformer 合并
  (这些工作依赖 Arena 已就位、VTable 已定义、Profiler 管道已建立)
```

## 三轨并行策略

六方案按依赖关系合并为三轨，可并行推进：

```
轨 A: 基础设施 + 内存 (RC6 → RC1)
  负责人 A: RC6 (error.h + log.h) → RC1 (arena.h + arena.c + Context 改造)

轨 B: 接口 + 去重 (RC3 → RC5)
  负责人 B: RC3 (VTable 定义 + 注册表重写) → RC5 (CNN/Transformer/MNIST 去重)

轨 C: 构建 + 配置 (RC4)
  负责人 C: RC4 (cmake/ 模块 + CMakePresets + 脚本重写)

三轨并行完成后:
  → RC2 (Profiler 流水线): 依赖 A 的 Arena + B 的 VTable + C 的构建系统
  → 最终集成测试
```

## 三轨不影响的问题（独立处理）

以下问题不归属六个根因，在任意轨进行时可独立修复：

| 问题 | 修复 | 轨归属 |
|------|------|--------|
| STB-M10: wasm demo JS 优先级 | 修正括号 | 轨 C (构建) |
| STB-L06: demo_runtime_paths.h POSIX | `/proc/self/exe` + `_NSGetExecutablePath` | 独立 |
| USA-L02 L03: "????????" 占位符 | 替换为有意义的描述 | 独立 |
| USA-L09: .gitignore 围栏标记 | 删除 Markdown 标记 | 独立 |
| FUN-H01 H02 H03 H06: Web Editor | 删除遗留代码 + 连接生成 + package-lock | 独立（前端轨） |
| SEC-M05~M07: Web Editor 前端安全 | CSP + textContent + host:false | 独立（前端轨） |

---

# 第四部分：理想架构下的代码变更清单

## 新增文件

```
src/utils/
    arena.h                Arena bump allocator (mark/restore 语义)
    arena.c
    error.h                ActionCError 枚举 + LOG_ERROR/ASSERT 宏
    log.h                  _action_c_log 实现
    log.c
    safe_math.h            整数安全运算 (SAFE_MUL 等)

src/nn/
    nn_backend.h           NNInferBackend / NNTrainBackend vtable 定义
    nn_weight_header.h     统一 NNWeightHeader + validate 函数

cmake/
    compiler_flags.cmake   单一真实来源: strict_warnings + hardening + lto
    network_types.cmake    单一真实来源: action_c_enable_network_types
    version.cmake          单一真实来源: 版本号 → 生成头文件

CMakePresets.json          四种预设: debug / release / hardened / sanitized

demo/mnist_common/
    mnist_idx_reader.h
    mnist_idx_reader.c
```

## 修改文件

```
src/profiler/
    profiler.c             → generate_v2 管道化，接收 FlatNetwork*
    prof_codegen.c         → 7 个发射器改接收 const FlatNetwork*
    prof_flatten.c/h       → FlatNetwork 含 capacity + 邻接表 + ID 哈希
    prof_validate.c        → 单次 flatten，传递 FlatNetwork*
    prof_hash.c/h          → 新增 prof_string_hash_map
    prof_error.c           → 使用 ActionCError

src/nn/
    nn_train_registry.c    → VTable 注册模型，合并为单次扫描
    nn_infer_registry.c    → 同上
    nn_graph_*_contract.c  → 删除桥接层（VTable 替代）
    nn_codegen_hooks.h     → 定义 vtable 类型 + weight header

src/nn/types/mlp/
    mlp_infer_ops.c        → 实现 NNInferBackend vtable + Arena 使用
    mlp_train_ops.c        → 实现 NNTrainBackend vtable + Arena 使用
    mlp_layers.c           → SAFE_MUL + Arena allocation
    mlp_layers.h           → ActivationFn typedef
    nn_type_mlp_infer.c    → constructor 注册 vtable
    nn_type_mlp_train.c    → constructor 注册 vtable

src/nn/types/transformer/
    transformer_forward.c  → 新建：合并训练/推理前向
    (infer_ops / train_ops → 改为调用统一前向 + vtable 实现)

src/nn/types/cnn/
    cnn_config.h           → + CnnPoolingMode 枚举
    cnn_ops.c              → 合并 CNN + Dual-Pool 卷积/池化
    (删除 src/nn/types/cnn_dual_pool/ 整个目录)

src/nn/types/rnn/
    rnn_train_ops.c        → Arena + 指针交换
    rnn_infer_ops.c        → vtable 实现

src/nn/types/gnn/
    gnn_infer_ops.c        → Arena + vtable 实现
    gnn_train_ops.c        → Arena + vtable 实现

CMakeLists.txt (根)        → 添加 cmake/ include
src/CMakeLists.txt         → include cmake/*，删除函数定义
src/nn/CMakeLists.txt      → CONFIGURE_DEPENDS + 移除 cnn_dual_pool
demo/demo_common.cmake     → include cmake/*，删除重复定义

wasm/
    CMakeLists.txt          → 自动提取导出列表
    scripts/build_wasm.sh  → EMSDK 展开修复
    src/wasm_exports.c     → 补全缺失函数 + Arena 集成
    js/action_nn_c.d.ts    → 与导出列表同步
```

## 删除文件

```
web_editor/js/editor.js         (Rete 1.x 死代码)
web_editor/js/nodes.js
web_editor/css/style.css
web_editor/start.sh

src/nn/types/cnn_dual_pool/     (整个目录，合并到 cnn/)

demo/cs/tools/cs_tool_common.h  (CsLabelSegment 废弃结构体)
demo/cn/... (cs_tool_common.c 中废弃函数)

.codex_build_cnn_rnn_infer.cmd  (被 CMakePresets 替代)
build_demos.ps1                  (重写为 preset 调用)
```

---

# 第五部分：效果预测

| 维度 | 修复前 | 修复后 |
|------|--------|--------|
| **表面问题总数** | 156 | ~0（根因消除后症状自动消失） |
| **根因数量** | — | 6（全部消除） |
| **新增基础设施文件** | — | 12 个 |
| **修改文件数** | — | ~55 个 |
| **删除文件/目录数** | — | 8 个文件 + 2 个目录 |
| **热路径堆分配** | ~30 次/step | 0 次/step |
| **Profiler 遍历次数** | 7-10 次 | 1 次 |
| **CNN 实现数** | 2 个 | 1 个 |
| **Transformer 前向实现数** | 2 个 | 1 个 |
| **MNIST 读取器数** | 2 个 | 1 个 |
| **CMake 函数重复定义** | 2 处 | 0 |
| **硬编码路径** | ~8 处 | 0 |
| **函数指针 UB** | 12 处/6 文件 | 0 |
| **错误码约定** | 3 种混用 | 1 种 (ActionCError) |
| **Wasm 导出不一致** | 8 个不匹配 | 0（自动生成） |
| **构建预设** | 0 | 4 种 |
