# profiler 接口草案（C99）

## 1. 目标与范围

本草案定义以下接口边界：

- 用户程序如何把网络结构体传给 profiler
- profiler 如何接收模块级输出路径（`.c/.h` 可分开存储）
- profiler 如何返回错误码与英文错误描述
- 生成模块在训练/推理中的对外 API 形态
- 训练与推理“双模式循环能力”（自主循环 + 单步触发）

本草案是接口级约定，不包含具体实现代码。

## 2. 目录与文件约束

- 源码实现目录：`src/nn/[network_type]/`
- 每个网络类型目录必须包含：
  - 训练接口与实现：`train_*.h` + `train_*.c`
  - 推理接口与实现：`infer_*.h` + `infer_*.c`
  - 目录说明：`README.md`
- 依赖方向固定：
  - 训练模块可依赖推理模块
  - 推理模块不可依赖训练模块

## 3. 统一类型草案

```c
typedef enum {
    PROF_STATUS_OK = 0,
    PROF_STATUS_INVALID_ARGUMENT = 1,
    PROF_STATUS_VALIDATION_FAILED = 2,
    PROF_STATUS_CYCLE_DETECTED = 3,
    PROF_STATUS_PATH_INVALID = 4,
    PROF_STATUS_IO_FAILED = 5,
    PROF_STATUS_HASH_MISMATCH = 6,
    PROF_STATUS_LAYOUT_MISMATCH = 7,
    PROF_STATUS_INTERNAL_ERROR = 255
} ProfStatus;

typedef struct {
    char* buffer;
    size_t capacity;
} ProfErrorBuffer;

typedef struct {
    const char* c_path;
    const char* h_path;
} ProfModulePath;
```

## 4. profiler 生成接口草案

### 4.1 模块输出路径参数

```c
typedef struct {
    ProfModulePath tokenizer;
    ProfModulePath network_init;
    ProfModulePath weights_load;
    ProfModulePath train;
    ProfModulePath weights_save;
    ProfModulePath infer;
    const char* metadata_path;
} ProfOutputLayout;
```

### 4.2 生成请求与返回

```c
typedef struct {
    const NN_NetworkDef* network_def;
    ProfOutputLayout output_layout;
    ProfErrorBuffer error;
} ProfGenerateRequest;

typedef struct {
    uint64_t network_hash;
    const char* metadata_written_path;
} ProfGenerateResult;
```

### 4.3 生成函数

```c
ProfStatus profiler_generate(
    const ProfGenerateRequest* req,
    ProfGenerateResult* out_result
);
```

### 4.4 约束

- `req->network_def` 是唯一结构输入源，不使用回调注册。
- `error.capacity` 建议不小于 256。
- 失败时必须：
  - 返回非 0 错误码
  - 向终端输出英文错误
  - 向 `error.buffer` 写入英文错误
- 首个可检测错误出现后立即停止。

## 5. 生成模块对外接口草案

以下接口定义为“固定头文件模板中的声明形态”，网络变化时签名不变。

## 5.1 tokenizer

```c
ProfStatus nn_tokenizer_encode(
    const NNTokenizerInput* input,
    NNTokenizerOutput* output,
    ProfErrorBuffer* error
);
```

## 5.2 网络初始化

```c
ProfStatus nn_runtime_init(
    NNRuntimeContext* ctx,
    const NNInitConfig* config,
    ProfErrorBuffer* error
);
```

## 5.3 权重保存 / 加载

```c
ProfStatus nn_weights_save(
    const NNRuntimeContext* ctx,
    const char* weight_file_path,
    ProfErrorBuffer* error
);

ProfStatus nn_weights_load(
    NNRuntimeContext* ctx,
    const char* weight_file_path,
    ProfErrorBuffer* error
);
```

加载接口强制行为：

- 每次读取文件都验证文件内 Hash 与加载模块内 Hash 常量一致
- Hash 不一致返回 `PROF_STATUS_HASH_MISMATCH`
- 不允许继续装载不同网络权重

### 5.3.1 子网保存/加载注册接口（编译期）

```c
typedef struct {
    const char* network_type;
    ProfStatus (*save_serialize)(const void* train_ctx, NNSerializer* writer, ProfErrorBuffer* error);
    ProfStatus (*load_deserialize)(void* infer_ctx, NNDeserializer* reader, ProfErrorBuffer* error);
} NNSubnetSerdeVTable;

ProfStatus nn_register_subnet_serde(const NNSubnetSerdeVTable* vtable);
```

约束：

- 注册用于编译期装配不同网络类型能力。
- 网络类型启用状态由 CMakeLists 开关控制。
- `save_serialize` 由训练实现提供。
- `load_deserialize` 由推理实现提供。
- profiler 通过注册表生成调用，不直接依赖具体子网实现细节。
- 新增网络类型仅通过注册接入，不改变现有对外接口签名。
- 新增网络类型时仅更新注册配置文件或注册宏清单，不修改 profiler 主流程代码。
- 未启用的网络类型不参与编译与注册。

## 5.4 训练接口（双模式）

```c
ProfStatus nn_train_run_auto(
    NNTrainContext* ctx,
    const NNTrainRunConfig* config,
    NNTrainRunReport* report,
    ProfErrorBuffer* error
);

ProfStatus nn_train_step(
    NNTrainContext* ctx,
    const NNTrainStepInput* step_in,
    NNTrainStepOutput* step_out,
    ProfErrorBuffer* error
);
```

继续训练时强制校验：

- 结构签名 Hash 一致
- 参数布局摘要一致
- 任一不一致时拒绝继续训练并返回错误码

## 5.5 推理接口（双模式）

```c
ProfStatus nn_infer_run_auto(
    NNInferContext* ctx,
    const NNInferRunConfig* config,
    NNInferRunReport* report,
    ProfErrorBuffer* error
);

ProfStatus nn_infer_step(
    NNInferContext* ctx,
    const NNInferStepInput* step_in,
    NNInferStepOutput* step_out,
    ProfErrorBuffer* error
);
```

## 6. 元数据接口草案

```c
typedef struct {
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t abi_version;
} NNGeneratedMetadata;

ProfStatus nn_metadata_load(
    const char* metadata_path,
    NNGeneratedMetadata* out_meta,
    ProfErrorBuffer* error
);
```

元数据一致性规则：

- 训练/推理/保存/加载必须消费同一份元数据
- 元数据不一致按 `PROF_STATUS_LAYOUT_MISMATCH` 处理

## 7. 调用流程草案

1. 用户在 CMakeLists 通过开关启用网络类型。
2. 第一次编译用户程序与 profiler 所需组件，仅启用类型参与编译。
3. 用户程序构建 `NN_NetworkDef` 并填充 `ProfOutputLayout`。
4. 用户程序运行时调用 `profiler_generate`。
5. profiler 校验网络并生成训练/推理 `.c`，复制固定 `.h`，写出元数据。
6. 第二次编译训练工程，训练工程依赖“推理 `.c` + 训练 `.c`”。
7. 训练运行后生成权重 `.bin` 与权重 `.c`。
8. 第三次编译推理工程，推理工程仅依赖推理 `.c`。
9. 推理运行时通过 `nn_weights_load` 加载 `.bin` 或直接链接权重 `.c`。
10. 根据场景使用 `*_run_auto` 或 `*_step`。

## 8. C99 约束

- 接口类型与实现必须可在 C99 编译模式下通过。
- 不使用超出 C99 的语言特性。
- 固定头文件模板与生成 `.c` 一并遵守 C99。
- 源码与生成代码的注释量必须不低于 50%（必须遵守）。
- 编译阶段必须启用最严格校验，并将所有警告提升为错误，不允许警告通过。

## 9. 设计决策原则

- 问题处理优先从流程、过程、结构层面解决，不以数据补丁作为主方案。
- 流程正确性由机制保证，数据验证仅用于校验数据是否符合流程，不用于验证流程本身。
- 存在设计取舍或不合理方案风险时，先请示用户再执行。
- 扩展新网络类型优先通过统一配置和注册流程，不通过改动旧代码路径实现兼容。
