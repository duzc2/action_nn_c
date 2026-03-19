# profiler API 合同草案（字段与错误码）

## 1. 文档目标

本草案用于补充 `profiler_interface_draft.md`，聚焦两类内容：

- 错误码清单（含触发条件、调用方处理建议）
- 接口参数字段表（必填/可选/默认行为）

本草案不改变既有接口签名，仅明确调用合同。

## 2. 错误码清单

| 错误码 | 名称 | 典型触发条件 | 调用方处理建议 |
|---|---|---|---|
| 0 | PROF_STATUS_OK | 成功 | 继续后续流程 |
| 1 | PROF_STATUS_INVALID_ARGUMENT | 空指针、非法长度、缺失必填字段 | 修正参数后重试 |
| 2 | PROF_STATUS_VALIDATION_FAILED | 网络定义校验失败（非环路类） | 修复网络描述并重试 |
| 3 | PROF_STATUS_CYCLE_DETECTED | 拓扑存在环路 | 修复连接关系并重试 |
| 4 | PROF_STATUS_PATH_INVALID | 输出路径无效、不可写、冲突 | 修正路径映射后重试 |
| 5 | PROF_STATUS_IO_FAILED | 文件读写失败 | 检查权限/磁盘后重试 |
| 6 | PROF_STATUS_HASH_MISMATCH | 权重文件 Hash 与当前网络不一致 | 禁止加载，改用匹配权重 |
| 7 | PROF_STATUS_LAYOUT_MISMATCH | 元数据或参数布局不一致 | 禁止继续训练/加载，重新生成或匹配版本 |
| 255 | PROF_STATUS_INTERNAL_ERROR | 未分类内部错误 | 记录错误并中止 |

错误输出统一机制：

- 返回错误码
- 终端输出纯英文错误描述
- 写入 `ProfErrorBuffer.buffer`（纯英文）
- 首个错误快速失败，不继续后续步骤

## 3. 字段合同：核心结构体

## 3.1 ProfErrorBuffer

| 字段 | 类型 | 必填 | 约束 | 默认行为 |
|---|---|---|---|---|
| buffer | char* | 是 | 可写内存，非空 | 为空则返回 INVALID_ARGUMENT |
| capacity | size_t | 是 | 建议 >= 256 | 小于建议值可继续，但可能截断错误文本 |

## 3.2 ProfModulePath

| 字段 | 类型 | 必填 | 约束 | 默认行为 |
|---|---|---|---|---|
| c_path | const char* | 是 | 有效可写目标路径 | 为空返回 PATH_INVALID |
| h_path | const char* | 是 | 有效可写目标路径 | 为空返回 PATH_INVALID |

## 3.3 ProfOutputLayout

| 字段 | 类型 | 必填 | 约束 | 默认行为 |
|---|---|---|---|---|
| tokenizer | ProfModulePath | 是 | 指向 tokenizer 产物目录/文件 | 缺失返回 PATH_INVALID |
| network_init | ProfModulePath | 是 | 指向 init 产物目录/文件 | 缺失返回 PATH_INVALID |
| weights_load | ProfModulePath | 是 | 指向 load 产物目录/文件 | 缺失返回 PATH_INVALID |
| train | ProfModulePath | 是 | 指向 train 产物目录/文件 | 缺失返回 PATH_INVALID |
| weights_save | ProfModulePath | 是 | 指向 save 产物目录/文件 | 缺失返回 PATH_INVALID |
| infer | ProfModulePath | 是 | 指向 infer 产物目录/文件 | 缺失返回 PATH_INVALID |
| metadata_path | const char* | 是 | 有效可写路径 | 为空返回 PATH_INVALID |

说明：

- `.c/.h` 支持分开存储
- 生成代码路径由该结构体显式指定，不使用隐式固定目录

## 3.4 ProfGenerateRequest

| 字段 | 类型 | 必填 | 约束 | 默认行为 |
|---|---|---|---|---|
| network_def | const NN_NetworkDef* | 是 | 非空、结构完整 | 为空返回 INVALID_ARGUMENT |
| output_layout | ProfOutputLayout | 是 | 路径合同满足 3.3 | 任一路径无效返回 PATH_INVALID |
| error | ProfErrorBuffer | 是 | 满足 3.1 | 无效返回 INVALID_ARGUMENT |

## 3.5 ProfGenerateResult

| 字段 | 类型 | 必填 | 约束 | 默认行为 |
|---|---|---|---|---|
| network_hash | uint64_t | 输出 | 成功时返回有效 Hash | 失败时未定义 |
| metadata_written_path | const char* | 输出 | 成功时返回元数据输出路径 | 失败时为 NULL 或未定义 |

## 4. 字段合同：运行时接口参数

## 4.1 nn_weights_load

| 参数 | 必填 | 约束 | 默认行为 |
|---|---|---|---|
| ctx | 是 | 已初始化运行时上下文 | 无效返回 INVALID_ARGUMENT |
| weight_file_path | 是 | 可读文件路径 | 无效返回 IO_FAILED |
| error | 是 | 有效错误缓冲区 | 无效返回 INVALID_ARGUMENT |

附加规则：

- 必须校验文件内 Hash 与加载模块常量 Hash
- 不一致返回 `PROF_STATUS_HASH_MISMATCH`

## 4.2 nn_train_run_auto / nn_train_step

| 参数 | 必填 | 约束 | 默认行为 |
|---|---|---|---|
| ctx | 是 | 已初始化训练上下文 | 无效返回 INVALID_ARGUMENT |
| config / step_in | 是 | 满足训练输入约束 | 无效返回 INVALID_ARGUMENT |
| report / step_out | 是 | 可写输出对象 | 无效返回 INVALID_ARGUMENT |
| error | 是 | 有效错误缓冲区 | 无效返回 INVALID_ARGUMENT |

附加规则：

- 继续训练必须校验结构签名与布局摘要
- 不一致返回 `PROF_STATUS_LAYOUT_MISMATCH`

## 4.3 nn_infer_run_auto / nn_infer_step

| 参数 | 必填 | 约束 | 默认行为 |
|---|---|---|---|
| ctx | 是 | 已初始化推理上下文 | 无效返回 INVALID_ARGUMENT |
| config / step_in | 是 | 满足推理输入约束 | 无效返回 INVALID_ARGUMENT |
| report / step_out | 是 | 可写输出对象 | 无效返回 INVALID_ARGUMENT |
| error | 是 | 有效错误缓冲区 | 无效返回 INVALID_ARGUMENT |

## 5. 生成阶段前置检查顺序

1. 请求对象与指针有效性检查
2. 输出路径映射检查（模块级）
3. 网络定义一致性校验（含 DAG、连接策略、展开顺序）
4. 网络 Hash 与布局映射生成
5. 代码与元数据输出

在任一步骤出现首个错误即停止并返回。

## 6. 最低可用调用模板

```c
ProfGenerateRequest req;
ProfGenerateResult result;
ProfStatus st = profiler_generate(&req, &result);
if (st != PROF_STATUS_OK) {
    /* 终端已有英文错误，同时可读取 req.error.buffer */
}
```

## 7. 与现有文档关系

- 接口签名来源：`profiler_interface_draft.md`
- 需求与约束来源：`network_topology_requirements.md`
- 模块数据流来源：`profiler_module_blueprint.md`

## 8. 注释比例约束

- 源码与 profiler 生成代码都必须满足注释比例超过 50%。
