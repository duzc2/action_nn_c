# action_c 静态代码扫描报告

**扫描日期**: 2026-05-12
**扫描范围**: 全工程（src/、demo/、wasm/、web_editor/、构建系统）
**扫描标准**: MISRA C、CERT C、OWASP Top 10、ISO/IEC 25010
**代码语言**: C (C11, ~90%)、JavaScript (~5%)、CMake、HTML/CSS、TypeScript (<1%)

---

## 一、总览

| 维度 | Critical | High | Medium | Low | 合计 |
|------|----------|------|--------|-----|------|
| 安全性 (Security) | 3 | 5 | 11 | 6 | 25 |
| 稳定性 (Stability) | 1 | 7 | 14 | 8 | 30 |
| 性能 (Performance) | 4 | 6 | 17 | 7 | 34 |
| 功能性 (Functionality) | 2 | 6 | 10 | 12 | 30 |
| 易用性 (Usability) | 0 | 1 | 5 | 14 | 20 |
| 可维护性 (Maintainability) | 0 | 3 | 6 | 8 | 17 |
| **合计** | **10** | **28** | **63** | **55** | **156** |

---

## 二、Critical 级别问题（10项）

### 2.1 安全性 - Critical (3项)

#### [SEC-C01] mlp_dense_create 整数溢出漏洞
- **文件**: `src/nn/types/mlp/mlp_layers.c:266`
- **详情**: `input_size * output_size * sizeof(float)` 未经溢出检查。当 `input_size` 和 `output_size` 都很大时，乘法溢出导致分配缓冲区不足，后续写入造成堆缓冲区溢出。CERT INT32-C。
- **修复**: 使用 `SIZE_MAX / sizeof(float) / input_size < output_size` 预先检查。

#### [SEC-C02] mlp_load_weights 缺少 abi_version 校验
- **文件**: `src/nn/types/mlp/mlp_infer_ops.c:391-495`
- **详情**: 权重文件加载时未校验 `abi_version` 字段。若文件格式变更，旧格式权重可能被错误加载导致静默数据损坏。Transformer/RCNN 后端有此校验，但 MLP 缺失。
- **修复**: 添加 `if (header.abi_version != EXPECTED_ABI_VERSION) return -1;`。

#### [SEC-C03] 所有 nn_type_*_train.c 中不兼容的函数指针转换
- **文件**: `src/nn/types/*/nn_type_*_train.c` (6个文件)
- **详情**: 将 `int (*)(MlpTrainContext*, const float*, const float*)` 强制转换为 `int (*)(void*, const void*, const void*)` 后通过函数指针调用，属于未定义行为（CERT EXP37-C）。虽然在常见 ABI 上可行，但不符合 C 标准。
- **修复**: 将具体类型的函数改为接受 `void*` 参数并在内部转换，或使用 union 类型安全分发。

### 2.2 稳定性 - Critical (1项)

#### [STB-C01] wasm CMakeLists.txt 导出不存在函数
- **文件**: `wasm/CMakeLists.txt:83-97`
- **详情**: CMakeLists.txt 声明导出的 `action_c_wasm_infer_create`、`action_c_wasm_infer_run` 在 `wasm_exports.c` 中不存在。训练接口同样不匹配。构建将因链接错误失败。
- **修复**: 同步 CMakeLists.txt 导出列表与 `wasm_exports.c` 中实际实现的函数。

### 2.3 性能 - Critical (4项)

#### [PRF-C01] prof_codegen_generate_all 重复构建叶子图 7 次
- **文件**: `src/profiler/prof_codegen.c:2308-2321`
- **详情**: 每个代码生成模块（metadata、tokenizer、network_init、infer、train、weights_save、weights_load）各自独立构建 `ProfLeafGraph`。这意味着整个子网树被递归遍历和拓扑排序 7 次。
- **修复**: 计算一次叶子图并在所有模块间共享。

#### [PRF-C02] mlp_train_ops 在每步训练中分配堆内存
- **文件**: `src/nn/types/mlp/mlp_train_ops.c:400-401`
- **详情**: `train_backward_pass` 在每个训练步中用 `calloc` 分配 `current_delta` 和 `next_delta` 两个堆缓冲区，用完立即释放。对于小网络，堆分配开销将主导训练时间。
- **修复**: 在训练上下文中预分配并复用这些缓冲区。

#### [PRF-C03] gnn_infer_ops 每次推理执行 3 次堆分配
- **文件**: `src/nn/types/gnn/gnn_infer_ops.c:573-583, 665`
- **详情**: 每次前向传播调用 `calloc` 分配 `owned_cache`、`aggregated`、`pooled_hidden`，用完立即释放。每个推理步产生 3 次 `calloc` + 3 次 `free`。
- **修复**: 在推理上下文中预分配 scratch buffer。

#### [PRF-C04] gnn_train_ops 每次训练步执行 6 次堆分配
- **文件**: `src/nn/types/gnn/gnn_train_ops.c:300-323`
- **详情**: `gnn_backpropagate` 在每个训练步中分配 `stage_grad`、`previous_stage_grad`、`aggregated`、`aggregated_grad`、`pooled_hidden`、`pooled_grad`。
- **修复**: 在训练上下文中预分配所有 scratch buffer。

### 2.4 功能性 - Critical (2项)

#### [FUN-C01] mlp_train_ops 中 layout_hash 与 network_hash 混淆（复制粘贴错误）
- **文件**: `src/nn/types/mlp/mlp_train_ops.c:917`
- **详情**: 保存检查点时，`header.layout_hash = nn_mlp_get_network_hash(infer_ctx)` 对 `network_hash` 和 `layout_hash` 使用了相同的函数。这意味着 layout_hash 永远等于 network_hash，失去了独立验证的意义。
- **修复**: `header.layout_hash` 应调用 `compute_layout_hash`。

#### [FUN-C02] wasm build_wasm.sh 使用字面量 \$EMSDK 导致构建失败
- **文件**: `wasm/scripts/build_wasm.sh:130`
- **详情**: `-DCMAKE_TOOLCHAIN_FILE=\$EMSDK/...` 中的 `\$` 在双引号内被视为转义，传给 CMake 的是字面量 `$EMSDK` 字符串而不是环境变量展开值。CMake 配置将失败。
- **修复**: 使用 `"${EMSDK}/upstream/emscripten/..."` (不在双引号内转义) 或 `${EMSDK}`。

---

## 三、High 级别问题（28项）

### 3.1 安全性 - High (5项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| SEC-H01 | 全部 CMakeLists.txt | 全局 | 缺少安全加固编译标志：`-D_FORTIFY_SOURCE=2`、`-fstack-protector-strong`、`-fPIE`/`-pie`、`-Wl,-z,relro,-z,now` |
| SEC-H02 | `demo/cs/tools/CMakeLists.txt` | 41行 | 硬编码 VS 2022 Community 安装路径 |
| SEC-H03 | `.codex_build_cnn_rnn_infer.cmd` | 全部 | 硬编码 VS 2022 Community 路径 |
| SEC-H04 | `wasm/scripts/build_wasm.sh` | 130行 | 环境变量注入风险：EMSDK 路径未经校验 |
| SEC-H05 | `web_editor/vite.config.js` | 6行 | `host: true` 将开发服务器暴露到所有网络接口 |

### 3.2 稳定性 - High (7项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| STB-H01 | `src/nn/CMakeLists.txt` | 9-13行 | `file(GLOB)` 收集源文件，不会自动检测新文件（CMake 反模式） |
| STB-H02 | `demo/demo_common.cmake` + `src/CMakeLists.txt` | 各处 | `action_c_apply_strict_warnings` 函数定义重复，CMake 产生策略警告 |
| STB-H03 | `demo/demo_common.cmake` | 42行 | `action_c_get_demo_generated_dir` 使用 `../data` 相对路径，非标准构建目录下失效 |
| STB-H04 | demo 各 `infer/`/`train/` CMakeLists.txt | 全局 | 生成的文件引用无 CMake 级依赖约束 (generate 必须先运行) |
| STB-H05 | demo 各 `run_demo.sh` | 全局 | `.sh` 脚本硬编码 `.exe` 扩展名，在 Linux 上不可用 |
| STB-H06 | `build_demos.ps1` | 31-33行 | 在构建目录内执行源内风格构建 (Remove-Item CMakeCache.txt) |
| STB-H07 | `demo/sevenseg/train_main.c` | 82-84行 | `train_step()` 和 `infer_auto_run()` 的返回值完全未检查 |

### 3.3 性能 - High (6项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| PRF-H01 | `src/profiler/prof_flatten.c` | 39-46行 | O(n^2) realloc 增长 (每次追加一个元素) |
| PRF-H02 | `src/profiler/prof_flatten.c` | 302-328行 | Kahn 算法 O(V*E) 而非 O(V+E)，未构建邻接表 |
| PRF-H03 | `src/profiler/prof_flatten.c` | 267-268行 | 每次连接遍历都做 O(L) 线性字符串扫描查找子网索引 |
| PRF-H04 | `src/nn/types/cnn/cnn_infer_ops.c` | 342-418行 | 卷积 6 层嵌套循环，无 tiling、无缓存分块、无 SIMD |
| PRF-H05 | `src/nn/types/cnn/cnn_train_ops.c` | 200-233行 | 反向传播 7 层嵌套循环 |
| PRF-H06 | `src/nn/types/transformer/transformer_infer_ops.c` | 133行 | `transformer_forward_cache_init` 每次前向传播分配 12 个堆缓冲区 |

### 3.4 功能性 - High (6项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| FUN-H01 | `web_editor/js/editor.js` 等 | 4个文件 (1306行) | Rete.js 1.x 遗留代码，未被任何页面加载 |
| FUN-H02 | `web_editor/package.json` | 全局 | 缺少 `package-lock.json`，依赖版本不可复现 |
| FUN-H03 | `web_editor/src/main.js` | 477-480行 | 连接生成的 C 代码仅为注释，未实现连接创建 |
| FUN-H04 | `wasm/js/action_nn_c.d.ts` | 24-32行 | TypeScript 声明引用 CMakeLists.txt 中不存在函数的类型接口 |
| FUN-H05 | `demo/cs/tools/CMakeLists.txt` | 38-53行 | `cs_runtime_probe32` 仅 Windows 构建，`cs_state_trace` 无法跨平台 |
| FUN-H06 | `web_editor/start.sh` | 27行 | 绕过 Vite 使用原始 Python HTTP 服务器，README 文档过时 |

### 3.5 易用性 - High (1项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| USA-H01 | `wasm/CMakeLists.txt` | 83-97行 | 导出的 API 与 C 源文件中实现的 API 不匹配，用户调用将失败 |

### 3.6 可维护性 - High (3项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| MNT-H01 | `src/nn/types/cnn/` vs `cnn_dual_pool/` | 全局 | ~90% 代码重复，仅 pooling 缓存不同 |
| MNT-H02 | `src/nn/types/transformer/` | train vs infer | Transformer 训练前向传播与推理前向传播高度重复 |
| MNT-H03 | `demo/mnist/` vs `demo/mnist_cnn/` | dataset.c | MNIST IDX 文件读取代码 ~95% 重复 |

---

## 四、Medium 级别问题（63项）

### 4.1 安全性 - Medium (11项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| SEC-M01 | `src/nn/nn_train_registry.c` | copy_type_name 函数 | `strlen(source)` 无长度上限 (CERT STR31-C) |
| SEC-M02 | 所有 `nn_type_*_*.c` | 全局 | `strcmp(config->type_config_type_name, ...)` 假设字符串以 NULL 结尾 |
| SEC-M03 | `src/nn/types/mlp/mlp_infer_ops.c` | 404-411行 | `fread` 返回值 (size_t) 被转为 int 比较，如有大值则截断 (CERT INT31-C) |
| SEC-M04 | `src/nn/types/transformer/transformer_infer_ops.c` | 701-708行 | uint64_t → size_t 截断风险 (32位平台上) |
| SEC-M05 | `src/nn/types/gnn/gnn_config.h` | 18行 | `GNN_FEATURE_INDEX_NONE = SIZE_MAX` 与其他值比较时无符号溢出风险 |
| SEC-M06 | `wasm/src/wasm_exports.c` | 92-98行 | Wasm 内存指针无大小验证，JS 可传入任意指针导致越界读 |
| SEC-M07 | `wasm/examples/browser_demo.html` | 无 | 无 CSP 头 (Content-Security-Policy) |
| SEC-M08 | `src/nn/nn_train_registry.c` | 85行 | bootstrap 在部分注册失败后继续，产生不一致状态 |
| SEC-M09 | `src/nn/types/rnn/rnn_infer_ops.c` | 296-310行 | RNN 只使用最终隐藏状态，无注意力聚合 (功能风险) |
| SEC-M10 | `wasm/config/wasm_config.h` | 155-178行 | 网络类型字符串未验证长度边界 |
| SEC-M11 | `src/profiler/prof_codegen.c` | 100-102行 | `fprintf` 和 `fclose` 返回值被丢弃，写失败静默忽略 |

### 4.2 稳定性 - Medium (14项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| STB-M01 | `src/profiler/prof_flatten.c` | 39-42行 | realloc 失败时原始指针泄漏 |
| STB-M02 | `src/profiler/prof_path.c` | 75行 | 512 字节栈缓冲区可能截断长路径 |
| STB-M03 | `src/profiler/prof_codegen.c` | 17行 | `CODE_BUFFER_CAPACITY` 硬编码 512KB，大网络可能不足 |
| STB-M04 | `src/profiler/prof_codegen.c` | 524, 554, 1774行 | snprintf 截断导致符号名冲突 |
| STB-M05 | `src/nn/nn_train_registry.c` | 170-171行 | 部分注册失败后状态不一致 |
| STB-M06 | `src/nn/nn_graph_*_contract.c` | 67-99行 | 注册表清除后契约缓存未失效 |
| STB-M07 | `src/nn/types/mlp/mlp_train_ops.c` | 645-653行 | 销毁时根据 config.optimizer 决定清理逻辑，但 optimizer 可能在构造后被修改 |
| STB-M08 | `src/nn/types/mlp/mlp_train_ops.c` | 1083-1084行 | 训练步分配全零虚拟输入/目标缓冲区 (无意义训练) |
| STB-M09 | `wasm/src/wasm_exports.c` | 108-116行 | `infer_destroy` 只调用 `free()`，泄漏内部资源 |
| STB-M10 | `wasm/examples/browser_demo.html` | 214行 | JavaScript 运算符优先级错误：`1024.toFixed(2)` 先求值 |
| STB-M11 | `demo/demo_common.cmake` | 3行 | 使用已弃用的 `include(CMakeParseArguments)` |
| STB-M12 | `demo/cnn_rnn_react/generate/CMakeLists.txt` | 14-16行 | 硬编码 `Debug/` 输出目录 |
| STB-M13 | `build_demos.ps1` | 33行 | `Out-Null` 抑制 CMake 配置错误输出 |
| STB-M14 | `wasm/scripts/build_wasm.sh` | 117行 | 构建脚本输出目录与文档推荐目录不一致 |

### 4.3 性能 - Medium (17项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| PRF-M01 | `src/profiler/prof_validate.c` | 82-107行 | O(n^2) 重复 ID 检测 |
| PRF-M02 | `src/profiler/prof_validate.c` | 330-341行 | `prof_validate_network_def` 内 flatten 执行两次 |
| PRF-M03 | `src/profiler/prof_validate.c` | 330-525行 | 同一验证流程中 flatten 执行三次 |
| PRF-M04 | `src/profiler/prof_hash.c` | 55-57行 | `sizeof(size_t)` 使哈希平台依赖 (32/64位) |
| PRF-M05 | `src/profiler/prof_codegen.c` | 680行 | tokenizer 仅 4096 字节栈缓冲区 |
| PRF-M06 | `src/profiler/prof_codegen.c` | 789行 | network_init 仅 8192 字节栈缓冲区 |
| PRF-M07 | `src/nn/nn_train_registry.c` + `nn_infer_registry.c` | 70-85行 | 注册时两次线性扫描 |
| PRF-M08 | `src/nn/nn_graph_*_contract.c` | 67-99行 | 每次查找调用 bootstrap (已有 latch 但在外层重复检查) |
| PRF-M09 | `src/nn/types/mlp/mlp_layers.c` | 32行 | pcg_rand 使用 64位乘法但结果截断为 32位 |
| PRF-M10 | `src/nn/types/mlp/mlp_layers.c` | 99-161行 | 激活函数用 switch 分发 (热路径分支重) |
| PRF-M11 | `src/nn/types/transformer/transformer_train_ops.c` | 336-341行 | 训练步分配 3 个堆缓冲区 |
| PRF-M12 | `src/nn/types/transformer/transformer_train_ops.c` | 429行 | 训练步分配 output_cache |
| PRF-M13 | `src/nn/types/rnn/rnn_train_ops.c` | 163-164行 | BPTT 用 memcpy 交换隐藏状态 (应使用指针交换) |
| PRF-M14 | `src/nn/types/gnn/gnn_infer_ops.c` | 567行 | 每次前向传播重新验证配置 |
| PRF-M15 | `src/nn/` 全部 .c 文件 | 全局 | 所有张量操作为标量循环，无 SIMD/向量化 |
| PRF-M16 | 全部 CMakeLists.txt | 全局 | 无 LTO (-flto) 配置 |
| PRF-M17 | 全部 CMakeLists.txt | 全局 | 无显式 `--parallel` 编译并行配置 |

### 4.4 功能性 - Medium (10项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| FUN-M01 | `src/nn/types/transformer/transformer_train_ops.c` | 350行 | 训练损失计算中 logf 参数可能下溢为 -inf |
| FUN-M02 | `src/nn/types/transformer/transformer_train_ops.c` | 393行 | 硬编码 0.85/0.15 移动平均，非标准参数更新规则 |
| FUN-M03 | `src/nn/types/transformer/transformer_train_ops.c` | 444行 | 梯度计算假设 tanh 激活但未验证 |
| FUN-M04 | `src/nn/types/gnn/gnn_infer_ops.c` | 665-670行 | `pooled_hidden` 仅在一个分支分配，另一个分支 free(NULL) 脆弱 |
| FUN-M05 | `src/nn/types/gnn/gnn_infer_ops.c` | 263-296行 | `gnn_find_anchor_node` 使用硬编码 -1000000.0f 哨兵值 |
| FUN-M06 | `src/nn/types/gnn/gnn_infer_ops.c` | 775-830行 | 权重加载未校验 network_hash |
| FUN-M07 | `demo/cs/tools/cs_dataset_build.c` | 689,815,977行 | JSON 解析使用硬编码游标偏移量 (cursor += N) |
| FUN-M08 | `wasm/src/wasm_exports.c` | 239-247行 | profiler_create 返回 NULL，profiler_destroy 无操作 |
| FUN-M09 | `wasm/config/wasm_config.h` | 98-99行 | `WASM_EXPORT_NAME` 宏接收参数但不使用 |
| FUN-M10 | `wasm/src/wasm_exports.c` | 36行 | `get_version_string` 返回硬编码字符串，与 CMake 定义不同步 |

### 4.5 易用性 - Medium (5项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| USA-M01 | `src/nn/` 全部 .c 文件 | 全局 | 无错误日志基础设施，失败时静默返回错误码 |
| USA-M02 | `demo/` 全部 demo CMakeLists.txt | 全局 | `CMAKE_C_EXTENSIONS OFF` 不一致 |
| USA-M03 | `wasm/CMakeLists.txt` | 75-89行 | 使用已弃用的 `--export-file` Emscripten 语法 |
| USA-M04 | `wasm/CMakeLists.txt` | 166行 | 版本号硬编码在两处 (CMakeLists.txt + wasm_config.h) |
| USA-M05 | `web_editor/src/main.js` | 138-259行 | 绕过 Rete.js 渲染系统直接操作 DOM (50ms setTimeout) |

### 4.6 可维护性 - Medium (6项)

| ID | 文件 | 位置 | 描述 |
|----|------|------|------|
| MNT-M01 | `src/nn/` 多个文件 | copy_type_name | 同一函数在 4 个文件中完全重复 |
| MNT-M02 | `src/nn/types/*/nn_type_*_train.c` | 全局 | SGD 更新函数在所有后端中结构相同 |
| MNT-M03 | `src/nn/` 全部 .c 文件 | 全局 | 错误返回值不一致 (-1, 0, 1 混合) |
| MNT-M04 | `demo/` 全部 run_demo.bat/sh | 全局 | 脚本高度重复，应参数化 |
| MNT-M05 | `src/profiler/prof_validate.c` | 全局 | flatten 调用在多个验证阶段重复 |
| MNT-M06 | `demo/cs/tools/cs_tool_common.h` | 47-63行 | 遗留 CsLabelSegment 结构体被标记为废弃但未删除 |

---

## 五、Low 级别问题（55项）

### 5.1 安全性 - Low (6项)

| ID | 描述 |
|----|------|
| SEC-L01 | `demo/move/infer_main.c:86` scanf 返回值未验证 |
| SEC-L02 | `demo/mnist/*.c` `demo/mnist_cnn/*.c` 硬编码 4 级深层相对路径 |
| SEC-L03 | `demo/cs/tools/*.c` 硬编码 Windows 风格反斜杠路径 |
| SEC-L04 | `web_editor/src/main.js:519` innerHTML = content.replace() 仅转义 `<>`，未转义 `&"'` |
| SEC-L05 | `web_editor/index.html` 无 CSP 头 |
| SEC-L06 | `demo/cs/tools/CMakeLists.txt` vcvars32.bat 路径如被篡改有供应链风险 |

### 5.2 稳定性 - Low (8项)

| ID | 描述 |
|----|------|
| STB-L01 | `src/profiler/profiler.c:58-59` hash 函数接受 NULL 并静默返回默认值 |
| STB-L02 | `src/profiler/profiler.c:50-55,66-71` error.buffer 空检查冗余 (prof_error_set 内部已做) |
| STB-L03 | `src/profiler/profiler.c:74-77` out_result 仅在成功路径部分写入 |
| STB-L04 | `src/profiler/prof_flatten.c:57-90` 无界递归深度 |
| STB-L05 | `src/nn/types/mlp/mlp_layers.c:139` softmax 未检查 size > 0 |
| STB-L06 | `demo/demo_runtime_paths.h:54-57` POSIX 上 getcwd 返回当前目录而非可执行目录 |
| STB-L07 | `wasm/src/wasm_exports.c:72,260` extern 声明在 `#ifdef __EMSCRIPTEN__` 外 |
| STB-L08 | `wasm/examples/browser_demo.html:167-170` cwrap 失败无 try-catch |

### 5.3 性能 - Low (7项)

| ID | 描述 |
|----|------|
| PRF-L01 | `src/profiler/profiler.c:46` 不必要的 const 转换 |
| PRF-L02 | `src/profiler/prof_hash.c:45-50` hash_string 每次重算 strlen |
| PRF-L03 | `src/profiler/prof_error.c:91` 512字节栈缓冲区在冷路径上分配 |
| PRF-L04 | `src/nn/types/mlp/mlp_infer_ops.c:360` 乒乓缓冲区交换使用三元表达式 |
| PRF-L05 | `src/nn/types/mlp/mlp_train_ops.c:422` calloc 零初始化后被 memset 重复清零 |
| PRF-L06 | `src/nn/types/cnn/cnn_infer_ops.c:392-398` 深层循环内条件缓存写入 |
| PRF-L07 | `wasm/config/wasm_config.h:59` 初始 64MB Wasm 内存对移动端可能过大 |

### 5.4 功能性 - Low (12项)

| ID | 描述 |
|----|------|
| FUN-L01 | `src/profiler/prof_hash.c:93-106` NULL overrides 且 count > 0 无检查 |
| FUN-L02 | `src/profiler/prof_validate.c:417-419` NULL network 返回 OK (可被直接调用绕过) |
| FUN-L03 | `src/nn/types/mlp/mlp_infer_ops.c:64-66` nn_mlp_infer_create 总是返回 NULL |
| FUN-L04 | `src/nn/types/mlp/mlp_layers.c:178-183` 权重/偏置为 NULL 时填充零输出但不报错 |
| FUN-L05 | `src/nn/types/gnn/gnn_config.h` neighbor_index 使用负哨兵值但不检查特定常量 |
| FUN-L06 | `demo/cnn_rnn_react/cnn_rnn_react_scene.c:828-849` describe_move 只有 2 个独特输出 |
| FUN-L07 | `web_editor/src/main.js:294-308` 连接遍历可能读取不到连接 (Rete.js 版本差异) |
| FUN-L08 | `web_editor/src/main.js:80` select 初始值未验证是否在选项列表中 |
| FUN-L09 | `wasm/CMakeLists.txt:115` EXPORT_ES6=0 与 .d.ts 的 ES 模块声明矛盾 |
| FUN-L10 | `wasm/js/action_nn_c.js:61-65` 抛出同步 Error 在期望 Promise 的场景中不可捕获 |
| FUN-L11 | `src/nn/types/cnn_dual_pool/` 90% 代码与 CNN 重复 |
| FUN-L12 | `src/nn/types/rnn/rnn_infer_ops.c:262` memset 清空隐藏状态 (无有状态 RNN 选项) |

### 5.5 易用性 - Low (14项)

| ID | 描述 |
|----|------|
| USA-L01 | `src/nn/types/*/nn_type_*` bridge 文件用 `0` 替代 `NULL` |
| USA-L02 | `demo/cnn_rnn_react_scene.c:535` "????????" 占位符注释 |
| USA-L03 | `demo/nested_nav/nested_nav_scene.h:536` 相同 "????????" 注释 |
| USA-L04 | `src/nn/types/mlp/mlp_infer_ops.c *` nn_mlp_infer_create 返回 NULL 令人困惑 |
| USA-L05 | `web_editor/index.html` 按钮无 aria-label/role |
| USA-L06 | `web_editor/vite.config.js` 无 `base` 路径配置 |
| USA-L07 | `web_editor/package.json` 无 `engines` 字段 |
| USA-L08 | `wasm/config/wasm_config.h:98-99` `WASM_EXPORT_NAME` 宏未使用 |
| USA-L09 | `.gitignore` 包含 Markdown 代码围栏标记 |
| USA-L10 | `src/nn/` 全部 .c 文件 无 assert() 调试辅助 |
| USA-L11 | demo CMakeLists.txt 使用固定目录深度 include |
| USA-L12 | `web_editor/` 无 `package-lock.json` 导致环境不一致 |
| USA-L13 | `wasm/js/action_nn_c.js` 为构建产物占位符而非功能代码 |
| USA-L14 | `web_editor/vite.config.js` port:5173 冗余 (Vite 默认值) |

### 5.6 可维护性 - Low (8项)

| ID | 描述 |
|----|------|
| MNT-L01 | `src/profiler/prof_codegen.c:755,767` st 变量被中间代码分隔导致可读性差 |
| MNT-L02 | `src/nn/` 全部 .c 文件 缺少 restrict 修饰符 |
| MNT-L03 | `src/nn_types/*/` open-closed 原则：添加新类型需修改多个文件 |
| MNT-L04 | `demo/cs/tools/cs_tool_common.*` 遗留函数 write_label_segments/read_label_segments |
| MNT-L05 | `demo/mnist/dataset.c` 与 `demo/mnist_cnn/dataset.c` IDX 读取 95% 重复 |
| MNT-L06 | `demo/cnn_rnn_react/` 缺少 run_demo.sh |
| MNT-L07 | `demo/cnn_rnn_react_scene.c:973` 参数被 (void) 抑制未使用 |
| MNT-L08 | `wasm/js/action_nn_c.d.ts` 引用不存在的版本函数名 |

---

## 六、线程安全性分析

全局静态变量无同步保护：

| 位置 | 变量 | 风险 |
|------|------|------|
| `nn_train_registry.c` | `g_slots[32]`, `g_bootstrapped`, `g_bootstrap_failed` | 多线程并发注册/查询产生数据竞争 |
| `nn_infer_registry.c` | `g_slots[32]`, `g_bootstrapped` | 同上 |
| `nn_graph_train_contract.c` | `g_train_contract_slots[32]`, `g_train_contract_count` | 多线程并发缓存查找/写入产生数据竞争 |
| `nn_graph_infer_contract.c` | `g_infer_contract_slots[32]`, `g_infer_contract_count` | 同上 |

当前代码库中无 `pthread_mutex`、`atomic` 操作或任何同步原语。对于当前单线程使用场景可接受，但多线程使用时会导致未定义行为。

---

## 七、修复建议优先级

### P0 - 立即修复（Critical + High 安全性）

1. **[SEC-C01]** 修复 `mlp_dense_create` 整数溢出
2. **[SEC-C02]** 添加 MLP `abi_version` 校验
3. **[SEC-C03]** 修复所有不兼容的函数指针转换
4. **[SEC-H01]** 添加安全编译标志 (`-D_FORTIFY_SOURCE=2`, `-fstack-protector-strong`, `-fPIE`, `-Wl,-z,relro,-z,now`)
5. **[FUN-C02]** 修复 `build_wasm.sh` 中 `\$EMSDK` 字面量问题

### P1 - 短期内修复（Critical/High 性能 + 稳定性）

6. **[PRF-C01]** 重构 `prof_codegen_generate_all` 共享叶子图
7. **[PRF-C02]** 预分配 MLP 反传 delta 缓冲区
8. **[PRF-C03][PRF-C04]** 预分配 GNN scratch buffer
9. **[STB-C01]** 同步 Wasm CMakeLists.txt 导出函数名与实际实现
10. **[FUN-C01]** 修复 MLP `layout_hash` 复制粘贴错误
11. **[STB-H01]** 替换 `file(GLOB)` 为显式源文件列表
12. **[STB-H02]** 消除 `action_c_apply_strict_warnings` 重复定义

### P2 - 中期优化（Medium 级别）

13. 添加 LTO 编译配置
14. 引入线程安全保护 (或文档化单线程限制)
15. 删除 `web_editor/js/` 和 `web_editor/css/` 中的遗留代码
16. 统一错误返回值约定
17. 添加 `restrict` 修饰符到热路径指针参数
18. 为关键张量运算添加 SIMD 内联

### P3 - 长期改进（Low 级别 + 架构优化）

19. 合并 CNN 与 CNN Dual Pool 共享代码
20. 为所有 demo 构建脚本创建参数化版本
21. 添加 `package-lock.json`
22. 添加 Web Editor ARIA 无障碍标签
23. 清理 `.gitignore` Markdown 标记
24. 添加断言和运行时诊断日志

---

## 八、扫描方法说明

本报告通过以下方法生成：

1. **全量代码读取**: 读取所有 `.c`、`.h`、`.js`、`.ts`、`.html`、`.css`、`CMakeLists.txt`、构建脚本
2. **静态模式匹配**: 搜索不安全函数 (`strcpy`, `sprintf`, `scanf`, `gets` 等)、未使用变量、内存泄漏模式
3. **数据流分析**: 追踪指针生命周期、内存分配/释放配对、整数运算溢出可能
4. **CERT C 合规**: 对照 CERT C 安全编码标准检查整数溢出 (INT30-C, INT31-C, INT32-C)、字符串操作 (STR31-C)、表达式 (EXP37-C)
5. **OWASP**: 对照 OWASP Top 10 检查 Web 前端安全 (XSS、CSP、不安全反序列化)
6. **性能剖析**: 识别算法复杂度退化 (O(n^2) vs O(n)), 识别热路径上的堆分配
7. **ISO/IEC 25010**: 从功能性、性能效率、安全性、可用性、可靠性、可维护性、可移植性维度评估

**工具辅助**: 使用多代理并行分析架构，覆盖 profiler 内核、网络类型实现、demo 应用、构建系统、Web 编辑器 + Wasm 五大部分。
