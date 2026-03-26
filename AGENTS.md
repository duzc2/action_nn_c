# AGENTS 开发行为规范

必须先阅读 docs/ 里的所有文件。
使用clang进行编译。
你在windows里运行，环境是win11 powershell，执行命令要严格遵循 powershell语法。

## Windows 工具链约束（本机已验证）
- 本机当前 shell 默认**没有**注入 Visual Studio C/C++ 构建环境；直接运行 `clang` / `clang-cl` + `Ninja` 的 CMake configure 可能卡在 toolchain probe / try-compile 阶段。
- 本机已经安装：
  - LLVM：`C:\Program Files\LLVM\bin`
  - Visual Studio 2022 Community
  - Windows SDK 10.0.26100.0
- 本机要求的 **clang 构建入口** 是：先调用  
  `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`  
  注入 x64 的 `cl/link/rc/mt/INCLUDE/LIB` 环境，再继续运行 CMake/Ninja。
- 在本机上，**禁止**直接裸跑：
  - `cmake -G Ninja -DCMAKE_C_COMPILER=clang ...`
  - `cmake -G Ninja -DCMAKE_C_COMPILER=clang-cl ...`
  除非当前命令已经先 `call vcvars64.bat`。
- 在本机上，`VsDevCmd.bat` 默认可能落到 x86；若目标是 x64，优先使用 `vcvars64.bat`，不要假设 `VsDevCmd.bat` 会自动进入 x64。
- 本机未安装 Visual Studio 的 `ClangCL` 平台工具集，因此**不要**使用：
  - `cmake -G "Visual Studio 17 2022" -T ClangCL`
- 若需要保留 clang 要求，优先使用如下模式：
  - `cmd /c "call \"...\\vcvars64.bat\" && cmake -S ... -B ... -G Ninja -DCMAKE_C_COMPILER=clang-cl ..."`
  - `cmd /c "call \"...\\vcvars64.bat\" && cmake --build ..."`
- 任何构建命令都应：
  - 一次只跑一个 demo
  - 一次只跑一条命令
  - 先 configure，再 build，再 run
  - 避免并行试错，避免在日志不可见时长时间挂起

## 1. 适用范围
- 本规范适用于在本仓库内执行开发、重构、修复、生成代码与文档的所有智能体。
- 智能体必须以 `docs/` 目录中的文档为唯一开发依据，不得绕过或弱化文档要求。
- 在修改或者编译任何demo之前，先阅读 demo里的 markdown 文件。


## 2. 文档优先与执行顺序
- 在开始任何实现前，必须先阅读并对齐相关文档，至少包含：
  - `docs/user_manual.md`
  - `docs/developer_manual.md`
  - `docs/network_topology_requirements.md`
  - `docs/profiler_api_contract_draft.md`
  - `docs/profiler_interface_draft.md`
  - `docs/profiler_development_plan.md`
- 需求、接口、流程存在冲突时，按“更具体、更新近、可执行性更强”的文档条款执行，并在交付说明中写明依据。
- 禁止脱离 `docs/` 自行扩展需求、引入未定义概念或创建额外流程。

## 3. 强制开发流程
- 必须遵循文档定义的主链路：
  1. 首次编译（按 CMake 开关启用网络类型）
  2. 首次运行（调用 profiler 生成训练/推理 `.c` 与固定 `.h`）
  3. 第二次编译训练工程（依赖推理 `.c` + 训练 `.c`）
  4. 第三次编译推理工程（仅依赖推理 `.c`）
  5. 测试验证与结果记录
- 未启用的网络类型不得参与编译、注册和运行流程。

## 4. 变更边界与架构约束
- 新增网络类型必须通过注册配置与 CMake 开关接入，不得改动主流程来做特例适配。
- 禁止保留旧入口、旧配置路径、兼容分支；迁移后统一走新流程。
- 不得随意移动文件或重排目录；确需调整时，必须先满足文档约束并在交付说明中标注原因与影响。

## 5. 代码与生成物质量要求
- 所有源码、模板头文件、生成代码必须严格符合 C99。
- 注释密度必须满足文档要求（源码与生成代码不少于 50%）。
- 编译要求执行严格校验：警告即错误，且目标为零警告。
- 错误处理遵循“首错即停（快速失败）”，并按统一错误机制返回可定位信息。

## 6. 实施原则
- 优先通过流程、结构和边界治理问题，不使用数据补丁替代流程修复。
- 对存在设计取舍、风险分歧或文档未明确事项，必须先给出方案再执行，不得擅自决策。
- 交付时必须说明：所依据的文档、实际改动范围、验证结果，以及与文档的一致性结论。
