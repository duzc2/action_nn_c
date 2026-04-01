# CS Demo 文档总览

## 1. 目的

本目录用于沉淀 `demo/cs` 的前期方案文档。  
当前阶段只做需求理解、设计完善与方案构思，**不做开发**。

## 2. 文档清单

- `requirements.md`：需求理解、范围收敛、约束、验收口径
- `design.md`：总体架构、网络构思、运行流程、阶段规划
- `location_callouts.md`：`de_dust2` 第一版支持的地点标识符草案
- `implementation_strategy.md`：后续实现思路、训练流程、网络结构与扩展建议
- `module_versions.md`：按功能模块拆分的串行版本路线图
- `v1_area_recognition_spec.md`：Version 1 区域识别的详细规格说明
- `v1_dataset_spec.md`：Version 1 数据集生成、标注与目录规范
- `v1_data_collection_tooling.md`：Version 1 数据采集工具与用户操作流程
- `v1_tool_cli_spec.md`：Version 1 数据采集工具的 CLI 规格
- `v1_label_schema.md`：Version 1 采集与数据集文件结构定义
- `v1_user_workflow.md`：Version 1 从打开 CS 到得到数据集的用户操作流程
- `v1_dev_plan.md`：Version 1 的实际开发顺序、里程碑与测试门禁
- `place_dictionary.json.spec.md`：`place_token -> place_id` 正式字典格式规范
- `v1_test_plan.md`：Version 1 的测试项、验收线与失败记录规范
- `v1_risks_and_mitigations.md`：Version 1 的风险、征兆、排查顺序与应对策略
- `v1_task_breakdown.md`：Version 1 可执行开发任务清单

## 3. 设计定位

原始设想是：

- 启动 demo 时联动启动 CS1.6
- 用户先在游戏内完成开局
- 用户在命令行输入目标地点标识符
- 程序自动截图、分析画面、模拟键盘输入
- 角色自动向目标点移动

为保证可控性、可验证性与合规性，当前文档将方案**明确收敛为本地离线/私有测试环境下的导航演示**：

- 不面向公开多人对战
- 不讨论反作弊绕过
- 不讨论注入、内存读取、封包操控
- 第一阶段只聚焦“视觉感知 + 地点标识目标 + 移动控制”
- 开发推进采用单功能串行版本制：一个版本只做一个模块，测试通过后再进入下一个版本

## 4. 文档依据

本方案对齐了仓库现有约束，主要依据：

- `docs/user_manual.md`
- `docs/developer_manual.md`
- `docs/network_topology_requirements.md`
- `docs/profiler_api_contract_draft.md`
- `docs/profiler_interface_draft.md`
- `docs/profiler_development_plan.md`
- `docs/network_design_manual.md`
- `docs/profiler_module_blueprint.md`
- `demo/demo.md`
- 现有各 demo 的说明文档
