# Version 1 工具 CLI 规格

## 1. 文档目标

定义 Version 1 数据采集工具的命令行接口。  
目标是让后续工具实现有统一入口，不靠临时命令拼凑。

本文档覆盖 4 个工具：

- `cs_capture_session`
- `cs_state_trace`
- `cs_dataset_build`
- `cs_dataset_report`

补充约束：

- 以上 4 个工具后续必须同时提供 GUI 版本
- CLI 规格仍保留，作为脚本接口与自动化回归接口
- GUI 与 CLI 必须共享同一套核心参数和输出文件格式

## 2. 总体原则

- CLI 只做明确命令，不做自然语言理解
- 参数名固定、可脚本化
- 输出文件格式固定
- 错误时快速失败
- 所有路径使用相对路径或显式传入路径

补充 GUI 原则：

- GUI 是主操作入口，不是可有可无的附属面板
- GUI 必须展示当前状态、当前路径、当前 session 和错误信息
- GUI 触发的行为必须可映射到同一套 CLI/核心逻辑
- GUI 不得引入第二套独立文件格式

## 3. 工具 1：`cs_capture_session`

## 3.1 目标

附着 CS1.6 窗口并采集原始帧，生成一个 session 目录。

## 3.2 建议命令

```powershell
cs_capture_session start `
  --session-root demo/cs/data/v1_area/raw `
  --session-id session_0001 `
  --map de_dust2 `
  --width 1280 `
  --height 720 `
  --capture-fps 8 `
  --team t `
  --teacher-source cs_runtime
```

停止命令：

```powershell
cs_capture_session stop --session-id session_0001
```

状态命令：

```powershell
cs_capture_session status --session-id session_0001
```

## 3.3 子命令

### `start`

作用：

- 创建 session
- 检测窗口
- 开始截图

必要参数：

- `--session-root`
- `--session-id`
- `--map`
- `--width`
- `--height`
- `--capture-fps`

可选参数：

- `--team`
- `--notes`

### `stop`

作用：

- 结束截图
- 写回 session 元数据

必要参数：

- `--session-id`

### `status`

作用：

- 输出当前 session 的采集状态

## 3.4 主要输出

生成：

- `frames/`
- `session.json`
- `capture_state.json`
- `state_trace.jsonl`

## 3.5 GUI 要求

`cs_capture_session` GUI 至少需要：

- 启动采集按钮
- 停止采集按钮
- 当前 session id 显示
- 当前窗口状态显示
- 当前分辨率显示
- 已采集帧数显示
- 状态采集状态显示
- 实时状态显示，例如 `starting / running / stopped / error`
- 输出目录显示
- 最近错误信息显示

## 4. 工具 2：`cs_state_trace`

## 4.1 目标

记录截图对齐的程序精确状态轨迹。

## 4.2 建议命令

启动状态采集：

```powershell
cs_state_trace start `
  --session-root demo/cs/data/v1_area/raw `
  --session-id session_0001
```

查看状态采集情况：

```powershell
cs_state_trace status `
  --session-root demo/cs/data/v1_area/raw `
  --session-id session_0001
```

停止状态采集：

```powershell
cs_state_trace stop `
  --session-root demo/cs/data/v1_area/raw `
  --session-id session_0001
```

## 4.3 子命令

### `start`

作用：

- 开始读取程序内部状态
- 输出状态轨迹

必要参数：

- `--session-root`
- `--session-id`

### `status`

作用：

- 输出当前状态采集状态

### `stop`

作用：

- 停止状态采集

## 4.4 主要输出

更新：

- `state_trace.jsonl`

## 4.5 GUI 要求

`cs_state_trace` GUI 至少需要：

- 当前 session 显示
- 当前帧号显示
- 最近一次状态采样结果显示
- 当前状态读取状态显示
- 最近错误信息显示

## 5. 工具 3：`cs_dataset_build`

## 5.1 目标

把原始 session 数据整理成可直接训练的数据集清单。

## 5.2 建议命令

```powershell
cs_dataset_build run `
  --raw-root demo/cs/data/v1_area/raw `
  --output-root demo/cs/data/v1_area/processed `
  --min-frame-step 3 `
  --dedup on `
  --blur-filter on `
  --split train:val:test=8:1:1
```

## 5.3 子命令

### `run`

作用：

- 读取所有 session
- 读取状态轨迹
- 自动投影区域标签
- 构建样本列表
- 去重
- 切分数据集

必要参数：

- `--raw-root`
- `--output-root`

可选参数：

- `--min-frame-step`
- `--dedup`
- `--blur-filter`
- `--split`

## 5.4 主要输出

生成：

- `samples/`
- `train_list.json`
- `val_list.json`
- `test_list.json`
- `build_report.json`

补充要求：

- 必须校验每个样本都能找到对齐状态
- 必须记录标签来源为 `teacher_projected`
- 对齐失败样本必须丢弃并计入报告

## 5.5 GUI 要求

`cs_dataset_build` GUI 至少需要：

- 原始数据目录输入框
- 输出目录输入框
- 抽样参数输入控件
- 去重与过滤开关
- split 配置输入控件
- 开始构建按钮
- 构建进度显示
- 构建结果摘要显示
- 失败时错误信息显示

## 6. 工具 4：`cs_dataset_report`

## 6.1 目标

输出数据集统计结果，帮助用户知道还缺什么数据。

## 6.2 建议命令

```powershell
cs_dataset_report run `
  --dataset-root demo/cs/data/v1_area/processed `
  --output-root demo/cs/data/v1_area/reports
```

## 6.3 子命令

### `run`

作用：

- 读取 train / val / test 清单
- 输出类别分布
- 输出 session 覆盖情况
- 输出缺口提示

必要参数：

- `--dataset-root`
- `--output-root`

## 6.4 主要输出

生成：

- `class_balance.json`
- `session_coverage.json`
- `dataset_report.md`

## 6.5 GUI 要求

`cs_dataset_report` GUI 至少需要：

- 数据集目录输入框
- 报告输出目录输入框
- 生成报告按钮
- 类别分布摘要显示
- session 覆盖摘要显示
- 缺类或失衡提示显示
- 报告文件路径显示

## 7. 标签投影规则校验

所有工具中涉及自动区域投影的地方，都必须：

- 使用统一的 `place_token -> place_id` 字典
- 使用统一的区域投影规则
- 对投影失败样本立即标记并丢弃

建议统一使用：

- `demo/cs/config/place_dictionary.json`

作为 token 字典输入源。

## 8. 推荐错误处理

错误时建议统一输出：

- 错误码
- 英文错误描述
- 当前 session / 文件路径 / token

典型错误：

- 找不到窗口
- 分辨率不匹配
- 地图不匹配
- session 不存在
- 状态读取失败
- 状态与截图无法对齐
- 标签投影失败
- 输出目录不可写

## 9. 推荐退出码

建议：

- `0`：成功
- `1`：参数错误
- `2`：运行时状态错误
- `3`：I/O 错误
- `4`：字典、投影或标签错误

## 10. 未来扩展

后续如果进入 Version 2/3，可继续沿用本 CLI 风格，不重新发明工具接口。

但扩展时仍需保持：

- GUI 继续作为主入口
- CLI 继续作为脚本接口
- 二者共享同一套核心逻辑与输出格式
