# Version 1 工具 CLI 规格

## 1. 文档目标

定义 Version 1 数据采集工具的命令行接口。  
目标是让后续工具实现有统一入口，不靠临时命令拼凑。

本文档覆盖 4 个工具：

- `cs_capture_session`
- `cs_label_session`
- `cs_dataset_build`
- `cs_dataset_report`

## 2. 总体原则

- CLI 只做明确命令，不做自然语言理解
- 参数名固定、可脚本化
- 输出文件格式固定
- 错误时快速失败
- 所有路径使用相对路径或显式传入路径

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
  --team t
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

## 4. 工具 2：`cs_label_session`

## 4.1 目标

记录“当前区域 token 在某一段时间内生效”的标签分段。

## 4.2 建议命令

设置当前标签：

```powershell
cs_label_session set `
  --session-root demo/cs/data/v1_area/raw `
  --session-id session_0001 `
  --place-token mid
```

查看当前标签：

```powershell
cs_label_session current `
  --session-root demo/cs/data/v1_area/raw `
  --session-id session_0001
```

结束当前标签段：

```powershell
cs_label_session close `
  --session-root demo/cs/data/v1_area/raw `
  --session-id session_0001
```

## 4.3 子命令

### `set`

作用：

- 结束上一个标签段
- 开启新的标签段

必要参数：

- `--session-root`
- `--session-id`
- `--place-token`

### `current`

作用：

- 输出当前激活标签

### `close`

作用：

- 结束当前激活标签段，但不新开标签

## 4.4 主要输出

更新：

- `label_segments.json`

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
- 读取标签段
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

## 7. place token 校验规则

所有工具中涉及 `--place-token` 的地方，都必须：

- 仅接受字典内 token
- 严格区分大小写
- 非法 token 立即报错

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
- token 非法
- 标签段未关闭
- 输出目录不可写

## 9. 推荐退出码

建议：

- `0`：成功
- `1`：参数错误
- `2`：运行时状态错误
- `3`：I/O 错误
- `4`：字典或标签错误

## 10. 未来扩展

后续如果进入 Version 2/3，可继续沿用本 CLI 风格，不重新发明工具接口。
