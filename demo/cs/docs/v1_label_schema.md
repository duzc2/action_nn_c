# Version 1 文件结构定义

## 1. 文档目标

定义 Version 1 数据采集与数据集构建阶段的核心文件结构。

目标：

- 让采集工具写出的文件固定
- 让训练工具读取的文件固定
- 让后续版本可在此基础上扩展

## 2. 核心文件列表

Version 1 建议固定以下文件：

- `session.json`
- `capture_state.json`
- `state_trace.jsonl`
- `train_list.json`
- `val_list.json`
- `test_list.json`
- `build_report.json`
- `class_balance.json`

## 3. `session.json`

## 3.1 作用

记录某次采集 session 的基础元数据。

## 3.2 示例结构

```json
{
  "session_id": "session_0001",
  "map_name": "de_dust2",
  "resolution": {
    "width": 1280,
    "height": 720
  },
  "capture_fps": 8,
  "team": "t",
  "start_time": "2026-04-02T10:00:00+08:00",
  "end_time": "2026-04-02T10:12:00+08:00",
  "notes": "v1 area collection"
}
```

## 4. `capture_state.json`

## 4.1 作用

记录采集进度与当前状态，供运行时状态查询使用。

## 4.2 示例结构

```json
{
  "session_id": "session_0001",
  "status": "running",
  "captured_frame_count": 1450,
  "last_frame_index": 1449,
  "last_state_timestamp": "2026-04-02T10:03:21.240+08:00"
}
```

## 5. `state_trace.jsonl`

## 5.1 作用

记录截图对应的程序精确状态轨迹，供后续自动派生标签使用。

## 5.2 示例结构

```json
{"frame_index":0,"timestamp":"2026-04-02T10:00:01.120+08:00","pos_x":128.0,"pos_y":64.0,"pos_z":-32.0,"yaw":90.0,"pitch":0.0,"velocity_x":0.0,"velocity_y":0.0,"velocity_z":0.0}
{"frame_index":1,"timestamp":"2026-04-02T10:00:01.245+08:00","pos_x":129.5,"pos_y":64.2,"pos_z":-32.0,"yaw":92.0,"pitch":0.0,"velocity_x":12.0,"velocity_y":1.6,"velocity_z":0.0}
```

## 6. `train_list.json` / `val_list.json` / `test_list.json`

## 6.1 作用

定义可直接供训练程序消费的样本清单。

## 6.2 单条样本建议结构

```json
{
  "sample_id": "session_0001_000321",
  "image_path": "samples/session_0001/frame_000321.png",
  "session_id": "session_0001",
  "frame_index": 321,
  "timestamp": "2026-04-02T10:04:20.015+08:00",
  "teacher_pose": {
    "pos_x": 512.0,
    "pos_y": 384.0,
    "pos_z": -24.0,
    "yaw": 135.0,
    "pitch": 0.0
  },
  "place_token": "mid",
  "place_id": 2,
  "label_source": "teacher_projected"
}
```

## 6.3 清单整体结构示例

```json
{
  "dataset_version": "v1_area",
  "split": "train",
  "samples": [
    {
      "sample_id": "session_0001_000321",
      "image_path": "samples/session_0001/frame_000321.png",
      "session_id": "session_0001",
      "frame_index": 321,
      "timestamp": "2026-04-02T10:04:20.015+08:00",
      "teacher_pose": {
        "pos_x": 512.0,
        "pos_y": 384.0,
        "pos_z": -24.0,
        "yaw": 135.0,
        "pitch": 0.0
      },
      "place_token": "mid",
      "place_id": 2,
      "label_source": "teacher_projected"
    }
  ]
}
```

## 7. `build_report.json`

## 7.1 作用

记录构建数据集时的处理摘要。

## 7.2 示例结构

```json
{
  "raw_session_count": 4,
  "raw_frame_count": 12000,
  "state_trace_count": 12000,
  "filtered_frame_count": 4300,
  "dedup_removed_count": 2200,
  "blur_removed_count": 800,
  "teacher_alignment_drop_count": 120,
  "train_count": 3440,
  "val_count": 430,
  "test_count": 430
}
```

## 8. `class_balance.json`

## 8.1 作用

记录每个类别在各 split 中的样本数量。

## 8.2 示例结构

```json
{
  "train": {
    "t_spawn": 420,
    "ct_spawn": 390,
    "mid": 700,
    "a_site": 610,
    "b_site": 590
  },
  "val": {
    "t_spawn": 50,
    "ct_spawn": 48,
    "mid": 90,
    "a_site": 70,
    "b_site": 72
  }
}
```

## 9. 文件一致性要求

所有文件都必须满足：

- `place_token` 与 `place_id` 来自同一份字典或同一套投影规则
- `session_id` 必须可追溯
- `image_path` 必须存在
- `state_trace.jsonl` 与 `frame_index` 必须可对齐
- 每条样本的 `label_source` 必须明确
- `train/val/test` 不允许样本重复

## 10. 对用户和工具的意义

这样定义后：

- 用户不需要猜文件怎么组织
- 工具之间不会各写各的格式
- train 程序能稳定读取
- 后续测试和复盘更容易
