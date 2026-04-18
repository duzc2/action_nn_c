# Version 1 工具迁移清单

## 1. 文档目标

本文档把当前 `demo/cs/tools` 源码，从旧的：

> `capture + label_segments`

迁移到新的：

> `capture + state_trace + teacher projection`

所需的**最小改造点**固定下来。  
它不是实现代码，只是面向开发的最小改造清单。

## 2. 当前现状总结

当前工具链仍然围绕人工分段标注设计：

- `cs_capture_session` 会初始化 `label_segments.json`
- `cs_label_session` 负责用户切换 `place_token`
- `cs_dataset_build` 直接消费 `label_segments.json`
- `cs_tool_common.h/.c` 中的核心共享结构还是 `CsLabelSegment`

这与新的 Version 1 文档已不一致。  
新的目标是：

- `cs_capture_session`：截图与 session 生命周期
- `cs_state_trace`：程序精确状态轨迹
- `cs_dataset_build`：自动对齐 + 自动投影标签
- `cs_dataset_report`：统计结果与投影失败计数

## 3. 最小改造原则

- 先改数据链路，不先碰网络训练
- 先让 `raw -> processed` 跑通，再优化状态读取细节
- 先支持 Version 1 的 5~8 个大区域，不先追全图
- 先保证失败可见，再考虑性能

## 4. 文件级改造清单

## 4.1 `demo/cs/tools/CMakeLists.txt`

最小改造：

- 删除 `cs_label_session` 目标
- 新增 `cs_state_trace` 目标
- 继续保留 `cs_capture_session`
- 继续保留 `cs_dataset_build`
- 继续保留 `cs_dataset_report`

建议顺序：

1. 先新增 `cs_state_trace.c`
2. 让它能单独编译
3. 再删除 `cs_label_session.c` 的构建入口

## 4.2 `demo/cs/tools/cs_tool_common.h`

当前问题：

- 仍暴露 `CsLabelSegment`
- 仍暴露 `CsLabelSegmentsFile`
- `CsCaptureState` 仍带 `current_place_token / current_place_id`

最小改造：

- 删除或废弃：
  - `CsLabelSegment`
  - `CsLabelSegmentsFile`
  - `cs_tool_write_label_segments`
  - `cs_tool_read_label_segments`
- 新增：
  - `CsStateTraceRecord`
  - `CsTeacherProjectionZone`
  - `CsTeacherProjectionConfig`
- 扩展 `CsCaptureState`：
  - `last_state_timestamp`
  - `state_trace_count`
  - `teacher_source`
  - 可选 `state_trace_status`

建议：

- `state_trace.jsonl` 采用逐行写入，不要一次性全量载入内存
- 投影配置和状态轨迹解析，可先做成显式小函数，不引入复杂解析框架

## 4.3 `demo/cs/tools/cs_tool_common.c`

当前问题：

- `capture_state.json` 的读写格式仍绑定 `current_place_token / current_place_id`
- 仍实现 `label_segments.json` 的序列化与反序列化

最小改造：

- 改写 `cs_tool_write_capture_state` / `cs_tool_read_capture_state`
  - 输出 `last_state_timestamp`
  - 输出 `state_trace_count`
  - 输出 `teacher_source`
- 删除或废弃 `label_segments` 读写函数
- 新增：
  - `state_trace.jsonl` 逐行追加写入函数
  - `state_trace` 逐行读取函数
  - teacher 投影配置读取函数
  - 2D 点落区判断函数
  - 边界模糊带判定函数

建议先支持：

- `polygon_2d`
- `aabb_2d`

不要一开始加：

- 多地图
- 复杂层级区域
- 路径图联动

## 4.4 `demo/cs/tools/cs_capture_session.c`

当前问题：

- `cs_capture_prepare_session` 仍创建 `label_segments.json`
- `status` 输出仍包含 `current_place_token / current_place_id`
- 生命周期只覆盖截图，不覆盖状态轨迹链路

最小改造：

- `start` 时创建：
  - `session.json`
  - `capture_state.json`
  - 空的 `state_trace.jsonl`
- 移除 `label_segments.json` 初始化
- 新增 `--teacher-source` 参数
- `status` 输出改成：
  - `captured_frame_count`
  - `last_frame_index`
  - `last_state_timestamp`
  - `state_trace_count`
  - `teacher_source`
  - `state_trace_status`

建议实现策略：

- 短期最小实现：继续让 `cs_capture_session` 只负责截图
- `cs_state_trace` 独立启动并写同一 session
- 后续若需要，再考虑由 `cs_capture_session` 统一拉起 `cs_state_trace`

## 4.5 `demo/cs/tools/cs_label_session.c`

当前问题：

- 这个工具的职责已经和新方案冲突

最小改造：

- 用新文件 `cs_state_trace.c` 替换它

`cs_state_trace` 最小职责：

- `start`
  - 进入运行态
  - 开始写 `state_trace.jsonl`
- `status`
  - 输出当前状态读取状态
  - 输出最近一次采样结果
- `stop`
  - 停止状态读取

最小输出字段：

- `frame_index`
- `timestamp`
- `pos_x / pos_y / pos_z`
- `yaw / pitch`
- `velocity_x / velocity_y / velocity_z`

## 4.6 `demo/cs/tools/cs_dataset_build.c`

这是最大的改造点。

当前问题：

- 仍按 `label_segments.json` 遍历段
- 样本标签直接来自人工段标签
- `build_report.json` 还没有 teacher 链路统计

最小改造：

- 输入从：
  - `label_segments.json`
  改为：
  - `state_trace.jsonl`
  - `de_dust2_v1_teacher_projection.json`
- 处理流程改成：

```text
frame
-> find aligned state record
-> project zone
-> drop ambiguous/outside/disabled
-> emit teacher_projected sample
```

- 样本结构新增：
  - `timestamp`
  - `teacher_pose`
  - `label_source`
- `build_report.json` 新增：
  - `state_trace_count`
  - `teacher_alignment_drop_count`
  - `projection_outside_count`
  - `projection_ambiguous_count`
  - `projection_disabled_count`

建议分两步做：

1. 先做“按 frame_index 精确对齐”
2. 再扩成“按时间戳最近邻对齐”

Version 1 最小版更推荐：

- 先把截图和状态一一对应写成同一 `frame_index`
- 这样 `dataset_build` 最简单

## 4.7 `demo/cs/tools/cs_dataset_report.c`

当前问题不大，因为它主要按 `place_id` 汇总。

最小改造：

- 保留现有 `train / val / test` 统计主逻辑
- 新增读取 `build_report.json` 的 teacher 相关计数
- 在 `dataset_report.md` 中显示：
  - 对齐丢弃数
  - 区域外样本数
  - 边界歧义样本数

这样就能把“模型数据少”与“投影规则有问题”区分开。

## 5. 推荐实施顺序

最小可落地顺序建议为：

1. 先改 `cs_tool_common.h/.c`
2. 再新增 `cs_state_trace.c`
3. 再改 `cs_capture_session.c`
4. 再重写 `cs_dataset_build.c` 的输入链路
5. 最后补 `cs_dataset_report.c`

## 6. 第一批必须打通的最小闭环

只要下面这条链能跑通，Version 1 的 teacher 路线就算真正成立：

```text
cs_capture_session start
-> 保存 frames
-> cs_state_trace start
-> 保存 state_trace.jsonl
-> cs_dataset_build run
-> 自动投影出 place_id
-> 生成 train_list.json
```

## 7. 暂时不要做的事

当前阶段不建议先做：

- GUI 大重构
- 多地图支持
- 精细点位全覆盖
- 朝向桶自动投影
- 轨迹平滑器
- 行为 teacher action

这些都可以放到后续版本。

## 8. 完成定义

只有满足以下条件，才算工具链迁移完成：

1. `cs_label_session` 不再是主链路
2. `state_trace.jsonl` 成为原始数据标准文件
3. `cs_dataset_build` 能自动生成 `teacher_projected` 标签
4. `build_report.json` 能反映对齐与投影失败原因
5. 文档、CLI 和目录结构三者一致
