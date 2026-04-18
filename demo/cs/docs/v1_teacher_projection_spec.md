# Version 1 teacher 状态投影规则

## 1. 文档目标

本文档定义 `demo/cs` 的 Version 1 中：

> **如何把 `state_trace.jsonl` 中的程序精确状态，稳定投影为 `place_token / place_id`。**

本文档只服务于训练/采集阶段。  
推理阶段仍然只使用截图，不依赖该投影链路。

## 2. 适用范围

本规则只适用于：

- 地图：`de_dust2`
- 字典：`demo/cs/config/place_dictionary.json`
- Version 1 启用标签
- 私有离线训练/采集环境

本文档不处理：

- 公开对战环境
- 运行时闭环推理
- 多地图共用投影
- 自动从视觉中反推地图坐标

## 3. 总体原则

Version 1 的区域标签必须来自以下固定链路：

```text
state_trace.jsonl
-> pose normalize
-> region projection
-> ambiguity filter
-> place_token / place_id
```

必须遵守：

- 投影结果必须完全可复现
- 相同输入状态必须得到相同输出标签
- 同一帧不得同时投影到多个最终标签
- 不确定样本宁可丢弃，不做猜测性归类

## 4. 输入与输出

## 4.1 输入

投影器至少需要以下输入：

- `session.json`
- `state_trace.jsonl`
- `demo/cs/config/place_dictionary.json`
- 固定的投影规则文件

其中 `state_trace.jsonl` 每条记录至少包含：

- `frame_index`
- `timestamp`
- `pos_x`
- `pos_y`
- `pos_z`
- `yaw`
- `pitch`

## 4.2 输出

每个可用样本必须输出：

- `place_token`
- `place_id`
- `label_source = "teacher_projected"`
- `projection_status`

其中：

- `projection_status = ok`：可用于训练
- `projection_status = ambiguous`：边界不确定，丢弃
- `projection_status = outside`：未落入任何已定义区域，丢弃
- `projection_status = disabled`：区域存在但未启用 Version 1，丢弃

## 5. 投影规则文件

建议固定新增一份规则文件：

```text
demo/cs/config/de_dust2_v1_teacher_projection.json
```

该文件不用于训练输入，只用于离线标签生成。

## 5.1 规则文件职责

它必须固定：

- 坐标系版本
- 投影区域集合
- 每个区域的几何边界
- 区域优先级
- 边界模糊带宽度
- Version 1 是否启用

## 5.2 推荐顶层结构

```json
{
  "projection_name": "de_dust2_v1_teacher_projection",
  "map_name": "de_dust2",
  "dictionary_name": "de_dust2_v1_places",
  "version": 1,
  "boundary_margin": 24.0,
  "zones": []
}
```

## 5.3 zone 结构

每个 zone 建议至少包含：

```json
{
  "place_id": 2,
  "place_token": "mid",
  "enabled_in_v1": true,
  "priority": 100,
  "z_min": -128.0,
  "z_max": 256.0,
  "shape": {
    "type": "polygon_2d",
    "points": [
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0]
    ]
  }
}
```

说明：

- `priority` 数值越小优先级越高
- `z_min / z_max` 用于排除高低层误投影
- `shape.type` 在 Version 1 只建议支持：
  - `polygon_2d`
  - `aabb_2d`

## 6. 固定投影算法

Version 1 必须使用如下固定顺序，不允许运行时切换算法。

### 6.1 Step 1：基础校验

先校验：

- `session.json.map_name == "de_dust2"`
- 投影规则文件 `map_name == "de_dust2"`
- 字典版本与投影规则版本一致
- `place_token / place_id` 可在字典中查到

任一失败：

- 本次构建立即失败

### 6.2 Step 2：状态归一化

对每条 `state_trace` 记录：

- 保留原始 `pos_x / pos_y / pos_z`
- `yaw` 归一化到 `[0, 360)`
- `pitch` 原样保留

Version 1 的区域投影只依赖：

- `pos_x`
- `pos_y`
- `pos_z`

也就是说：

- `yaw / pitch` 不参与区域投影
- 它们留给朝向任务和后续版本

### 6.3 Step 3：先做 z 轴过滤

只有当：

```text
z_min <= pos_z <= z_max
```

对应 zone 才能继续参与候选。

这样做的目的：

- 避免上下层通道重叠
- 避免同一平面投影误命中

### 6.4 Step 4：做 2D 几何包含测试

对通过 z 轴过滤的 zone：

- 若 `shape.type == polygon_2d`，则做点在多边形内测试
- 若 `shape.type == aabb_2d`，则做点在矩形内测试

只有几何包含成功，zone 才进入候选集合。

### 6.5 Step 5：边界模糊带过滤

对命中的候选 zone，再计算该点到边界的最短距离。

若最短距离小于：

```text
boundary_margin
```

则该样本标记为：

- `projection_status = ambiguous`

并从训练集丢弃。

Version 1 的原则是：

- 不主动吸收边界样本
- 先保证中心区域标签纯度

### 6.6 Step 6：冲突消解

如果一个点同时命中多个 zone，按以下固定顺序处理：

1. 先比较 `priority`
2. 若 `priority` 相同，则判为 `ambiguous`
3. `ambiguous` 样本直接丢弃

不允许：

- 运行时随机选一个
- 用最近中心点强行兜底
- 根据前后帧“猜”一个标签

### 6.7 Step 7：Version 1 启用过滤

如果命中的 zone：

- `enabled_in_v1 == false`

则该样本标记为：

- `projection_status = disabled`

并丢弃，不进入 `train / val / test`。

### 6.8 Step 8：输出最终标签

只有满足以下条件的样本才可输出：

- 通过基础校验
- 命中唯一 zone
- 不在边界模糊带内
- zone 已启用 Version 1

此时输出：

- `place_token`
- `place_id`
- `label_source = "teacher_projected"`
- `projection_status = ok`

## 7. Version 1 的区域定义要求

Version 1 不追求覆盖全图细粒度点位。  
建议优先只为以下启用标签定义投影区：

- `t_spawn`
- `ct_spawn`
- `mid`
- `a_site`
- `b_site`
- `long_doors`
- `catwalk`
- `upper_tuns`

原因：

- 与当前 `place_dictionary.json` 一致
- 这些区域更大、更稳定
- 投影规则更容易先做准

## 8. Build 阶段的处理要求

`cs_dataset_build` 在读取 `state_trace.jsonl` 后，必须：

1. 对每帧查找对应状态
2. 执行固定投影算法
3. 只保留 `projection_status = ok` 的样本
4. 记录被丢弃样本数量

`build_report.json` 至少应新增统计：

- `teacher_alignment_drop_count`
- `projection_outside_count`
- `projection_ambiguous_count`
- `projection_disabled_count`

## 9. 失败优先级

Version 1 中，以下情况必须优先修规则，不允许先靠模型硬吞：

### 9.1 大量 `outside`

说明：

- 区域定义不全
- 坐标系理解有误
- 部分轨迹超出 Version 1 支持区域

### 9.2 大量 `ambiguous`

说明：

- 区域边界重叠过多
- `boundary_margin` 过大
- zone 画法过粗

### 9.3 同一区域误投到相邻区域

说明：

- zone 边界不准确
- `priority` 设计不合理

## 10. 不允许的实现方式

Version 1 明确禁止：

- 用截图内容反推投影标签
- 用人工逐段切 token 替代投影
- 用最近邻硬贴标签覆盖所有边界样本
- 在 `dataset_build` 中默默吞掉冲突而不记日志

## 11. 最小实现建议

为了尽快落地，建议最小实现顺序如下：

1. 先定义 `de_dust2_v1_teacher_projection.json`
2. 先只覆盖 5 个大区域：
   - `t_spawn`
   - `ct_spawn`
   - `mid`
   - `a_site`
   - `b_site`
3. 在 `cs_dataset_build` 中实现：
   - 读取 `state_trace.jsonl`
   - 基础点落区投影
   - 边界样本丢弃
4. 跑一轮构建报告
5. 再扩到：
   - `long_doors`
   - `catwalk`
   - `upper_tuns`

## 12. 完成定义

只有满足以下条件，才算 Version 1 投影规则完成：

1. 规则文件路径固定
2. 投影算法顺序固定
3. 同一状态输入结果可复现
4. `dataset_build` 能自动生成 `teacher_projected` 标签
5. 报告中能看见对齐失败、边界歧义、区域外样本的统计
