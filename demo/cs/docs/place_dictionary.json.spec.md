# place_dictionary.json 格式规范

## 1. 文档目标

定义 `demo/cs` 中 `place_dictionary.json` 的正式格式。

这个文件的作用是固定：

- `place_token`
- `place_id`
- 基础说明信息

它是以下模块的共同输入：

- `cs_label_session`
- `cs_dataset_build`
- `cs_dataset_report`
- Version 1 train
- Version 1 infer

所以它必须被视为：

> **Version 1 的基础字典文件。**

## 2. 设计原则

- 一个 `place_token` 只对应一个 `place_id`
- 一个 `place_id` 只对应一个 `place_token`
- 字典必须稳定、唯一、可版本化
- 不支持别名
- 不支持动态推断
- 工具和训练流程必须共用同一份字典

## 3. 推荐存放位置

建议固定为：

```text
demo/cs/config/place_dictionary.json
```

## 4. 顶层结构

建议使用如下顶层字段：

```json
{
  "dictionary_name": "de_dust2_v1_places",
  "map_name": "de_dust2",
  "version": 1,
  "entries": []
}
```

## 5. 字段定义

## 5.1 顶层字段

### `dictionary_name`

类型：

- string

作用：

- 标识当前字典名称

建议值：

- `de_dust2_v1_places`

### `map_name`

类型：

- string

作用：

- 标识该字典只适用于哪张地图

建议值：

- `de_dust2`

### `version`

类型：

- integer

作用：

- 标识字典版本

### `entries`

类型：

- array

作用：

- 存放所有地点条目

## 5.2 条目字段

每个条目建议结构如下：

```json
{
  "place_id": 0,
  "place_token": "t_spawn",
  "display_name": "T Spawn",
  "enabled_in_v1": true,
  "notes": "T spawn main area"
}
```

### `place_id`

类型：

- integer

约束：

- 全局唯一
- 从 0 开始连续编号更好

### `place_token`

类型：

- string

约束：

- 全局唯一
- 仅使用固定 token
- 严格区分大小写

### `display_name`

类型：

- string

作用：

- 仅用于日志和文档显示

说明：

- 不是输入命令
- 不参与训练输入

### `enabled_in_v1`

类型：

- boolean

作用：

- 标识该地点是否纳入 Version 1

### `notes`

类型：

- string

作用：

- 记录补充说明

## 6. Version 1 推荐示例

```json
{
  "dictionary_name": "de_dust2_v1_places",
  "map_name": "de_dust2",
  "version": 1,
  "entries": [
    {
      "place_id": 0,
      "place_token": "t_spawn",
      "display_name": "T Spawn",
      "enabled_in_v1": true,
      "notes": "T spawn main area"
    },
    {
      "place_id": 1,
      "place_token": "ct_spawn",
      "display_name": "CT Spawn",
      "enabled_in_v1": true,
      "notes": "CT spawn main area"
    },
    {
      "place_id": 2,
      "place_token": "mid",
      "display_name": "Mid",
      "enabled_in_v1": true,
      "notes": "Mid main area"
    },
    {
      "place_id": 3,
      "place_token": "a_site",
      "display_name": "A Site",
      "enabled_in_v1": true,
      "notes": "A site main area"
    },
    {
      "place_id": 4,
      "place_token": "b_site",
      "display_name": "B Site",
      "enabled_in_v1": true,
      "notes": "B site main area"
    },
    {
      "place_id": 5,
      "place_token": "long_doors",
      "display_name": "Long Doors",
      "enabled_in_v1": true,
      "notes": "Long doors area"
    },
    {
      "place_id": 6,
      "place_token": "catwalk",
      "display_name": "Catwalk",
      "enabled_in_v1": true,
      "notes": "Catwalk main area"
    },
    {
      "place_id": 7,
      "place_token": "upper_tuns",
      "display_name": "Upper Tunnels",
      "enabled_in_v1": true,
      "notes": "Upper tunnels area"
    }
  ]
}
```

## 7. 一致性要求

所有工具都必须校验以下内容：

1. `map_name` 必须为 `de_dust2`
2. `place_id` 不允许重复
3. `place_token` 不允许重复
4. `enabled_in_v1=true` 的条目必须可用于 Version 1
5. `label_segments.json` 中的 token 和 id 必须来自此字典
6. `train_list.json` / `val_list.json` / `test_list.json` 中的 token 和 id 必须来自此字典

## 8. 对各工具的使用方式

## 8.1 `cs_label_session`

用途：

- 校验用户输入的 `place_token` 是否合法
- 自动补出对应 `place_id`

## 8.2 `cs_dataset_build`

用途：

- 校验标注文件中的 token / id 是否一致
- 统一生成样本清单中的 token / id

## 8.3 train

用途：

- 固定类别总数
- 固定输出层类别编号

## 8.4 infer

用途：

- 把预测出的 `place_id` 映射回 `place_token`
- 供日志输出使用

## 9. 未来扩展规则

如果未来进入 Version 2/3，需要新增地点条目，建议遵守：

- 尽量不修改已有 `place_id`
- 新增条目放到新版本字典中
- 通过 `version` 区分

不要：

- 在旧字典中重排已有编号
- 让同一个 token 在不同版本含义变化

## 10. 最终要求

后续只要开始实际开发，建议第一时间先落这个文件。  
因为没有统一字典，后面的：

- 标签
- 数据集
- 训练
- 推理

都会失去统一编号基础。
