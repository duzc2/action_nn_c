# de_dust2 地点标识符草案

## 1. 文档目标

定义 `demo/cs` 第一版建议支持的 `de_dust2` 地点标识符集合。  
这些标识符只用于命令参数与内部映射，不承担人类语义理解职责。

## 2. 设计原则

- 每个地点只有一个合法标识符
- 标识符必须明确、唯一、简短、可读
- 不支持别名
- 不做大小写宽松匹配
- 不做语义理解
- 命令层先完成字符串到整数 ID 的转换，再进入网络层

## 3. 第一版建议支持的地点标识符

### 3.1 全局区域

| place_token | place_id | 说明 |
|---|---:|---|
| `t_spawn` | 0 | T 出生区域 |
| `ct_spawn` | 1 | CT 出生区域 |
| `mid` | 2 | 中路主区域 |
| `a_site` | 3 | A 包点主区域 |
| `b_site` | 4 | B 包点主区域 |

### 3.2 A 路区域

| place_token | place_id | 说明 |
|---|---:|---|
| `long_doors` | 5 | A 大门外/门口区域 |
| `outside_long` | 6 | A 大外场区域 |
| `pit` | 7 | A 坑位区域 |
| `a_ramp` | 8 | A 坡区域 |
| `catwalk` | 9 | A 小道区域 |
| `goose` | 10 | A 点鹅位区域 |
| `a_car` | 11 | A 车位区域 |

### 3.3 中路区域

| place_token | place_id | 说明 |
|---|---:|---|
| `top_mid` | 12 | 中路上段 |
| `lower_mid` | 13 | 中路下段 |
| `xbox` | 14 | Xbox 区域 |
| `mid_doors` | 15 | 中门区域 |

### 3.4 B 路区域

| place_token | place_id | 说明 |
|---|---:|---|
| `upper_tuns` | 16 | B 上层通道 |
| `lower_tuns` | 17 | B 下层通道 |
| `b_doors` | 18 | B 门区域 |
| `b_window` | 19 | B 窗区域 |
| `b_platform` | 20 | B 平台区域 |
| `back_plat` | 21 | B 后平台区域 |
| `b_car` | 22 | B 车位区域 |

## 4. 命令接口建议

CLI 命令建议直接传 `place_token`，例如：

- `goto a_site`
- `goto mid`
- `goto long_doors`
- `goto upper_tuns`

错误处理建议：

- token 不存在：立即报错
- token 大小写不匹配：立即报错
- token 拼写不完整：立即报错

## 5. 对网络和系统设计的影响

地点标识符进入系统后，建议立刻转换为：

1. `place_id`
2. `goal_anchor`
3. `goal_region_radius`
4. 可选的 `waypoint_sequence`

因此网络层不需要处理原始字符串，也不需要做语义理解。

推荐做法：

- CLI/配置层：处理字符串 token
- 控制层入口：转换为整数 ID
- 网络层：消费整数 ID / one-hot / embedding

## 6. 第一版建议固定的约束

- `place_token -> place_id` 映射必须固定
- 训练集、推理输入、日志输出必须共用同一份字典
- 不允许同一地点存在多个 token
- 不允许一个 token 指向多个地点

## 7. 后续可扩展方向

- 为每个 token 补充目标区域示意图
- 为每个 token 补充邻接节点列表
- 为每个 token 补充训练样本覆盖统计
