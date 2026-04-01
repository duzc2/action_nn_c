# Version 1 用户操作流程

## 1. 文档目标

本文档面向真正执行 Version 1 数据采集的用户。  
目标是把流程写成可以照着做的操作手册。

Version 1 只做：

> **区域识别数据采集与数据集生成**

不做：

- 朝向识别
- 路线规划
- 动作控制

## 2. 用户最终要得到什么

完成整个流程后，用户应得到：

- 原始采集 session
- 分段标签文件
- 处理后的样本清单
- train / val / test 划分结果
- 数据集统计报告

也就是说，最终目标不是“录一堆图”，而是：

> **得到一个可以直接喂给 Version 1 train 程序的数据集。**

## 3. 开始前准备

用户需要准备：

1. CS1.6 可以正常运行
2. 地图使用 `de_dust2`
3. 画面设置固定
4. 窗口分辨率固定
5. 已知本轮要采哪些区域标签

建议第一轮只采：

- `t_spawn`
- `ct_spawn`
- `mid`
- `a_site`
- `b_site`

等这 5 类稳定后，再扩大范围。

## 4. 一次完整采集的总体流程

建议按这个顺序执行：

1. 打开 CS1.6
2. 进入固定设置的 `de_dust2`
3. 启动 `cs_capture_session`
4. 在第一个区域设置当前 `place_token`
5. 在该区域内随意走动和转头
6. 到下一个区域时切换 `place_token`
7. 重复多次
8. 停止采集
9. 运行 `cs_dataset_build`
10. 运行 `cs_dataset_report`
11. 检查结果是否可用

## 5. 详细操作步骤

## Step 1：启动游戏

用户操作：

1. 打开 CS1.6
2. 进入固定地图 `de_dust2`
3. 确保窗口模式、分辨率、画面设置与之前一致

目标：

- 保证采集条件一致

## Step 2：启动采集工具

用户操作：

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

工具应完成：

- 找到游戏窗口
- 校验分辨率
- 创建 `session_0001`
- 开始截图

如果失败，应立即报错并停止。

## Step 3：设置第一个区域标签

例如当前人在 `t_spawn`。

用户操作：

```powershell
cs_label_session set `
  --session-root demo/cs/data/v1_area/raw `
  --session-id session_0001 `
  --place-token t_spawn
```

这一步之后，工具应开始把当前这段时间内的帧都标记为 `t_spawn`。

## Step 4：在该区域内自由移动

用户此时只需要：

- 前后走动
- 左右平移
- 左右转头
- 轻微改变视角

但不要：

- 快速乱甩视角过度制造模糊
- 频繁切出游戏窗口
- 乱切地图或改设置

采集目标是：

> **让同一个区域里出现尽量多样但仍然合理的视角。**

## Step 5：切换到下一个区域

例如从 `t_spawn` 到 `mid`。

当用户确认自己已经进入 `mid` 后，再执行：

```powershell
cs_label_session set `
  --session-root demo/cs/data/v1_area/raw `
  --session-id session_0001 `
  --place-token mid
```

这一步非常重要：

- 不要求逐帧标注
- 但要求区域切换时及时切 token

## Step 6：继续重复

用户继续重复：

1. 进入某区域
2. 设置该区域 token
3. 在区域内自由移动和观察

建议一次 session 覆盖 3~5 个区域，不必一次把所有区域都采完。

## Step 7：停止采集

采集结束后，用户执行：

```powershell
cs_capture_session stop --session-id session_0001
```

工具应完成：

- 结束截图
- 写入 `session.json`
- 写入最终状态文件
- 保留 `label_segments.json`

## Step 8：构建数据集

采完一个或多个 session 后，用户执行：

```powershell
cs_dataset_build run `
  --raw-root demo/cs/data/v1_area/raw `
  --output-root demo/cs/data/v1_area/processed `
  --min-frame-step 3 `
  --dedup on `
  --blur-filter on `
  --split train:val:test=8:1:1
```

工具应自动完成：

- 读取原始 session
- 读取标签段
- 抽样
- 去重
- 过滤坏帧
- 生成训练/验证/测试清单

## Step 9：生成报告

用户执行：

```powershell
cs_dataset_report run `
  --dataset-root demo/cs/data/v1_area/processed `
  --output-root demo/cs/data/v1_area/reports
```

工具应输出：

- 每类样本数量
- 每个 session 的覆盖情况
- 哪些类别样本不足

## Step 10：人工检查

用户最后需要做少量检查：

1. 类别是否明显失衡
2. 是否有区域完全缺样本
3. 是否有明显错误标签
4. 是否有大量黑屏或模糊图

如果发现问题：

- 不要直接进入训练
- 先补采或修正数据

## 6. 推荐的单次采集节奏

建议每次 session 控制在一个适中长度。  
不要过短，也不要长到不可管理。

建议策略：

- 一次 session 只采几个区域
- 每个区域停留一小段时间
- 多做几次 session

这样更利于：

- 后续切分 train / val / test
- 发现某次采集是否有问题

## 7. 用户在采集时的简单行为规范

用户只需要做简单行为，但还是要有基本规范。

建议：

- 在区域内自然移动
- 视角变化要有覆盖，但不过度极端
- 保持当前地图和设置不变
- 进入新区后再切 token

不建议：

- 一边采集一边频繁切设置
- 快速瞎转造成大量模糊帧
- token 切换严重滞后

## 8. 推荐的第一轮采集计划

建议第一轮只做一个最小可训练集。

### 第一天

- 采 `t_spawn`
- 采 `mid`
- 采 `a_site`

### 第二天

- 采 `ct_spawn`
- 采 `b_site`

### 第三天

- 构建数据集
- 看报告
- 决定补采哪些类别

这样做的好处是：

- 先尽快得到一个小而完整的可训练集
- 再逐步补充，而不是一开始就想采满整张地图

## 9. 采集完成的判断标准

对用户来说，一轮采集是否完成，不是看“我走了多久”，而是看：

1. 是否生成了有效 session
2. 是否生成了有效标签段
3. 是否成功构建出数据集清单
4. 是否报告中每类样本都有一定数量

只有这几步都成立，这轮采集才有价值。

## 10. 用户最少需要记住的事情

如果只记最核心的操作，可以记这 5 条：

1. 固定设置进入 `de_dust2`
2. 启动 `cs_capture_session`
3. 进入区域后设置正确的 `place_token`
4. 在区域里自然移动
5. 停止后运行 `cs_dataset_build` 和 `cs_dataset_report`

## 11. 下一步是什么

用户按本文档走完后，就应该进入：

> **Version 1 的训练与测试阶段**

也就是使用生成好的数据集，去训练区域识别模型，而不是继续无限制地采集数据。
