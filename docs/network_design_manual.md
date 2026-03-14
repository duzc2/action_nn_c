# 网络结构设计手册

## 1. 目标

本文档讲“如何设计网络结构”，不是“如何填写配置项”。  
本文档按示例任务从需求分析开始，解释每个结构决策的原因、作用和取值影响。

## 2. 先看示例任务本质

示例任务不是“生成一句完整动作计划”，而是“每一帧做一次局部决策”。

任务输入：
- 命令文本（例如 `move right fast`）
- 当前状态（位置、目标误差等）

任务输出：
- 当前帧动作（4 维）

实现位置：
- 样例数据写入：[write_demo_training_csv](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c#L57-L85)
- 前向与训练入口：[predict_logits](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c#L155-L188)、[train_one_sample](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c#L234-L285)
- 外部循环：[run_external_goal_loop](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c#L398-L514)

结论：
- 这是控制问题，不是长文本生成问题。
- 模型核心职责是“当前帧动作映射”。

## 3. 从需求推导结构，而不是先选结构

先回答四个问题：
1. 输出动作是离散开关还是连续控制？
2. 决策依赖当前帧，还是长历史序列？
3. 指令词汇是否稳定、规模是否可控？
4. 时延预算和内存预算是多少？

本示例的答案：
- 输出是混合型（离散 + 连续）
- 主要依赖当前帧状态
- 词汇规模小且稳定
- 需要轻量和可解释

因此选择：
- 文本分支：token 平均表示
- 状态分支：线性映射
- 融合方式：加和 + bias
- 输出头：按通道激活映射

## 4. 为什么示例用“线性可解释头”

示例的参数布局：
- token 权重：`VOCAB_SIZE * OUTPUT_DIM`
- state 权重：`STATE_DIM * OUTPUT_DIM`
- bias：`OUTPUT_DIM`

定义位置：
- [DemoModelLayout](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c#L25-L30)

这么设计的原因：
- 参数总量可直接计算
- 每个权重对输出影响可解释
- 训练和排错路径短
- 便于演示“从数据到控制”的全链路

这不是说只能用线性头。  
它是示例中的“最小可验证结构”。

## 5. config_user.h 中每类结构参数的设计意义

配置文件：
- [config_user.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/config_user.h)

### 5.1 I/O 契约类参数

#### VOCAB_SIZE
- 决定可容纳的命令词数
- 过小会导致大量 `<unk>`
- 过大增加参数和存储

#### STATE_DIM
- 决定状态表达能力
- 过小会丢关键信息
- 过大可能引入噪声特征和过拟合

#### OUTPUT_DIM
- 决定执行器通道数
- 必须和执行端协议一一对应
- 训练前必须冻结语义

### 5.2 结构容量类参数

#### EMBED_DIM
- 决定 token 表示容量
- 增大可提高表达能力，也提高计算/参数

#### NUM_LAYERS
- 决定表示深度
- 增大可建模更复杂关系，但时延增加

#### NUM_HEADS
- 决定注意力分解粒度
- 过少可能表达不足，过多会增加开销

#### FFN_DIM
- 决定非线性映射容量
- 常见设为 `4 * EMBED_DIM`

#### MAX_SEQ_LEN
- 决定命令长度上限
- 过小截断命令，过大浪费资源

### 5.3 输出语义类参数

#### IO_MAPPING_ACTIVATIONS
- 决定每个输出通道的值域
- `Sigmoid` 适合开关量
- `Tanh` 适合连续控制量

#### IO_MAPPING_NAMES
- 决定日志和调试可读性
- 直接影响定位速度

## 6. 不同数量变化会产生什么效果

### 6.1 容量上调

上调 `EMBED_DIM / NUM_LAYERS / FFN_DIM` 的典型效果：
- 优点：表达能力提升，复杂关系更容易拟合
- 缺点：参数量、FLOPs、时延上升

### 6.2 维度上调

上调 `STATE_DIM` 的典型效果：
- 优点：可接入更多状态信息
- 风险：无效特征变多，训练更不稳定

上调 `OUTPUT_DIM` 的典型效果：
- 优点：可控制更多执行器
- 风险：监督难度提升，标签成本上升

### 6.3 长度上调

上调 `MAX_SEQ_LEN` 的典型效果：
- 优点：支持更长命令
- 风险：计算与内存开销增加

## 7. 设计流程（建议直接照用）

### 步骤 1：先冻结输出契约
- 定义 `OUTPUT_DIM`
- 定义每个通道语义
- 定义 `IO_MAPPING_ACTIVATIONS`

### 步骤 2：再定义状态表达
- 先保留最小必要状态
- 每个状态维度都要有业务解释

### 步骤 3：再定义命令空间
- 统计词汇并设置 `VOCAB_SIZE`
- 处理未登录词策略

### 步骤 4：最后定结构容量
- 从轻量规模起步
- 通过 profiler 评估资源
- 通过回归测试评估收益

## 8. 如何判断“该加大结构还是该改数据”

优先检查数据，不要先盲目加大结构。

先看这三件事：
1. 标签是否一致
2. 状态是否包含决策所需信息
3. 命令词是否稳定且可区分

只有在数据质量合格后，才增加结构容量。

## 9. profiler 在结构设计中的作用

profiler 不是“填完配置后跑一下就结束”。  
它是结构选型迭代工具。

你应记录：
- 参数量
- 峰值激活内存
- FLOPs

实现位置：
- [profiler.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/profiler.c#L178-L231)

## 10. 示例到业务的迁移方法

示例中你看到的是最小结构。  
迁移到业务时按下面做：
1. 保持外部循环架构不变
2. 替换你的状态定义和标签定义
3. 按资源预算调结构容量
4. 保持输出契约稳定
5. 用模型专项测试验证鲁棒性

不要做的事：
- 一开始就把结构拉满
- 在训练中途变更输出语义
- 只看 demo 是否能跑
