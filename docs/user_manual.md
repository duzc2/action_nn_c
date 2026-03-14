# 用户手册

## 1. 先建立认知

### 1.1 这个系统解决什么问题

它解决“每帧控制决策”问题：
- 输入：命令文本 + 状态向量
- 输出：本帧动作向量
- 运行方式：外部循环持续调用

### 1.2 什么时候用

适合：
- C99 环境部署
- 指令驱动控制
- 实时逐帧决策

不适合：
- 大规模训练平台
- GPU/分布式训练

### 1.3 责任边界

- 模型负责：当前帧动作决策
- 外部系统负责：任务循环、停止条件、安全策略

## 2. 系统核心概念

### 2.1 命令文本

命令通过 tokenizer 编码为 token ids。  
接口：`tokenizer_encode`  
[tokenizer.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/tokenizer.h#L127-L131)

### 2.2 状态向量

状态是连续特征向量。  
维度由 `STATE_DIM` 指定。  
默认是 8 维。  
[config_user.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/config_user.h#L36-L43)

### 2.3 动作向量

动作是输出向量。  
维度由 `OUTPUT_DIM` 指定。  
每个通道激活函数由 `IO_MAPPING_ACTIVATIONS` 指定。  
[config_user.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/config_user.h#L51-L56)

### 2.4 协议帧

两种输入协议：
- `RAW|<text>\n`
- `TOK|<count>|id0,id1,...\n`

接口：  
[protocol.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/protocol.h#L39-L95)

## 3. 系统执行流程

每一帧固定 5 步：
1. 文本编码：`tokenizer_encode`
2. 前向计算：`token + state -> logits`
3. 输出映射：`op_actuator`
4. 动作下发：`driver_stub_apply`
5. 外部循环判断是否继续

关键接口：
- [tokenizer.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/tokenizer.h)
- [ops.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/ops.h)
- [platform_driver.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/platform_driver.h)

## 4. config_user.h 全项说明

配置文件：  
[config_user.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/config_user.h)

### 4.1 网络规模

- `MAX_SEQ_LEN`：命令最大 token 数  
- `EMBED_DIM`：向量维度  
- `NUM_LAYERS`：层数  
- `NUM_HEADS`：注意力头数  
- `FFN_DIM`：前馈层维度

影响：主要决定计算量和内存占用。

### 4.2 I/O 维度

- `VOCAB_SIZE`：词表容量  
- `STATE_DIM`：状态维度  
- `OUTPUT_DIM`：动作维度

影响：决定输入输出契约，训练数据必须匹配。

### 4.3 输出语义

- `IO_MAPPING_ACTIVATIONS`：每个输出通道激活函数  
  - `0` = Sigmoid  
  - `1` = Tanh
- `IO_MAPPING_NAMES`：每个通道名称

影响：决定执行端如何解释每个输出值。

## 5. 先做配置，再做预估

构建并运行 profiler：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build --target profiler
.\build\profiler.exe
```

你要看三个指标：
- 参数量
- 峰值激活内存
- FLOPs

实现：  
[profiler.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/profiler.c)

## 6. 训练数据怎么设计

CSV 格式固定为：
- `command,state[8],target[4]`

数据加载接口：  
[csv_loader.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/csv_loader.h)

数据最小要求：
- 训练集和评估集要有分布差异
- 包含边界输入
- 包含错误输入

## 7. 怎么训练自己的网络

训练示例入口：  
[min_train_loop.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/min_train_loop.c)

运行：

```powershell
cmake --build build --target min_train_loop
.\build\min_train_loop.exe
```

训练产物：
- `build/min_weights.bin`
- `build/weights.c`

权重接口：
- [weights_io.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/weights_io.h)

## 8. 怎么写自己的业务代码

不要把 demo 当成唯一入口。  
正确做法是按调用链自己写业务入口：

1. 初始化词表与 tokenizer  
2. 加载权重  
3. 在你的外部循环里逐帧调用推理  
4. 每帧更新状态并执行动作  
5. 在外部判断结束条件

参考代码：
- 全链路参考：[c99_full_demo.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c)
- 外部循环实现：[run_external_goal_loop](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c#L398-L514)

## 9. 完整例子：目标点控制

目标：从 `(0,0)` 移动到 `(15,15)`。  
每帧输入当前误差，输出一步动作。  
外部循环决定是否继续下一帧。

运行例子：

```powershell
cmake --build build --target c99_full_demo
.\build\c99_full_demo.exe
```

你需要观察：
- loss 是否下降
- 权重是否导出成功
- 帧日志中位姿是否向目标收敛

## 10. 测试与验收

运行全量测试：

```powershell
cmake --build build --target full_test_suite
.\build\full_test_suite.exe
```

日志目录：
- `test/logs/`

重点看模型专项测试：
- 泛化
- OOD
- 对抗扰动
- 长时序
- 预期错误路径

## 11. 常见问题

### 11.1 乱码

```powershell
chcp 65001
```

### 11.2 构建失败

先清理 build 目录后重新配置。  
如果生成器冲突，使用新的 build 目录。

### 11.3 我该先改哪里

按顺序改：
1. `src/include/config_user.h`
2. 你的训练数据
3. 你的业务入口代码

## 12. 学习路径

学习顺序：
1. 看本手册第 1~4 章
2. 做第 5~7 章
3. 按第 8 章写你的业务入口
4. 用第 10 章做回归验收

网络结构专题阅读：
- [network_design_manual.md](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/docs/network_design_manual.md)
- 重点章节：
  - 第 0 章：零基础术语课
  - 第 5.4 节：参数语义层次
  - 第 6 章：五个设计坐标轴

完整用户故事阅读：
- [user_story_from_zero.md](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/docs/user_story_from_zero.md)
