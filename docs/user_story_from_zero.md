# 从 0 开始的完整用户故事

## 1. 课程说明

这是一份实操课程。  
你会从空白需求走到可运行系统。  
你不需要先看 demo 源码。  
你按章节执行即可。

课程结果：
- 你会写出一份需求规格
- 你会写出配置和数据
- 你会得到可训练、可推理、可测试的工程链路

## 2. 课程场景

课程使用统一业务场景。  
场景名称：二维移动控制。

业务目标：
- 输入命令文本
- 输入当前位置和目标位置
- 输出当前帧动作

命令示例：
- `move left`
- `move right fast`
- `move stop`

状态示例：
- 当前 x、当前 y
- 目标误差 dx、dy
- 速度和辅助信号

动作示例：
- 离散动作：是否左移、是否右移
- 连续动作：x 方向步长、y 方向步长

## 3. 第 1 课：写需求规格

### 3.1 你要创建的文件

在项目根目录新建 `project_spec.md`。  
复制下面模板并填写。

```markdown
# 需求规格

## 任务目标
- 目标描述：
- 成功标准：

## 输入定义
- 命令集合：
- 状态字段：

## 输出定义
- 动作字段：
- 每个字段值域：

## 运行约束
- 最大时延：
- 最大内存：
- 最大帧数：
```

### 3.2 为什么先做这一步

模型不会理解你的业务目标。  
训练和部署都依赖规格定义。  
规格缺失会导致后续每一步返工。

### 3.3 完成标准

你能回答三件事：
- 模型输入是什么
- 模型输出是什么
- 系统什么时候停止

## 4. 第 2 课：冻结动作接口

### 4.1 动作接口表

在 `project_spec.md` 增加一张表。  
表结构如下。

```markdown
| 通道 | 名称 | 类型 | 值域 | 含义 |
|---|---|---|---|---|
| 0 | Move_Left  | Binary | 0~1   | 是否左移 |
| 1 | Move_Right | Binary | 0~1   | 是否右移 |
| 2 | Step_X     | Analog | -1~1  | x 方向步长 |
| 3 | Step_Y     | Analog | -1~1  | y 方向步长 |
```

### 4.2 对应配置项

在 [config_user.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/config_user.h) 设置：
- `OUTPUT_DIM`
- `IO_MAPPING_NAMES`
- `IO_MAPPING_ACTIVATIONS`

### 4.3 完成标准

训练数据、推理代码、执行端都使用同一份动作接口定义。

## 5. 第 3 课：定义状态向量

### 5.1 状态字段表模板

在 `project_spec.md` 增加状态表。

```markdown
| 索引 | 字段名 | 物理含义 | 取值范围 | 归一化 |
|---|---|---|---|---|
| 0 | pos_x | 当前 x | 0~20 | /20 |
| 1 | pos_y | 当前 y | 0~20 | /20 |
| 2 | err_x | 目标误差 x | -20~20 | /20 |
| 3 | err_y | 目标误差 y | -20~20 | /20 |
| 4 | speed_x | x 速度 | -5~5 | /5 |
| 5 | speed_y | y 速度 | -5~5 | /5 |
| 6 | dist | 距离 | 0~30 | /30 |
| 7 | bias | 常数项 | 1 | 1 |
```

### 5.2 对应配置项

- `STATE_DIM` 必须等于表行数。  
- 训练数据和推理代码必须使用同一顺序。

### 5.3 完成标准

任意状态向量都能被人读懂。  
任意索引都有固定语义。

## 6. 第 4 课：定义命令词表

### 6.1 命令清单模板

新建 `vocab.txt`。  
每行一个 token。

`<unk>` 是未知词占位符。  
当输入命令里出现词表中没有的词时，tokenizer 会把该词映射到 `<unk>`。  
你必须保留 `<unk>`，否则未知词无法被稳定处理。

```text
<unk>
move
left
right
stop
fast
slow
```

### 6.2 配置项

- `VOCAB_SIZE` >= token 数量
- `MAX_SEQ_LEN` >= 命令最大 token 数

### 6.3 完成标准

你能把业务命令稳定映射成 token ids。  
同义词策略已经确定。

## 7. 第 5 课：确定第一版网络规模

### 7.1 第一版建议值

先使用一组保守值：
- `EMBED_DIM=32`
- `NUM_LAYERS=2`
- `NUM_HEADS=2`
- `FFN_DIM=128`

### 7.2 为什么先小后大

小规模先验证链路。  
小规模先验证数据是否有效。  
链路正确后再扩容。

### 7.3 完成标准

你有一组明确的初版参数。  
你知道后续扩容方向。

## 8. 第 6 课：运行资源预估

### 8.1 执行命令

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build --target profiler
.\build\profiler.exe
```

### 8.2 记录表

把输出填到 `resource_report.md`：

```markdown
| 指标 | 数值 | 预算 | 结论 |
|---|---:|---:|---|
| 参数量 |  |  |  |
| 峰值激活内存 |  |  |  |
| FLOPs |  |  |  |
```

### 8.3 完成标准

所有指标在预算以内。  
否则回到第 5 课调整结构。

## 9. 第 7 课：生成训练数据

### 9.1 数据格式

固定格式：
- `command,state[8],target[4]`

示例行：

```csv
move left,0.8,0.2,-0.4,0.1,0.0,0.0,0.5,1.0,1.0,0.0,-0.5,0.1
move right fast,0.2,0.8,0.6,-0.2,0.1,0.0,0.7,1.0,0.0,1.0,0.8,0.2
move stop,0.5,0.5,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0
```

### 9.2 数据集拆分

你至少需要三份数据：
- `train.csv`
- `eval.csv`
- `badcase.csv`

### 9.3 完成标准

数据覆盖常规、边界、错误三类输入。

## 10. 第 8 课：实现训练程序

### 10.1 目标

你需要一个自己的训练入口。  
文件名建议：`src/tools/my_train_loop.c`。

### 10.2 训练程序固定流程

1. 初始化词表与 tokenizer  
2. 读取 CSV  
3. 前向计算  
4. 计算 loss  
5. 更新参数  
6. 导出权重

### 10.3 关键接口

- [tokenizer.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/tokenizer.h)
- [csv_loader.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/csv_loader.h)
- [ops.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/ops.h)
- [weights_io.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/weights_io.h)

### 10.4 CMake 接入步骤

在 [CMakeLists.txt](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/CMakeLists.txt) 增加：

```cmake
add_executable(my_train_loop
    src/tools/my_train_loop.c
)
target_include_directories(my_train_loop PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
target_link_libraries(my_train_loop PRIVATE dnn_core)
```

### 10.5 训练执行命令

```powershell
cmake --build build --target my_train_loop
.\build\my_train_loop.exe
```

### 10.6 完成标准

你看到稳定的 loss 输出。  
你得到权重文件和导出 C 文件。

## 11. 第 9 课：实现推理程序

### 11.1 目标

你需要一个自己的推理入口。  
文件名建议：`src/tools/my_infer_app.c`。

### 11.2 单帧推理流程

1. 读取当前命令  
2. 构造当前 state  
3. 编码命令  
4. 前向推理  
5. 激活映射  
6. 动作下发

### 11.3 外部循环流程

1. 维护任务状态  
2. 每帧调用单帧推理  
3. 每帧更新状态  
4. 判断停止条件

### 11.4 关键接口

- [tokenizer_encode](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/tokenizer.h#L127-L131)
- [op_actuator](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/ops.h#L62-L69)
- [driver_stub_apply](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/platform_driver.h#L52)

### 11.5 完成标准

每帧日志可追踪。  
动作语义和规格一致。  
任务可按条件停止。

## 12. 第 10 课：跑测试并验收

### 12.1 执行命令

```powershell
cmake --build build --target full_test_suite
.\build\full_test_suite.exe
```

### 12.2 你要检查的项目

- 泛化样本结果
- OOD 样本结果
- 对抗扰动样本结果
- 长时序稳定性
- 错误路径行为

### 12.3 验收报告模板

新建 `acceptance_report.md`：

```markdown
# 验收报告

## 功能验收
- 结论：

## 性能验收
- 结论：

## 稳定性验收
- 结论：

## 风险项
- 列表：
```

## 13. 第 11 课：迭代方法

每轮迭代只改一类变量：
1. 先改数据
2. 再改状态定义
3. 最后改网络规模

每轮迭代都重复四步：
1. profiler
2. 训练
3. 推理回归
4. 全量测试

## 14. 全流程执行顺序

1. 写需求规格
2. 冻结动作接口
3. 定义状态向量
4. 定义命令词表
5. 设定初版结构
6. 跑资源预估
7. 生成训练数据
8. 写训练程序并训练
9. 写推理程序并接入外部循环
10. 跑测试并输出验收报告

## 15. 你现在可以怎么开始

今天直接做三件事：
1. 新建 `project_spec.md`
2. 写完整动作接口表和状态字段表
3. 运行 profiler，记录第一版资源报告
