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

在 [config_user.h](../src/include/config_user.h) 设置：
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

词表来源规则：
- `vocab.txt` 是唯一词表来源
- 训练和推理都读取同一个 `vocab.txt`
- 不在代码里手写第二份 token 列表

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

三份数据的作用不同：
- `train.csv`：用于训练参数。模型在这份数据上更新权重。
- `eval.csv`：用于评估效果。只做前向计算，不更新权重。它用于检查泛化能力。
- `badcase.csv`：用于错误和极端输入验证。它用于检查系统在异常条件下是否稳定。

为什么必须拆分：
- 只用 `train.csv` 会高估效果。模型可能只记住训练样本。
- 有 `eval.csv` 才能判断模型在未见样本上的表现。
- 有 `badcase.csv` 才能提前发现线上风险。

### 9.2.1 三份数据怎么写

#### train.csv 写法

目标：让模型学会主要任务行为。  
写法要求：
- 覆盖主要命令组合
- 覆盖常见状态区间
- 样本数量最多
- 标签要稳定，不互相冲突

推荐比例：
- 约占总样本的 70%~80%

#### eval.csv 写法

目标：验证泛化能力。  
写法要求：
- 不要和 train.csv 重复行
- 使用相同字段定义和同样归一化规则
- 分布与训练接近，但不是完全相同
- 加入少量插值样本和跨区间样本

推荐比例：
- 约占总样本的 15%~20%

#### badcase.csv 写法

目标：验证系统在异常条件下的行为。  
写法要求：
- 包含边界值状态
- 包含罕见命令组合
- 包含噪声状态和冲突状态
- 保持格式合法，避免 CSV 结构错误

推荐比例：
- 约占总样本的 5%~10%

### 9.2.2 写数据时的统一规则

所有数据文件都要满足：
- 字段顺序完全一致
- 归一化规则完全一致
- `target` 必须满足动作值域约束
- `command` 文本与词表策略一致

值域检查：
- Sigmoid 通道目标值在 `[0,1]`
- Tanh 通道目标值在 `[-1,1]`

### 9.2.3 常见错误

- 训练集和评估集重复样本太多
- badcase 文件里混入大量常规样本
- 归一化规则在三个文件中不一致
- 同一输入对应多个冲突标签
- target 越界

### 9.2.4 最小可用模板

```text
data/
  train.csv
  eval.csv
  badcase.csv
```

先按这个模板创建三个文件。  
再分别写入常规、评估、异常样本。

### 9.3 完成标准

数据覆盖常规、边界、错误三类输入。

## 10. 第 8 课：实现训练程序（可直接执行）

### 10.1 目标

你要实现一个可执行训练程序。  
程序文件名使用 `src/tools/my_train_loop.c`。  
程序输入是 `train.csv`。  
程序输出是权重文件。

### 10.2 先创建文件

创建文件：
- `src/tools/my_train_loop.c`

把下面代码骨架复制到文件中，再按你的业务补全。

```c
#include <stdio.h>
#include "../include/workflow.h"

int main(void) {
    WorkflowTrainOptions options;
    int rc = 0;
    memset(&options, 0, sizeof(options));
    options.csv_path = "data/train.csv";
    options.vocab_path = "vocab.txt";
    options.out_weights_bin = "build/my_weights.bin";
    options.out_weights_c = "build/my_weights.c";
    options.out_symbol = "g_my_weights";
    options.epochs = 20U;
    options.learning_rate = 0.05f;

    rc = workflow_train_from_csv(&options);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_train_from_csv failed rc=%d\n", rc);
        return 1;
    }
    printf("train done\n");
    return 0;
}
```

### 10.3 接入 CMake

打开 [CMakeLists.txt](../CMakeLists.txt)。  
加入下面配置：

```cmake
add_executable(my_train_loop
    src/tools/my_train_loop.c
)
target_include_directories(my_train_loop PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
target_link_libraries(my_train_loop PRIVATE dnn_core)
```

### 10.4 执行训练

```powershell
cmake --build build --target my_train_loop
.\build\my_train_loop.exe
```

### 10.5 训练成功检查

你要看到以下结果：
- 控制台有 epoch 和 loss 输出
- `build/my_weights.bin` 存在
- `build/my_weights.c` 存在

如果失败，按顺序检查：
1. `data/train.csv` 是否存在
2. CSV 列数是否正确
3. `STATE_DIM` 和 `OUTPUT_DIM` 是否匹配数据

## 11. 第 9 课：实现推理程序（可直接执行）

### 11.1 目标

你要实现一个推理应用。  
程序文件名使用 `src/tools/my_infer_app.c`。  
程序读取权重并执行外部循环。

### 11.2 创建代码骨架

这段 `build_input` 是推理输入回调。  
它不参与训练，不会写入模型参数。  
训练只读取第 10 课的 `train.csv / eval.csv / badcase.csv`。

```c
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../include/workflow.h"

typedef struct AppContext {
    float x;
    float y;
    float goal_x;
    float goal_y;
} AppContext;

static int build_input(size_t frame,
                       char* out_command,
                       size_t command_capacity,
                       float* out_state,
                       size_t state_dim,
                       void* user_data) {
    AppContext* ctx = (AppContext*)user_data;
    float dx = 0.0f;
    float dy = 0.0f;
    (void)frame;
    if (ctx == NULL || out_command == NULL || out_state == NULL) return -1;
    if (state_dim < 8U || command_capacity < 16U) return -1;

    dx = ctx->goal_x - ctx->x;
    dy = ctx->goal_y - ctx->y;
    if (fabsf(dx) < 0.3f && fabsf(dy) < 0.3f) return 1;

    if (dx > 2.0f) strcpy(out_command, "move right fast");
    else if (dx > 0.0f) strcpy(out_command, "move right");
    else if (dx < -2.0f) strcpy(out_command, "move left fast");
    else if (dx < 0.0f) strcpy(out_command, "move left");
    else strcpy(out_command, "move stop");

    out_state[0] = ctx->x / 20.0f;
    out_state[1] = ctx->y / 20.0f;
    out_state[2] = dx / 20.0f;
    out_state[3] = dy / 20.0f;
    out_state[4] = fabsf(dx) / 20.0f;
    out_state[5] = fabsf(dy) / 20.0f;
    out_state[6] = (fabsf(dx) + fabsf(dy)) / 40.0f;
    out_state[7] = 1.0f;
    return 0;
}

static int on_action(const float* action_values, size_t action_count, void* user_data) {
    AppContext* ctx = (AppContext*)user_data;
    size_t i = 0U;
    float dx = 0.0f;
    float dy = 0.0f;
    float step_x = 0.0f;
    float step_y = 0.0f;
    if (ctx == NULL || action_values == NULL || action_count < 4U) return -1;

    dx = ctx->goal_x - ctx->x;
    dy = ctx->goal_y - ctx->y;
    step_x = action_values[2] * 0.5f;
    step_y = ((action_values[3] + 1.0f) * 0.5f) * 0.4f;
    if (fabsf(step_x) > fabsf(dx)) step_x = dx;
    if (fabsf(step_y) > fabsf(dy)) step_y = dy;
    ctx->x += step_x;
    ctx->y += step_y;

    printf("[action]");
    for (i = 0U; i < action_count; ++i) {
        printf(" ch%zu=%.4f", i, (double)action_values[i]);
    }
    printf(" pose=(%.3f,%.3f)\n", (double)ctx->x, (double)ctx->y);
    return 0;
}

int main(void) {
    WorkflowRuntime runtime;
    AppContext ctx;
    size_t frame = 0U;
    int rc = 0;
    memset(&runtime, 0, sizeof(runtime));
    memset(&ctx, 0, sizeof(ctx));
    ctx.x = 0.0f;
    ctx.y = 0.0f;
    ctx.goal_x = 15.0f;
    ctx.goal_y = 15.0f;

    rc = workflow_runtime_init(&runtime, "vocab.txt", "build/my_weights.bin");
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_runtime_init failed rc=%d\n", rc);
        return 1;
    }

    for (frame = 0U; frame < 300U; ++frame) {
        char command[128];
        float state[8];
        float act[4];
        int brc = 0;
        memset(command, 0, sizeof(command));
        memset(state, 0, sizeof(state));
        memset(act, 0, sizeof(act));

        brc = build_input(frame, command, sizeof(command), state, 8U, &ctx);
        if (brc > 0) break;
        if (brc < 0) {
            workflow_runtime_shutdown(&runtime);
            return 1;
        }
        rc = workflow_run_step(&runtime, command, state, act);
        if (rc != WORKFLOW_STATUS_OK) {
            workflow_runtime_shutdown(&runtime);
            fprintf(stderr, "workflow_run_step failed rc=%d\n", rc);
            return 1;
        }
        if (on_action(act, 4U, &ctx) != 0) {
            workflow_runtime_shutdown(&runtime);
            return 1;
        }
    }

    workflow_runtime_shutdown(&runtime);
    printf("infer done\n");
    return 0;
}
```

`build_input` 只用于推理阶段动态生成输入。  
训练阶段仍然使用第 10 课的 `train.csv` 数据。

### 11.3 接入 CMake

在 [CMakeLists.txt](../CMakeLists.txt) 加入：

```cmake
add_executable(my_infer_app
    src/tools/my_infer_app.c
)
target_include_directories(my_infer_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
target_link_libraries(my_infer_app PRIVATE dnn_core)
```

### 11.4 推理执行

```powershell
cmake --build build --target my_infer_app
.\build\my_infer_app.exe
```

### 11.5 推理成功检查

你要看到：
- 每帧日志
- 当前命令
- 当前状态摘要
- 当前动作向量
- 停止条件触发日志

### 11.6 谁在调用执行层，怎么对接

当前调用链：
1. 你的 `my_infer_app.c` 调 `workflow_runtime_init`
2. 你的外部循环每帧准备 `command + state`
3. 你的外部循环调 `workflow_run_step`
4. 你的外部循环调执行层（例如 `on_action`）
5. 结束时调 `workflow_runtime_shutdown`

对应代码位置：
- 运行时初始化接口：[workflow.h](../src/include/workflow.h)
- 单步推理接口：[workflow.h](../src/include/workflow.h)
- 接口实现：[workflow.c](../src/core/workflow.c)

当前行为说明：
- 什么时候跑下一帧由外部循环决定
- 每帧输入是你传给 `workflow_run_step` 的参数
- 工作流不再内置循环节拍决策

如何切换执行端：
- 回调 A：`on_action` 里调用执行端 A 的 SDK
- 回调 B：`on_action` 里调用执行端 B 的 SDK

如何对接真实执行层：
1. 在业务循环里读取你的真实输入，组装 `command + state`
2. 调 `workflow_run_step(&runtime, command, state, act)`
3. 把 `act` 交给你的设备 SDK
4. 当你判断应停止时，结束循环并 `workflow_runtime_shutdown`

### 11.7 导出文件怎么用（按场景选）

训练后你会得到两类导出物：
- `build/my_weights.bin`
- `build/my_weights.c`

另外你也可以导出函数网络文件（可选）：
- `build/my_network_functions.c`

#### 场景 A：常规上线 / 可热更新（推荐）

使用 `build/my_weights.bin`。  
适合要替换模型、做版本回滚、做远程下发的场景。

调用方式：
```c
WorkflowRuntime runtime;
float act[4];
workflow_runtime_init(&runtime, "vocab.txt", "build/my_weights.bin");
workflow_run_step(&runtime, command, state, act);
on_action(act, 4U, &ctx);
workflow_runtime_shutdown(&runtime);
```

#### 场景 B：固件内嵌常量权重（不走文件系统）

使用 `build/my_weights.c`。  
它导出的是数据模块函数，不是网络结构函数：
- `<symbol>_count()`
- `<symbol>_data()`
- `<symbol>_copy()`

调用方式（示例）：
```c
size_t n = g_my_weights_count();
const float* w = g_my_weights_data();
```

适合 ROM 固化、无文件系统环境。  
你在业务代码里直接使用 `w` 做前向计算，或拷贝到你自己的运行缓存。

#### 场景 C：函数网络直调（节点函数图）

使用 `build/my_network_functions.c`（可选导出）。  
适合追求固定拓扑、无动态加载、代码调用路径可控的场景。

调用方式（示例）：
```c
float token_onehot[32U * 128U] = {0};
float state[8] = {0};
float out[4] = {0};
g_my_network_forward(token_onehot, state, out);
```

导出该文件需要你在“持有 weights 数组”的训练代码里调用：
```c
weights_export_c_function_network("build/my_network_functions.c",
                                  "g_my_network",
                                  weights,
                                  VOCAB_SIZE,
                                  MAX_SEQ_LEN,
                                  STATE_DIM,
                                  OUTPUT_DIM,
                                  activations);
```

#### 三种方式怎么选

- 优先用 `bin`：工程化最稳，切换模型最方便
- 要内嵌常量就用 `weights.c`：部署简单、无文件依赖
- 要函数图就用 `network_functions.c`：结构固定、可做极限场景优化

同一个项目可以三种并存，运行时按配置选择加载路径。

## 12. 第 10 课：测试与验收（逐项执行）

### 12.1 执行测试命令

```powershell
cmake --build build --target full_test_suite
.\build\full_test_suite.exe
```

### 12.2 验收检查表

新建 `acceptance_checklist.md`，填下面表格：

```markdown
| 项目 | 检查方法 | 结果 | 备注 |
|---|---|---|---|
| 功能正确 | 回放常规样本 |  |  |
| 泛化能力 | eval.csv 指标 |  |  |
| 异常稳定 | badcase.csv 回放 |  |  |
| 时延达标 | 单帧耗时统计 |  |  |
| 内存达标 | profiler 报告 |  |  |
```

### 12.3 输出验收报告

新建 `acceptance_report.md`，按模板填写：

```markdown
# 验收报告

## 1. 输入输出契约
- 是否冻结：
- 是否一致：

## 2. 训练结果
- 最终 loss：
- 权重文件：

## 3. 推理结果
- 是否稳定运行：
- 停止条件是否生效：

## 4. 风险项
- 风险 1：
- 风险 2：
```

## 13. 第 11 课：迭代方法（一轮一轮做）

每轮只改一个维度。  
你按下面顺序执行：
1. 改数据
2. 跑训练
3. 跑推理
4. 跑测试
5. 记录结果

迭代记录模板：

```markdown
| 轮次 | 仅修改项 | 训练结果 | 推理结果 | 测试结果 | 结论 |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
```

## 14. 全流程执行命令总表

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build --target profiler
.\build\profiler.exe
cmake --build build --target my_train_loop
.\build\my_train_loop.exe
cmake --build build --target my_infer_app
.\build\my_infer_app.exe
cmake --build build --target full_test_suite
.\build\full_test_suite.exe
```

## 15. 今天的落地任务

按这个清单执行：
1. 创建 `project_spec.md`
2. 创建 `data/train.csv`、`data/eval.csv`、`data/badcase.csv`
3. 创建 `src/tools/my_train_loop.c`
4. 创建 `src/tools/my_infer_app.c`
5. 修改 `CMakeLists.txt` 增加两个目标
6. 运行第 14 章命令总表
7. 生成 `acceptance_report.md`
