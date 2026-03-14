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
#include <stdlib.h>
#include <string.h>

/* 读取网络配置常量，如 VOCAB_SIZE / STATE_DIM / OUTPUT_DIM */
#include "../include/config_user.h"
/* 读取 CSV 数据集 */
#include "../include/csv_loader.h"
/* 文本编码接口 */
#include "../include/tokenizer.h"
/* 权重导出接口 */
#include "../include/weights_io.h"

int main(void) {
    /* 训练入口的核心对象：数据集、词表、编码器 */
    CsvDataset ds;
    Vocabulary vocab;
    Tokenizer tokenizer;
    int rc = 0;

    /* 先清零，避免未初始化字段导致随机错误 */
    memset(&ds, 0, sizeof(ds));
    memset(&vocab, 0, sizeof(vocab));
    memset(&tokenizer, 0, sizeof(tokenizer));

    /* 加载训练数据。无数据时立即退出。 */
    rc = csv_load_dataset("data/train.csv", &ds);
    if (rc != 0 || ds.count == 0U) {
        fprintf(stderr, "load train.csv failed rc=%d count=%zu\n", rc, ds.count);
        return 1;
    }

    /* 从 vocab.txt 读取词表。失败时释放已加载的数据。 */
    rc = vocab_load_text("vocab.txt", &vocab);
    if (rc != TOKENIZER_STATUS_OK) {
        fprintf(stderr, "load vocab.txt failed rc=%d\n", rc);
        csv_free_dataset(&ds);
        return 1;
    }

    /* 初始化 tokenizer。失败时释放词表和数据。 */
    rc = tokenizer_init(&tokenizer, &vocab, 0);
    if (rc != TOKENIZER_STATUS_OK) {
        fprintf(stderr, "tokenizer init failed rc=%d\n", rc);
        vocab_free(&vocab);
        csv_free_dataset(&ds);
        return 1;
    }

    /* 训练循环写在这里：
       1) tokenizer_encode 把命令转 token ids
       2) 前向计算得到 logits
       3) 计算 loss
       4) 反向更新参数
    */

    /* 训练完成后导出权重：
       - my_weights.bin 给运行时加载
       - my_weights.c 给编译内嵌
    */
    /* weights_save_binary("build/my_weights.bin", weights, weight_count); */
    /* weights_export_c_source("build/my_weights.c", "g_my_weights", weights, weight_count); */

    /* 释放资源 */
    vocab_free(&vocab);
    csv_free_dataset(&ds);
    return 0;
}
```

### 10.3 接入 CMake

打开 [CMakeLists.txt](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/CMakeLists.txt)。  
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

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 读取维度配置 */
#include "../include/config_user.h"
/* 文本编码接口 */
#include "../include/tokenizer.h"
/* 激活映射接口 */
#include "../include/ops.h"
/* 动作下发接口 */
#include "../include/platform_driver.h"
/* 权重加载接口 */
#include "../include/weights_io.h"

/* 单帧推理函数。外部循环每帧调用一次。 */
static int run_one_frame(const Tokenizer* tokenizer,
                         const float* weights,
                         const char* cmd,
                         const float* state,
                         float* out_act) {
    /* token 缓冲区大小由 MAX_SEQ_LEN 控制 */
    int ids[MAX_SEQ_LEN];
    size_t count = 0U;
    memset(ids, 0, sizeof(ids));
    /* 固定步骤：
       1) tokenizer_encode
       2) 前向计算
       3) op_actuator
       4) 写出 out_act
    */
    /* 骨架阶段先保留参数，避免编译告警 */
    (void)tokenizer;
    (void)weights;
    (void)cmd;
    (void)state;
    (void)out_act;
    return 0;
}

int main(void) {
    /* 主流程：
       1) 加载权重
       2) 从 vocab.txt 加载词表
       3) 初始化 tokenizer
       3) 外部循环逐帧执行
       4) 调用 driver_stub_apply 下发动作
       5) 命中停止条件后退出
    */
    /* 1) 加载权重
       2) 初始化 tokenizer
       3) while 循环:
            - 生成 state
            - run_one_frame
            - driver_stub_apply
            - 判断停止条件
    */
    return 0;
}
```

### 11.3 接入 CMake

在 [CMakeLists.txt](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/CMakeLists.txt) 加入：

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
