# 从 0 开始的完整用户故事

## 1. 读者和目标

读者：
- 第一次接触本项目的程序员
- 有软件工程经验
- 没有本项目背景知识

目标：
- 从需求分析开始
- 完成结构设计
- 完成配置
- 完成数据准备
- 完成训练
- 完成推理接入
- 完成测试验收

## 2. 场景定义

你要做一个控制任务。  
任务描述如下：
- 输入一条命令文本
- 输入当前状态向量
- 输出当前帧动作向量

示例业务：
- 命令是 `move left`、`move right fast`
- 状态是当前位置、目标误差、速度等
- 动作是开关动作和连续动作

## 3. 第 1 步：需求分析

### 你要做什么

写一页需求说明。内容只写四项：
1. 命令集合
2. 状态定义
3. 动作定义
4. 运行约束

### 你为什么要做

模型只负责映射。  
模型不会自动理解你的业务语义。  
你不先定义语义，后续训练和部署都会错。

### 这一步的产出

- `command_spec.md`
- `state_spec.md`
- `action_spec.md`
- `runtime_spec.md`

## 4. 第 2 步：接口冻结

### 你要做什么

冻结三件事：
- 动作维度
- 每个动作通道的语义
- 每个动作通道的值域

对应配置：
- `OUTPUT_DIM`
- `IO_MAPPING_NAMES`
- `IO_MAPPING_ACTIVATIONS`

配置文件：
- [config_user.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/config_user.h)

### 你为什么要做

这是系统接口。  
训练数据、推理代码、执行端都依赖它。  
中途改接口会导致全链路返工。

### 这一步的产出

- 冻结版动作接口表

## 5. 第 3 步：状态建模

### 你要做什么

定义状态向量每一维的含义。  
定义每一维的归一化方式。  
确定 `STATE_DIM`。

### 你为什么要做

状态是模型可见信息。  
信息不足，模型无法决策。  
信息混乱，模型会学到噪声。

### 这一步的产出

- 状态字段表
- 归一化规则表

## 6. 第 4 步：命令建模

### 你要做什么

整理命令词表。  
处理同义词。  
设置未登录词策略。  
确定 `VOCAB_SIZE` 和 `MAX_SEQ_LEN`。

### 你为什么要做

命令文本先变成 token ids。  
词表不稳定会导致输入分布漂移。  
序列长度不合理会造成截断或浪费。

### 这一步的产出

- 词表文件
- 命令规范

## 7. 第 5 步：结构初版设计

### 你要做什么

确定第一版结构规模：
- `EMBED_DIM`
- `NUM_LAYERS`
- `NUM_HEADS`
- `FFN_DIM`

建议：
- 先从小规模开始
- 先跑通闭环，再扩容

### 你为什么要做

结构规模决定容量和成本。  
你需要先得到可运行版本。  
你需要先建立可验证基线。

### 这一步的产出

- 第一版结构参数表

## 8. 第 6 步：资源预估

### 你要做什么

运行 profiler：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build --target profiler
.\build\profiler.exe
```

### 你为什么要做

你要确认部署可行性。  
你要确认时延和内存预算。  
你要避免训练后才发现跑不动。

### 这一步的产出

- 参数量
- 峰值激活内存
- FLOPs

## 9. 第 7 步：数据集设计与生成

### 你要做什么

按固定格式生成训练 CSV：
- `command,state[8],target[4]`

你要准备三类样本：
- 常规样本
- 边界样本
- 错误样本

### 你为什么要做

模型只能学习你给的数据分布。  
没有边界和错误样本，线上会失稳。

### 这一步的产出

- `train.csv`
- `eval.csv`

## 10. 第 8 步：训练实现

### 你要做什么

写你自己的训练入口文件。  
主流程如下：
1. 初始化 tokenizer
2. 加载 CSV
3. 前向计算
4. 计算损失
5. 更新权重
6. 打印指标

相关接口：
- [tokenizer.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/tokenizer.h)
- [csv_loader.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/csv_loader.h)
- [weights_io.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/weights_io.h)

### 你为什么要做

你的业务语义由你定义。  
训练入口必须和你的语义一致。  
示例代码只能提供结构参考。

### 这一步的产出

- 训练可执行程序
- 训练日志
- 权重文件

## 11. 第 9 步：推理接入

### 你要做什么

在你的业务主循环中接入推理。  
每一帧固定流程：
1. 构造 state
2. 编码命令
3. 前向推理
4. 激活映射
5. 动作下发

相关接口：
- [tokenizer.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/tokenizer.h#L127-L131)
- [ops.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/ops.h#L62-L69)
- [platform_driver.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/platform_driver.h)

### 你为什么要做

控制逻辑属于业务主循环。  
模型只负责单帧动作。  
这样才能保留安全控制权。

### 这一步的产出

- 可运行的业务推理入口

## 12. 第 10 步：测试与验收

### 你要做什么

运行全量测试：

```powershell
cmake --build build --target full_test_suite
.\build\full_test_suite.exe
```

重点看：
- 泛化
- OOD
- 对抗扰动
- 长时序稳定
- 预期错误路径

### 你为什么要做

跑通不代表可用。  
你需要验证稳定性和鲁棒性。  
你需要验证错误路径可控。

### 这一步的产出

- 测试日志
- 验收结论

## 13. 第 11 步：迭代策略

每次迭代只改一类变量：
1. 先改数据
2. 再改状态定义
3. 最后改结构容量

每次迭代都重复三件事：
1. 跑 profiler
2. 跑训练
3. 跑测试

## 14. 一页执行清单

按这个顺序执行：
1. 需求分析文档
2. 冻结动作接口
3. 定义状态和词表
4. 选第一版结构
5. 跑 profiler
6. 生成训练与评估数据
7. 写训练入口并训练
8. 写推理入口并接入外部循环
9. 跑全量测试
10. 输出验收报告
