# 应用手册

## 1. 文档目的

本文档说明如何把本项目用于真实任务。  
范围包括：模型设计、配置、资源预估、训练、部署、单步推理、多步推理。

## 2. 适用条件

满足以下条件再使用本项目：
- 需要 C99 实现
- 输入为文本命令和状态向量
- 输出为动作向量
- 控制循环由外部系统驱动

不满足以下需求：
- 大规模深度学习训练
- GPU 训练和自动微分

## 3. 工作流程总览

按以下顺序执行：
1. 定义任务 I/O
2. 配置模型参数
3. 用 profiler 估算资源
4. 准备训练数据
5. 训练并导出权重
6. 部署单步推理
7. 接入外部循环多步推理
8. 运行模型专项测试

## 4. 第一步：定义任务 I/O

必须先定义三项：
- 指令集合，例如 `move left/right/stop/fast/slow`
- 状态向量每一维的物理含义
- 输出向量每一维的控制含义

配置文件：
- [config_user.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/config_user.h)

重点配置项：
- `STATE_DIM`
- `OUTPUT_DIM`
- `VOCAB_SIZE`
- `IO_MAPPING_ACTIVATIONS`

规则：
- 开关量输出使用 Sigmoid
- 连续量输出使用 Tanh

## 5. 第二步：配置模型参数

在 `config_user.h` 设置以下参数：
- `MAX_SEQ_LEN`
- `EMBED_DIM`
- `NUM_LAYERS`
- `NUM_HEADS`
- `FFN_DIM`

建议流程：
1. 先用小参数跑通
2. 再逐步增大规模
3. 每次改参数后重新跑 profiler 和测试

## 6. 第三步：训练前资源预估

构建并运行 profiler：

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build --target profiler
.\build\profiler.exe
```

可选输出路径：

```powershell
.\build\profiler.exe -o src/include/network_def.h
```

输出指标：
- 参数量
- 峰值激活内存
- FLOPs
- 输出通道激活配置

实现文件：
- [profiler.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/profiler.c)

## 7. 第四步：准备训练数据

训练 CSV 格式：
- `command,state[8],target[4]`

参考实现：
- [c99_full_demo.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c)
- [csv_loader.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/train/csv_loader.c)

数据要求：
- 训练集和评估集分布不同
- 包含边界样本
- 包含异常样本
- 包含负向样本

## 8. 第五步：训练并导出权重

训练入口参考：
- [min_train_loop.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/min_train_loop.c)

训练主步骤：
1. 初始化词表和 tokenizer
2. 读取 CSV
3. 前向计算
4. 计算损失
5. 参数更新
6. 记录训练指标

导出接口：
- `weights_save_binary`
- `weights_export_c_source`

接口定义：
- [weights_io.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/weights_io.h)

## 9. 第六步：部署单步推理

单步推理顺序：
1. 编码命令
2. 执行前向
3. 激活映射
4. 下发动作

关键接口：
- [tokenizer_encode](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/tokenizer.h#L127-L131)
- [protocol_encode_raw / protocol_decode_packet](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/protocol.h#L50-L95)
- [platform_driver.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/platform_driver.h)

单步推理模板：

```c
tokenizer_encode(&tokenizer, command, ids, cap, &count);
predict_logits(weights, ids, count, state, logits);
activate_output(logits, act);
driver_stub_apply(&driver, act, OUTPUT_DIM);
```

## 10. 第七步：一条命令多步推理

原则：模型不维护循环，外部系统维护循环。

外部循环顺序：
1. 读取当前位置和目标位置
2. 生成本帧 state
3. 调用一次模型推理
4. 执行一次动作
5. 判断是否到达
6. 未到达则下一帧重复

参考实现：
- [run_external_goal_loop](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c#L398-L514)

停止条件建议：
- 误差小于阈值
- 或达到最大帧数

## 11. 第八步：验证可用性

运行全量测试：

```powershell
cmake --build build --target full_test_suite
.\build\full_test_suite.exe
```

日志路径：
- `test/logs/test_run_*.log`

必须关注模型专项测试：
- 泛化
- OOD
- 对抗扰动
- 长时序极限
- 预期错误路径

模型专项文件：
- [test_cases_model_special.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/test/src/test_cases_model_special.c)

## 12. 部署前检查清单

发布前必须全部满足：
- 参数配置已冻结
- profiler 指标符合设备资源
- 训练与评估指标达标
- 负向测试通过
- 权重导出和回加载一致
- 单步和多步推理都通过回归测试

