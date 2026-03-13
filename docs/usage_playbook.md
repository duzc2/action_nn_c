# 使用流程手册（简版）

本文件给出最短可执行流程。  
完整版本见 [application_manual.md](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/docs/application_manual.md)。

## 1. 配置模型

编辑 [config_user.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/config_user.h)：
- `VOCAB_SIZE`
- `STATE_DIM`
- `OUTPUT_DIM`
- `MAX_SEQ_LEN`
- `EMBED_DIM`
- `NUM_LAYERS`
- `NUM_HEADS`
- `FFN_DIM`
- `IO_MAPPING_ACTIVATIONS`

## 2. 预估资源

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build --target profiler
.\build\profiler.exe
```

查看输出：
- 参数量
- 峰值激活内存
- FLOPs

实现文件：  
[profiler.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/profiler.c)

## 3. 准备数据

训练数据格式：
- `command,state[8],target[4]`

参考文件：
- [c99_full_demo.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c)
- [csv_loader.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/train/csv_loader.c)

## 4. 训练并导出

```powershell
cmake --build build --target min_train_loop
.\build\min_train_loop.exe
```

输出文件：
- `build/min_weights.bin`
- `build/weights.c`

训练入口：  
[min_train_loop.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/min_train_loop.c)

## 5. 单步推理

步骤：
1. `tokenizer_encode`
2. 前向计算 logits
3. 激活映射
4. `driver_stub_apply`

关键接口：
- [tokenizer.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/tokenizer.h#L127-L131)
- [protocol.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/protocol.h#L50-L95)
- [platform_driver.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/platform_driver.h)

## 6. 多步推理（外部循环）

规则：
- 每帧只推理一次
- 每帧只执行一次动作
- 循环由外部控制

参考实现：  
[run_external_goal_loop](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c#L398-L514)

## 7. 回归验证

```powershell
cmake --build build --target full_test_suite
.\build\full_test_suite.exe
```

日志：
- `test/logs/test_run_*.log`

模型专项测试文件：  
[test_cases_model_special.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/test/src/test_cases_model_special.c)
