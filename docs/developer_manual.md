# 开发手册

## 1. 开发目标

本仓库面向“可解释、可移植、可验证”的 C99 模型工程实践。  
核心原则：
- 不依赖外部深度学习运行时
- 训练/推理/导出/测试全链路在仓库内闭环
- 外部循环驱动控制，模型只负责单帧决策

## 2. 架构总览

### 2.1 模块分层

- `src/include/`：公共头文件与接口声明
- `src/core/`：核心算子与兼容入口
- `src/tokenizer/`：分词、词表、运行时封装
- `src/platform/`：平台驱动核心 + PC/ESP32 封装
- `src/model/`：模型计算逻辑
- `src/train/`：CSV 读取与训练输入构造
- `src/tools/`：工具与 Demo 程序
- `test/`：测试框架与分组测试

### 2.2 数据流

1. 文本命令经过 tokenizer 编码为 token ids  
2. 状态向量与 token 特征进入模型前向  
3. 经过输出映射得到执行器动作  
4. 外部控制器执行单帧动作并决定下一帧

## 3. 构建与运行

### 3.1 配置

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
```

### 3.2 构建目标

```powershell
cmake --build build --target profiler
cmake --build build --target min_train_loop
cmake --build build --target c99_full_demo
cmake --build build --target full_test_suite
```

### 3.3 运行

```powershell
.\build\profiler.exe
.\build\min_train_loop.exe
.\build\c99_full_demo.exe
.\build\full_test_suite.exe
```

## 4. 测试体系说明

### 4.1 测试分组

测试入口位于：
- `test/src/test_suite.c`

当前分组包括：
- 单元
- 正确性
- 错误
- 边界
- 压力
- 集成
- 模型专项

### 4.2 模型专项目标

模型专项测试不只验证“必然正确样例”，还覆盖：
- 泛化（插值、跨区间）
- OOD（远分布输入）
- 对抗扰动（状态/Token）
- 极限闭环（长时序、多目标切换、延迟噪声、故障恢复）
- 预期错误路径（空命令、超长 token、损坏协议包）

### 4.3 日志规范

日志文件位于 `test/logs/`，每次运行生成一个新文件。  
日志包含：
- case_id / 分类 / 名称
- purpose / params / expected
- 实际结果与断言统计
- 分类汇总与总汇总

## 5. 如何新增一个模型测试

1. 在 `test/src/test_cases_model_special.c` 新增 `static int test_xxx(void)`  
2. 用 `TFW_ASSERT_*` 添加断言，避免只打印不校验  
3. 用 `testfw_log_info` 输出关键指标  
4. 在 `testcases_get_model_special_group()` 注册该用例  
5. 构建并运行 `full_test_suite`

## 6. 如何新增模型能力

1. 在 `src/model/` 或 `src/core/` 增加实现  
2. 在 `src/include/` 增加/调整接口声明  
3. 如需平台接入，更新 `src/platform/` 对应封装  
4. 在 `CMakeLists.txt` 接入新源文件  
5. 补充回归测试（至少 1 个正向 + 1 个负向预期）

## 7. 代码约束与建议

- 保持 C99 兼容
- 优先确定性实现，减少随机依赖
- 错误码必须可判定，不用隐式失败
- 新增功能必须同步新增测试
- 不提交运行产物和临时文件

## 8. 交付检查清单

- 功能可编译（clang）
- `full_test_suite` 全绿
- 关键日志可追踪（指标齐全）
- 文档同步更新（README/用户手册/开发手册）

