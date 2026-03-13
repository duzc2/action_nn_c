# 从需求到部署：实战使用手册

这份文档不是“看源码结构”，而是教你怎么从零把自己的任务跑起来。

目标回答 7 个问题：
1. 怎么设计模型？
2. 怎么配置参数？
3. 怎么在训练前做资源预估？
4. 怎么准备训练数据？
5. 怎么训练并导出可部署权重？
6. 怎么做单步推理？
7. 怎么做“一条命令多步执行”的外部循环推理？

---

## 1) 先判断你该不该用这个项目

适合你：
- 需要 C99 环境可运行的控制模型
- 需要“文本命令 + 状态向量 → 动作向量”
- 需要外部主循环控制（模型只做单帧决策）
- 需要可测试、可导出、可部署的最小闭环

不适合你：
- 追求大规模深度学习训练能力
- 需要自动微分、GPU、分布式训练框架

---

## 2) 模型设计：先定义 I/O 与任务边界

你的任务先写清楚 3 件事：
- 输入命令空间（例如：`move left/right/stop/fast/slow`）
- 状态维度语义（8 维每一维代表什么）
- 输出动作语义（4 维每一维控制什么）

对应配置文件：
- [config_user.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/config_user.h)

核心配置项：
- `VOCAB_SIZE`：词表上限
- `STATE_DIM`：状态输入维度
- `OUTPUT_DIM`：动作输出维度
- `IO_MAPPING_ACTIVATIONS`：每个输出通道使用 Sigmoid 还是 Tanh
- `EMBED_DIM / NUM_LAYERS / NUM_HEADS / FFN_DIM`：网络规模

设计建议：
- 二值动作（开关）用 Sigmoid
- 连续动作（位移/角度）用 Tanh
- 先用小模型跑通闭环，再放大规模

---

## 3) 训练前预估：先跑 profiler 再决定配置

先做静态资源估算，避免“设计后才发现跑不动”。

### 3.1 构建

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang
cmake --build build --target profiler
```

### 3.2 运行

```powershell
.\build\profiler.exe
```

或自定义输出路径：

```powershell
.\build\profiler.exe -o src/include/network_def.h
```

Profiler 会给你：
- 总参数量
- 峰值激活内存
- FLOPs 估算
- 输出通道激活函数映射

实现入口：
- [profiler.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/profiler.c)

---

## 4) 数据设计：训练前先定数据规范

当前训练数据格式（CSV）是：
- `command + state[8] + target[4]`

可参考：
- [c99_full_demo.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c)
- [csv_loader.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/train/csv_loader.c)

你应该自己定义：
- 状态归一化规则（例如坐标除以最大距离）
- 目标动作范围（Sigmoid 输出期望在 0~1，Tanh 在 -1~1）
- 采样覆盖策略（不要只采“必然正确”样本）

最低要求：
- 训练集与评估集分布不能完全相同
- 必须包含负向/异常/边界样本

---

## 5) 训练：你要写的最小主流程

可直接参考现成主流程：
- 最小闭环：[min_train_loop.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/min_train_loop.c)
- 完整闭环：[c99_full_demo.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c)

训练主流程应包含：
1. 初始化词表与 tokenizer
2. 读入 CSV 数据
3. 前向计算 logits
4. 激活映射为动作
5. 计算损失并更新权重
6. 输出训练指标（loss/误差/命中率）

训练后导出：
- 二进制权重：`weights_save_binary`
- C 源权重：`weights_export_c_source`

接口位置：
- [weights_io.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/weights_io.h)

---

## 6) 部署：单步推理怎么做

单步推理本质是 5 步：
1. 输入命令文本
2. tokenizer 编码成 token ids
3. 前向得到 logits
4. 激活映射得到动作向量
5. 下发到驱动层

关键接口：
- 编码：`tokenizer_encode`  
  [tokenizer.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/tokenizer.h#L127-L131)
- 协议（可选）：`protocol_encode_raw / protocol_decode_packet`  
  [protocol.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/protocol.h#L50-L95)
- 驱动下发：`driver_stub_apply`  
  [platform_driver.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/platform_driver.h)

单步推理模板（伪代码）：

```c
tokenizer_encode(&tokenizer, command, ids, cap, &count);
predict_logits(weights, ids, count, state, logits);
activate_output(logits, act);
driver_stub_apply(&driver, act, OUTPUT_DIM);
```

---

## 7) 一条命令多步推理：外部循环怎么做

你提的核心需求是对的：  
**循环必须由外部控制，不在模型内部。**

推荐控制器流程：
1. 外部维护 `current_pose` 与 `goal_pose`
2. 每帧根据误差构造 state
3. 每帧调用一次模型，得到小动作
4. 应用一步动作并更新当前位姿
5. 判断是否到达；未到达则下一帧继续

参考实现：
- [run_external_goal_loop](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/tools/c99_full_demo.c#L398-L514)

停止条件建议：
- `|dx| < eps && |dy| < eps`
- 或超过最大帧数（超时保护）

---

## 8) 如何避免“看起来能跑，实际上不可用”

你至少要做这 5 类验证：
- 泛化测试（未见值）
- OOD 测试（远分布）
- 对抗扰动（状态/token）
- 长时序稳定（多千帧）
- 预期错误路径（空命令、损坏包、超长输入）

当前模型专项测试入口：
- [test_cases_model_special.c](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/test/src/test_cases_model_special.c)

---

## 9) 你的实际落地步骤（建议照抄）

### 第一步：定义你的任务 I/O
- 修改 [config_user.h](file:///c:/Users/ASUS/Desktop/ai-build-ai/action_c/src/include/config_user.h)

### 第二步：跑 profiler 看资源是否可接受
- 运行 `profiler`

### 第三步：准备你自己的 CSV 数据
- 参考 `command + state + target` 格式

### 第四步：训练并导出权重
- 参考 `min_train_loop` / `c99_full_demo`

### 第五步：接入你的外部主循环
- 每帧调用一次推理，外部判断继续/停止

### 第六步：跑模型专项测试
- 确认泛化、鲁棒、负向预期都过线

---

## 10) 常见误区

- 误区 1：只跑 demo，不改数据和目标  
  结果：只能证明 demo 能跑，不能证明你的任务可用。

- 误区 2：模型自己 while 循环  
  结果：控制权丢失，难以做安全策略和抢占。

- 误区 3：没有负向测试  
  结果：线上异常输入会直接暴露脆弱性。

