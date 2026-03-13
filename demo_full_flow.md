# C99 完整 Demo 运行说明

## 目标
演示从训练数据准备到训练、导出模型、加载模型并推理的完整闭环。

## 可执行程序
- `c99_full_demo`

## 执行步骤
1. 使用 clang 生成构建目录：
   - `cmake -S . -B build-demo -G Ninja -DCMAKE_C_COMPILER=clang`
2. 编译 Demo：
   - `cmake --build build-demo --target c99_full_demo`
3. 运行 Demo：
   - `.\build-demo\c99_full_demo.exe`

## 运行后会生成
- `demo_train_data.csv`：训练数据（程序自动生成）
- `demo_weights.bin`：二进制权重
- `demo_weights_export.c`：可直接编译链接的 C 源码权重

## 预期输出要点
- 训练轮次 loss 逐步下降
- 输出权重导出成功信息
- 输出推理结果 actuator 向量
- 输出外部帧循环日志：从 `(0,0)` 朝 `(15,15)` 逐帧移动，每帧只执行一次动作

## 外部循环说明
- 循环由外部驱动，不在模型内部维护循环。
- 每一帧执行顺序：
  1. 外部根据当前位置与目标位置构造状态；
  2. 调用模型推理得到本帧动作；
  3. 执行一帧位姿更新；
  4. 进入下一帧，直到达到目标。
