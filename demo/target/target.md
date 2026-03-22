# Target Demo

## 功能说明

演示二维目标点移动控制的网络。

根据当前位置和目标位置，计算移动方向和距离。

## 6步使用流程

### 步骤1：编译生成器

```powershell
cmake --build . --target target_generate
```

### 步骤2：运行生成器

```powershell
./demo/target/Debug/target_generate.exe
```

### 步骤3：编译训练

```powershell
cmake --build . --target target_train
```

### 步骤4：运行训练

```powershell
./demo/target/Debug/target_train.exe
```

### 步骤5：编译推理

```powershell
cmake --build . --target target_infer
```

### 步骤6：运行推理

```powershell
./demo/target/Debug/target_infer.exe
```

## 输入输出格式

### 输入格式

目标点坐标 (tx, ty)

### 输出格式

移动方向和距离

## 网络规格

- **网络类型**: mlp
- **输入**: 当前位置 + 目标位置
- **输出**: 移动向量

## 文件说明

| 文件 | 说明 |
|------|------|
| `generate_main.c` | 调用 profiler 生成网络结构代码 |
| `train_main.c` | 训练网络，生成权重文件 |
| `infer_main.c` | 加载权重，执行推理 |
| `CMakeLists.txt` | 定义三个编译目标 |
