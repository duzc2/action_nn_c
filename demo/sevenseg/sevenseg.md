# SevenSeg Demo

## 功能说明

演示字符到七段数码管引脚控制的映射网络。

大模型学习数码管引脚配置后，生成小的映射网络，将数字/字符转换为引脚高低电平。

## 数据准备

训练数据由用户自行准备，存放在 `demo/sevenseg/data/` 目录下。

训练数据格式：
- 每行：输入数字/字符 + 7个引脚状态
- 例如：`0 1 1 1 1 1 1 0` 表示数字0对应的七段状态

## 6步使用流程

### 步骤1：编译生成器

```powershell
cmake --build . --target sevenseg_generate
```

### 步骤2：运行生成器

```powershell
./demo/sevenseg/Debug/sevenseg_generate.exe
```

### 步骤3：编译训练

```powershell
cmake --build . --target sevenseg_train
```

### 步骤4：运行训练

```powershell
./demo/sevenseg/Debug/sevenseg_train.exe
```

### 步骤5：编译推理

```powershell
cmake --build . --target sevenseg_infer
```

### 步骤6：运行推理

```powershell
./demo/sevenseg/Debug/sevenseg_infer.exe
```

## 输入输出格式

### 输入格式

数字 (0-9) 或字符 (A-F)

### 输出格式

七段数码管各段的状态

## 网络规格

- **网络类型**: mlp
- **输入**: 数字/字符编码
- **输出**: 7个引脚状态

## 文件说明

| 文件 | 说明 |
|------|------|
| `generate_main.c` | 调用 profiler 生成网络结构代码 |
| `train_main.c` | 训练网络，生成权重文件 |
| `infer_main.c` | 加载权重，执行推理 |
| `CMakeLists.txt` | 定义三个编译目标 |
