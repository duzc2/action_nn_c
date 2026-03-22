# Transformer Demo

## 功能说明

演示小参数量自然语言对话流程的网络。

使用 Transformer 架构实现简单的对话功能，适用于低资源场景下的语言模型部署。

## 6步使用流程

### 步骤1：编译生成器

```powershell
cmake --build . --target transformer_generate
```

### 步骤2：运行生成器

```powershell
./demo/transformer/Debug/transformer_generate.exe
```

### 步骤3：编译训练

```powershell
cmake --build . --target transformer_train
```

### 步骤4：运行训练

```powershell
./demo/transformer/Debug/transformer_train.exe
```

### 步骤5：编译推理

```powershell
cmake --build . --target transformer_infer
```

### 步骤6：运行推理

```powershell
./demo/transformer/Debug/transformer_infer.exe
```

## 输入输出格式

### 输入格式

英文句子或单词

### 输出格式

对话回复

## 网络规格

- **网络类型**: transformer
- **输入**: 文本序列
- **输出**: 文本序列

## 文件说明

| 文件 | 说明 |
|------|------|
| `generate_main.c` | 调用 profiler 生成网络结构代码 |
| `train_main.c` | 训练网络，生成权重文件 |
| `infer_main.c` | 加载权重，执行推理 |
| `CMakeLists.txt` | 定义三个编译目标 |
| `data/corpus.txt` | 训练语料 |
| `data/dialogue_pairs.txt` | 对话数据 |
