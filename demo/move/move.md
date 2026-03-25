# Move Demo

## 功能说明

演示连续指令驱动二维坐标移动的网络。

用户输入起始坐标和移动命令，网络输出执行后的坐标。

## 6步使用流程

### 步骤0：清理（如需要）

```powershell
cmake --build . --target move_clean
```

### 步骤1：编译生成器

```powershell
cmake --build . --target move_generate
```

### 步骤2：运行生成器

```powershell
./demo/move/Debug/move_generate.exe
```

生成文件到 `build/demo/move/data/`：
- `move.c` - 网络结构实现
- `move.h` - 网络接口头文件
- `network_spec.txt` - 网络规格

### 步骤3：编译训练

```powershell
cmake --build . --target move_train
```

### 步骤4：运行训练

```powershell
./demo/move/Debug/move_train.exe
```

生成权重文件：
- `weights.txt` - 文本格式权重
- `weights.c` - C 代码格式权重

### 步骤5：编译推理

```powershell
cmake --build . --target move_infer
```

### 步骤6：运行推理

```powershell
./demo/move/Debug/move_infer.exe
```

## 输入输出格式

### 输入格式

第一行：起始坐标
```
x y
```

后续行：移动命令
- `0` = 向上 (y+1)
- `1` = 向下 (y-1)
- `2` = 向左 (x-1)
- `3` = 向右 (x+1)
- `4` = 停止

### 输出格式

每行一个坐标
```
x=5 y=6
x=5 y=7
x=5 y=8
final_x=5 final_y=8
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `generate_main.c` | 调用 profiler 生成网络结构代码 |
| `train_main.c` | 训练网络，生成权重文件 |
| `infer_main.c` | 加载权重，执行推理 |
| `CMakeLists.txt` | 定义三个编译目标 |

## 网络规格

- **网络类型**: mlp (多层感知机)
- **输入**: x, y, one-hot command[5] (7个特征)
- **输出**: out_x, out_y (2个整数)
