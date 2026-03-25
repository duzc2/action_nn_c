# 开发手册

## 1. 项目定位与目标

本工程用于把大模型已掌握的技能固化为小型执行网络，并支持快速训练与部署。

典型应用场景：
- 大模型负责战略与规划（如CS游戏中的战术决策）
- 小网络负责低时延执行（如瞄准、躲避、射击等具体操作）
- demo 仅用于展示该协同方式

项目分为三层：
- 流程层：CMake 开关 + 编译期注册控制可用网络类型
- 生成层：用户程序调用 profiler，生成训练/推理 `.c` 并复制固定 `.h`
- 执行层：训练工程与推理工程按依赖边界分别编译运行

## 2. 关键模块说明

| 模块 | 路径 | 说明 |
|------|------|------|
| profiler | `src/profiler/` | 读取网络规格，生成网络结构 .c 文件 |
| 推理注册 | `src/nn/nn_infer_registry.c` | 编译期注册网络类型，运行时查找并调用 |
| 推理核心 | `src/infer/` | 独立运行的推理库，不依赖训练 |
| 训练核心 | `src/train/` | 训练库，依赖推理库 |
| 网络实现 | `src/nn/types/*/` | 具体网络类型实现（mlp、transformer等） |

## 3. Demo 开发流程（6步）

每个 demo 目录下包含：
- `generate_main.c` - 调用 profiler 生成网络结构代码
- `train_main.c` - 训练网络，生成权重文件
- `infer_main.c` - 加载权重，执行推理

补充约束：
- `generate_main.c` 属于用户侧代码。
- 用户侧若要构造具体网络类型配置，只允许包含 config-only 头文件。
- 例如：
  - `types/mlp/mlp_config.h`
  - `types/transformer/transformer_config.h`
- 不要在用户侧 `generate_main.c` 里直接包含 `*_infer_ops.h` 或 `*_train_ops.h`。

### 3.1 准备工作：清理旧构建（如需要）

```powershell
# 清理单个 demo 的生成文件
cmake --build . --target move_clean

# 或者清理所有 demo 的生成文件
cmake --build . --target clean_all
```

### 3.2 步骤1：编译生成器

```powershell
cd build
cmake .. -G "MinGW Makefiles"
cmake --build . --config Debug --target move_generate
```

说明：
- 编译 `move_generate` 可执行文件
- 该目标依赖 `profiler_core` 库

### 3.3 步骤2：运行生成器

```powershell
./demo/move/Debug/move_generate.exe
```

输出文件到 `build/demo/move/data/`：
- `move.c` - 网络结构实现
- `move.h` - 网络接口头文件
- `network_spec.txt` - 网络规格说明

注意：profiler 只生成网络结构代码，不生成训练数据。

### 3.4 步骤3：编译训练

```powershell
cmake --build . --config Debug --target move_train
```

说明：
- 链接 `move.c` 和推理库
- 训练后生成权重文件

### 3.5 步骤4：运行训练

```powershell
./demo/move/Debug/move_train.exe
```

输出：
- `weights.txt` - 文本格式权重
- `weights.c` - C 代码格式权重

### 3.6 步骤5：编译推理

```powershell
cmake --build . --config Debug --target move_infer
```

说明：
- 链接 `move.c` 和 `weights.c`
- 链接推理库（不是训练库）

### 3.7 步骤6：运行推理

```powershell
./demo/move/Debug/move_infer.exe
```

输入示例：
```
5 5      # 起始坐标 (x, y)
0         # 命令0：向上
0         # 命令0：向上
0         # 命令0：向上
4         # 命令4：停止
```

输出：
```
x=5 y=6
x=5 y=7
x=5 y=8
final_x=5 final_y=8
```

## 4. CMakeLists.txt 结构说明

每个 demo 的 CMakeLists.txt 包含三个目标：

```cmake
# 目标1：生成器 - 第一次编译
add_executable(move_generate generate_main.c)
target_link_libraries(move_generate PRIVATE profiler_core)

# 目标2：训练 - 第二次编译
add_executable(move_train train_main.c ${MOVE_GENERATED_C})
target_link_libraries(move_train PRIVATE nn_infer_core)
add_dependencies(move_train move_generate)  # 确保生成器先运行

# 目标3：推理 - 第三次编译
add_executable(move_infer infer_main.c ${MOVE_GENERATED_C} ${MOVE_WEIGHTS_C})
target_link_libraries(move_infer PRIVATE nn_infer_core infer_core)
add_dependencies(move_infer move_train)
```

关键点：
- `${CMAKE_BINARY_DIR}` 指向 build 目录
- 生成的 .c 文件在 build 目录下，不在源码目录
- 推理库 `infer_core` 是独立的，不包含训练代码

## 5. 新增网络类型流程

### 5.1 创建网络实现目录

```
src/nn/types/新类型名/
├── 新类型名_config.h          # 用户侧与生成代码共享的配置类型
├── nn_type_新类型名_infer.c   # 推理实现注册
├── nn_type_新类型名_train.c   # 训练实现注册
├── 新类型名_infer_ops.c       # 推理操作实现
├── 新类型名_infer_ops.h       # 推理操作头文件
├── 新类型名_train_ops.c       # 训练操作实现（可选）
└── 新类型名_train_ops.h       # 训练操作头文件（可选）
```

### 5.2 实现推理函数

在 `*_infer_ops.c` 中实现推理函数：

```c
#include "新类型名_infer_ops.h"

int nn_新类型名_infer_step(void* context) {
    // 推理实现
    return 0;
}
```

### 5.3 注册网络类型

在 `nn_type_新类型名_infer.c` 中注册：

```c
#include "nn_infer_registry.h"
#include "新类型名_infer_ops.h"

const NNInferRegistryEntry nn_type_新类型名_infer_entry = {
    "新类型名",
    nn_新类型名_infer_step
};
```

### 5.4 更新 CMakeLists.txt

在 `src/nn/CMakeLists.txt` 中添加源文件。

### 5.5 验证流程

1. 重新编译：`cmake --build . --target move_generate`
2. 运行生成器：生成网络代码
3. 编译训练：`cmake --build . --target move_train`
4. 运行训练：生成权重
5. 编译推理：`cmake --build . --target move_infer`
6. 运行推理：验证结果

## 6. 目录结构总览

```
action_c/
├── src/
│   ├── profiler/          # 代码生成器
│   ├── infer/            # 推理库（独立）
│   ├── train/            # 训练库（依赖推理）
│   └── nn/
│       ├── nn_infer_registry.c  # 网络注册
│       └── types/
│           ├── mlp/            # MLP 网络实现
│           └── transformer/    # Transformer 网络实现
├── demo/
│   └── move/
│       ├── generate_main.c     # 调用 profiler
│       ├── train_main.c        # 训练网络
│       ├── infer_main.c        # 推理网络
│       └── CMakeLists.txt      # 三个编译目标
└── docs/
    └── developer_manual.md     # 本文档
```

## 7. 注意事项

1. **绝对路径禁止**：所有路径使用相对路径，如 `demo/move/data`
2. **推理独立**：推理库不能包含任何训练代码
3. **编译时注册**：网络类型在编译时注册，不使用运行时枚举
4. **生成文件位置**：profiler 生成的 .c 文件在 build 目录下，不修改源码
