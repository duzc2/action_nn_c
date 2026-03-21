# 开发手册

## 1. 架构总览

项目分为三层：

- 流程层：CMake 开关 + 编译期注册控制可用网络类型
- 生成层：用户程序调用 profiler，生成训练/推理 `.c` 并复制固定 `.h`
- 执行层：训练工程与推理工程按依赖边界分别编译运行

## 2. 关键模块

- `src/tools/profiler.c`  
  读取用户传入网络结构体，生成训练/推理 `.c`，复制固定 `.h`
- `src/core/nn_registry*.c`  
  编译期注册启用网络类型，不启用类型不参与后续流程
- `src/train/workflow_train.c`  
  训练与权重导出
- `src/infer/workflow_runtime.c`  
  二进制权重加载与在线推理

## 3. 统一调用规范

- 训练工程依赖“推理 `.c` + 训练 `.c`”
- 推理工程仅依赖推理 `.c`
- 权重入口统一支持 `.bin` 加载或权重 `.c` 直接链接

## 4. 构建与测试

### 4.1 第一次编译（启用类型并构建生成阶段组件）

```powershell
cmake -S . -B build -DENABLE_NN_TRANSFORMER=ON
cmake --build build --config Debug --target profiler
```

说明：

- `sevenseg` 是 `demo/sevenseg` 下的示例工程，不在 `src/` 中
- `sevenseg` 不是新的 `nn` 网络类型，只是用于演示训练/推理流程的 demo 目标

### 4.2 第一次运行（用户程序调用 profiler 生成代码）

```powershell
build\Debug\profiler.exe
```

约束：

- profiler 生成训练 `.c` 与推理 `.c`
- `.h` 来自固定模板复制，不做动态头文件生成

### 4.3 第二次编译（训练工程）

```powershell
cmake --build build --config Debug --target sevenseg_train
```

训练工程依赖推理 `.c` 与训练 `.c`，训练后生成 `.bin` 与权重 `.c`。

### 4.4 第三次编译（推理工程）

```powershell
cmake --build build --config Debug --target sevenseg_infer_bin
```

推理工程仅依赖推理 `.c`，运行时可加载 `.bin` 或使用权重 `.c`。

### 4.5 验证

```powershell
ctest --test-dir build -C Debug --output-on-failure
```


## 6. 新增网络类型的开发流程

1. 新增 `src/nn/[网络类型]/` 并提供训练/推理实现
2. 保证 `src/nn/` 至少包含 `rnn`、`cnn`、`knn`、`transformer` 四类常见网络实现
3. 更新注册配置文件或注册宏清单
4. 在 CMakeLists 增加该类型开关
5. 开启开关后执行三次编译流程验证
6. 补充 demo 与测试（如 `demo/sevenseg`）

补充说明：

- `sevenseg` 属于 demo 示例，不属于本节“新增节点类型（nn 类型）”范畴

## 7. 禁止事项

- 不新增旧兼容接口
- 不绕过 CMake 开关直接接入网络类型
- 不在文档和代码中保留过时构建入口说明
