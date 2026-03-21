# 用户手册

## 1. 使用总览

本系统在用户侧是固定的“三次编译 + 一次生成运行”流程：

- 第一次编译：按 CMakeLists 开关启用网络类型，构建用户程序与 profiler 组件
- 第一次运行：执行用户程序，用户程序调用 profiler 生成训练/推理两套代码
- 第二次编译：单独编译训练工程（依赖推理 `.c` + 训练 `.c`）
- 第三次编译：单独编译推理工程（仅依赖推理 `.c`）

## 2. 第一次编译：启用网络类型

```bash
cmake -S . -B build -DENABLE_NN_TRANSFORMER=ON
cmake --build build --config Debug --target profiler
```

规则：

- 只编译开关启用的网络类型
- 未启用类型不进入注册表，不参与后续流程
- `src/nn/` 必须实现并维护常见网络类型：`rnn`、`cnn`、`knn`、`transformer`、`mlp`、`rbfn`、`autoencoder`、`variational_autoencoder`、`tcn`、`gnn`、`ssm`、`mamba_s4`、`esn`、`siamese_triplet`、`unet_encoder_decoder_skip`、`capsule`、`kan`、`som`、`tree_random_forest_xgboost`、`svm`、`tiny_tcn`
- `sevenseg` 位于 `demo/sevenseg`，属于 demo 示例，不是 `src` 下的 `nn` 网络类型

## 3. 第一次运行：调用 profiler 生成代码

用户程序内部步骤：

- 构建网络结构体
- 以网络结构体作为参数调用 profiler
- profiler 生成训练 `.c` 与推理 `.c`
- profiler 复制固定模板 `.h`（不做动态 `.h` 生成）

示例运行：

```bash
build/Debug/profiler.exe
```

## 4. 第二次编译：训练工程

训练工程依赖关系：

- 依赖推理 `.c`
- 依赖训练 `.c`

构建与运行示例：

```bash
cmake --build build --config Debug --target sevenseg_train
build/demo/sevenseg/Debug/sevenseg_train.exe
```

训练输出：

- 网络权重 `.bin`
- 网络权重 `.c`

## 5. 第三次编译：推理工程

推理工程依赖关系：

- 仅依赖推理 `.c`
- 不依赖训练 `.c`

构建与运行示例：

```bash
cmake --build build --config Debug --target sevenseg_infer_bin
build/demo/sevenseg/Debug/sevenseg_infer_bin.exe
```

推理可选加载方式：

- 加载训练阶段输出的 `.bin`
- 或直接链接权重 `.c`

## 6. 测试

```bash
ctest --test-dir build -C Debug --output-on-failure
```
