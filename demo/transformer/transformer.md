# Transformer Demo

## 功能说明

演示小型对话 transformer 网络的生成、训练与推理流程。

## 流程要求

本 demo 必须按 6 步执行，并且每一步都在独立构建目录中完成：

1. 配置并编译 `generate`
2. 运行 `generate`
3. 配置并编译 `train`
4. 运行 `train`
5. 配置并编译 `infer`
6. 运行 `infer`

可直接使用：

- `run_demo.bat`
- `run_demo.sh`

## 运行时生成物位置

运行时生成的代码、头文件、元数据等，不写回源码目录。

统一生成到相对当前可执行文件的：

```text
../../data/
```

对应实际目录：

```text
build/demo/transformer/data/
```

## 6 步命令

### 1. configure + build generate

```powershell
cmake -S demo/transformer/generate -B build/demo/transformer/generate
cmake --build build/demo/transformer/generate --config Debug
```

### 2. run generate

```powershell
build/demo/transformer/generate/Debug/transformer_generate.exe
```

### 3. configure + build train

```powershell
cmake -S demo/transformer/train -B build/demo/transformer/train
cmake --build build/demo/transformer/train --config Debug
```

### 4. run train

```powershell
build/demo/transformer/train/Debug/transformer_train.exe
```

### 5. configure + build infer

```powershell
cmake -S demo/transformer/infer -B build/demo/transformer/infer
cmake --build build/demo/transformer/infer --config Debug
```

### 6. run infer

```powershell
build/demo/transformer/infer/Debug/transformer_infer.exe
```

## 说明

- `train` 与 `infer` 的依赖代码都从 `build/demo/transformer/data/` 读取
- 静态示例语料可以保留在源码目录，但运行时生成物不能写回 `demo/transformer/`
