# SevenSeg Demo

## 功能说明

演示数字到七段数码管段位输出的映射网络。

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
build/demo/sevenseg/data/
```

## 6 步命令

### 1. configure + build generate

```powershell
cmake -S demo/sevenseg/generate -B build/demo/sevenseg/generate
cmake --build build/demo/sevenseg/generate --config Debug
```

### 2. run generate

```powershell
build/demo/sevenseg/generate/Debug/sevenseg_generate.exe
```

### 3. configure + build train

```powershell
cmake -S demo/sevenseg/train -B build/demo/sevenseg/train
cmake --build build/demo/sevenseg/train --config Debug
```

### 4. run train

```powershell
build/demo/sevenseg/train/Debug/sevenseg_train.exe
```

### 5. configure + build infer

```powershell
cmake -S demo/sevenseg/infer -B build/demo/sevenseg/infer
cmake --build build/demo/sevenseg/infer --config Debug
```

### 6. run infer

```powershell
build/demo/sevenseg/infer/Debug/sevenseg_infer.exe
```

## 说明

- `train` 与 `infer` 的依赖代码都从 `build/demo/sevenseg/data/` 读取
- 不应回到 `demo/sevenseg/` 源码目录寻找运行时生成物
