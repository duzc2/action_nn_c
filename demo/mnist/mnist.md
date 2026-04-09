# MNIST Demo

## 功能说明

演示使用 profiler 生成的 MLP 代码，对 MNIST 手写数字数据集进行训练和识别。

## 数据集说明

本 demo 依赖复制到源码目录下的原始 IDX 文件：

- `demo/mnist/dataset/train-images.idx3-ubyte`
- `demo/mnist/dataset/train-labels.idx1-ubyte`
- `demo/mnist/dataset/t10k-images.idx3-ubyte`
- `demo/mnist/dataset/t10k-labels.idx1-ubyte`

这些文件来自你提供的本机数据目录副本，训练和推理都会直接读取这份 demo 内部副本。

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

运行时生成的代码、头文件、权重等，不写回源码目录。

统一生成到相对当前可执行文件的：

```text
../data/
```

对应实际目录：

```text
build/demo/mnist/data/
```

## Windows clang 构建说明

在这台 Win11 机器上，需要先注入 VS x64 构建环境，再使用 clang + Ninja：

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake -S demo/mnist/generate -B build/demo/mnist/generate -G Ninja -DCMAKE_C_COMPILER=clang"
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build build/demo/mnist/generate"
```

`train` 与 `infer` 两个阶段同理，分别替换对应的 `-S/-B` 路径。

## 6 步命令

### 1. configure + build generate

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake -S demo/mnist/generate -B build/demo/mnist/generate -G Ninja -DCMAKE_C_COMPILER=clang"
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build build/demo/mnist/generate"
```

### 2. run generate

```powershell
build/demo/mnist/generate/mnist_generate.exe
```

### 3. configure + build train

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake -S demo/mnist/train -B build/demo/mnist/train -G Ninja -DCMAKE_C_COMPILER=clang"
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build build/demo/mnist/train"
```

### 4. run train

```powershell
build/demo/mnist/train/mnist_train.exe
```

### 5. configure + build infer

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake -S demo/mnist/infer -B build/demo/mnist/infer -G Ninja -DCMAKE_C_COMPILER=clang"
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build build/demo/mnist/infer"
```

### 6. run infer

```powershell
build/demo/mnist/infer/mnist_infer.exe
```

## 说明

- `train` 与 `infer` 的依赖代码都从 `build/demo/mnist/data/` 读取
- 静态 MNIST 数据集副本保留在 `demo/mnist/dataset/`
- 运行时不会把生成物写回 `demo/mnist/` 源码目录
