# MNIST CNN Demo

## 功能说明

演示一个独立于原 `mnist` demo 的 CNN 版本：先把 28x28 手写数字切成四个 14x14 象限，用 CNN 编码，再由 MLP 头完成 10 类分类。

## 数据集说明

本 demo 使用复制到源码目录中的原始 MNIST IDX 文件：

- `demo/mnist_cnn/dataset/train-images.idx3-ubyte`
- `demo/mnist_cnn/dataset/train-labels.idx1-ubyte`
- `demo/mnist_cnn/dataset/t10k-images.idx3-ubyte`
- `demo/mnist_cnn/dataset/t10k-labels.idx1-ubyte`

## 流程要求

本 demo 必须按 6 步执行，并且每一步都在独立构建目录中完成：

1. 配置并编译 `generate`
2. 运行 `generate`
3. 配置并编译 `train`
4. 运行 `train`
5. 配置并编译 `infer`
6. 运行 `infer`

## 运行时生成物位置

运行时生成的代码、头文件、权重等，不写回源码目录。

统一生成到相对当前可执行文件的：

```text
../data/
```

对应实际目录：

```text
build/demo/mnist_cnn/data/
```

## Windows clang 构建说明

在这台 Win11 机器上，需要先注入 VS x64 构建环境，再使用 clang + Ninja：

```powershell
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake -S demo/mnist_cnn/generate -B build/demo/mnist_cnn/generate -G Ninja -DCMAKE_C_COMPILER=clang"
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat\" && cmake --build build/demo/mnist_cnn/generate"
```

`train` 与 `infer` 两个阶段同理，分别替换对应的 `-S/-B` 路径。

## 说明

- 这是一个新 demo，不修改原有 `demo/mnist/`
- 输入会先被重排成 4 个象限，作为 CNN 的 4 帧序列输入
- `train` 与 `infer` 的依赖代码都从 `build/demo/mnist_cnn/data/` 读取
