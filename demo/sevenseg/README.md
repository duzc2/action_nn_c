# SevenSeg Demo

这是“数字到七段数码管引脚”的拆分演示（同一个 CMakeLists 产出多个可执行文件）：
- 1 个独立训练程序
- 3 个独立推理程序（bin / C数组 / C函数）
- 1 个独立性能测试程序

## 可执行文件

- `sevenseg_train`：独立训练并导出 `bin`、导出 C 数组、导出 C 函数网络
- `sevenseg_infer_bin`：从 `bin` 加载网络后推理
- `sevenseg_infer_c_array`：从导出的 C 数组符号加载权重后推理
- `sevenseg_infer_c_func`：直接调用导出的 C 语言前向函数推理
- `sevenseg_benchmark`：自动压测三种推理方式并输出 Markdown 报表

## 怎么实现的

核心实现拆分为以下文件：
- `train_main.c`
- `infer_bin_main.c`
- `infer_c_array_main.c`
- `infer_c_func_main.c`
- `sevenseg_shared.c/.h`

1. 数据准备  
   - 固定真值表 `kSevenSegTruth[10][7]`
   - 输入状态为归一化数字 `digit/9`
   - 输出向量为 `a,b,c,d,e,f,g`

2. 训练导出  
   - 使用 `workflow_train_from_memory(...)`
   - 导出 `demo_weights_sevenseg.bin`
   - 导出 `demo_weights_sevenseg_export.c`
   - 导出 `demo_network_sevenseg_functions.c`

3. 推理与渲染  
   - 三种推理方式都输入单个数字 `N`
   - 最终得到 `a,b,c,d,e,f,g` 并渲染 CLI 数码管

## 入口与构建

- 代码目录：`demo/sevenseg/`
- 目标名：`sevenseg_train`、`sevenseg_infer_bin`、`sevenseg_infer_c_array`、`sevenseg_infer_c_func`

在仓库根目录执行：

```bash
cmake -S . -B build
cmake --build build --target sevenseg_train sevenseg_infer_bin sevenseg_infer_c_array sevenseg_infer_c_func sevenseg_benchmark
```

## 运行

Windows:

```bash
.\build\demo\sevenseg\Debug\sevenseg_train.exe
.\build\demo\sevenseg\Debug\sevenseg_infer_bin.exe
.\build\demo\sevenseg\Debug\sevenseg_infer_c_array.exe
.\build\demo\sevenseg\Debug\sevenseg_infer_c_func.exe
.\build\demo\sevenseg\Debug\sevenseg_benchmark.exe
```

交互（3个推理程序一致）：
- 输入 `0-9` 进行推理并渲染
- 输入 `q` 退出

性能测试：
- `sevenseg_benchmark` 自动进行大量循环推理测试
- 终端打印 Markdown 表格摘要
- 同时输出 `demo/sevenseg/data/benchmark_report.md`

## 运行产物

运行时会写入到 `demo/sevenseg/data/`：
- `demo/sevenseg/data/demo_vocab_sevenseg.txt`
- `demo/sevenseg/data/demo_weights_sevenseg.bin`
- `demo/sevenseg/data/demo_weights_sevenseg_export.c`
- `demo/sevenseg/data/demo_network_sevenseg_functions.c`
