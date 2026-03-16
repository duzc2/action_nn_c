# SevenSeg Demo

这是“数字到七段数码管引脚”的演示（走项目训练/推理工作流）：
- 输入一位数字（0-9）
- 模型输出数码管引脚状态
- 在 CLI 渲染数码管图形效果

## 这个 Demo 在做什么

- 使用项目的 `workflow_train_from_memory` 与 `workflow_run_step`
- 采用 `STATE_DIM=1`、`OUTPUT_DIM=7` 的 sevenseg 专用构建配置
- 在终端输出引脚位与数码管图形

## 怎么实现的

核心实现入口是 `demo/sevenseg/main.c`，流程如下：

1. 数据准备  
   - 固定真值表 `kSevenSegTruth[10][7]`
   - 输入为归一化数字 `digit/9`，输出为 7 段开关位

2. 训练  
   - 输入状态：归一化数字 `digit / 9`
   - 输出向量：`a,b,c,d,e,f,g` 七个二值引脚
   - 训练完成后做 0~9 全量自检

3. 推理与渲染  
   - 输入命令格式：`N`（单个数字）
   - 一次推理直接得到 `a,b,c,d,e,f,g`
   - 在 CLI 显示引脚值与七段图形

## 入口与构建

- 代码入口：`demo/sevenseg/main.c`
- 本地目标名：`sevenseg_demo`

在仓库根目录执行：

```bash
cmake -S . -B build
cmake --build build --target sevenseg_demo
```

## 运行

Windows:

```bash
.\build\demo\sevenseg\Debug\sevenseg_demo.exe
```

交互：
- 输入 `0-9` 进行推理并渲染
- 输入 `q` 退出

## 运行产物

运行时会写入到 `demo/sevenseg/data/`：
- `demo/sevenseg/data/demo_vocab_sevenseg.txt`
- `demo/sevenseg/data/demo_weights_sevenseg.bin`
- `demo/sevenseg/data/demo_weights_sevenseg_export.c`
- `demo/sevenseg/data/demo_network_sevenseg_functions.c`
