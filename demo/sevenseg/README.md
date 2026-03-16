# SevenSeg Demo

这是“数字到七段数码管引脚”的演示：
- 输入一位数字（0-9）
- 模型输出数码管引脚状态
- 在 CLI 渲染数码管图形效果

## 这个 Demo 在做什么

- 程序化生成 `0` 到 `9` 的训练样本
- 训练后执行推理，把数字映射为七段引脚
- 在终端输出引脚位与数码管图形

## 怎么实现的

核心实现入口是 `demo/sevenseg/main.c`，流程如下：

1. 数据准备  
   - 生成词表 `demo_vocab_sevenseg.txt`
   - 构造训练样本：每个数字拆成两次推理 bank（4+4 引脚）

2. 训练与导出  
   - 调用 `workflow_train_from_memory(...)`
   - 导出 `bin` 与 `c` 权重文件
   - 导出函数网络文件用于静态集成验证

3. 推理与渲染  
   - 输入命令格式：`N`（单个数字）
   - 先推理 bank0 得到 `a,b,c,d`，再推理 bank1 得到 `e,f,g,dp`
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
