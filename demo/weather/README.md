# Weather Demo

## 功能说明

演示使用 profiler 生成的 MLP 代码，基于历史气象数据训练三座城市（北京、上海、纽约）的极小模型，预测次日天气特征。

每座城市独立训练一个 MLP 模型（~268 个参数），输入过去 7 天的 4 项气象指标，输出第 8 天的预测值。模型架构相同但权重独立，分别学习各城市的气候规律。

## 数据来源

数据来自 [Open-Meteo](https://open-meteo.com/) 免费 API，底层使用 ECMWF ERA5 全球再分析数据集（气象学界权威数据源之一）。

数据覆盖 2015-01-01 至 2024-12-31，每日 4 项指标：

| 字段 | 含义 | 单位 |
|------|------|------|
| temp_max | 日最高温度 | °C |
| temp_min | 日最低温度 | °C |
| precipitation | 日总降水量 | mm |
| wind_speed | 日最大风速 | m/s |

每座城市约 3,653 条记录，CSV 格式，已提交到 `dataset/` 目录。

## 数据集说明

本 demo 依赖提交到源码目录的 CSV 文件：

- `demo/weather/dataset/beijing.csv`
- `demo/weather/dataset/shanghai.csv`
- `demo/weather/dataset/new_york.csv`

如需重新获取最新数据：

```bash
python demo/weather/fetch_data.py
```

## 模型架构

| 项目 | 配置 |
|------|------|
| 网络类型 | MLP |
| 输入层 | 28 个神经元（7 天 × 4 特征） |
| 隐藏层 | 1 层 × 8 个神经元 |
| 输出层 | 4 个神经元（次日 4 特征） |
| 隐藏激活 | ReLU |
| 输出激活 | NONE（线性，回归任务） |
| 损失函数 | MSE |
| 优化器 | Adam（lr=0.001） |
| 参数量 | ~268（每城市） |

### 滑动窗口

```
[天1] [天2] [天3] [天4] [天5] [天6] [天7] → 预测 → [天8]
└────────── 28 个归一化特征 ──────────┘         └ 4 个目标值 ┘
```

### 训练/测试分割

- **训练集**：前 80%（约 2,916 个窗口）
- **测试集**：后 20%（约 730 个窗口）
- 不随机打乱，保持时序因果性（避免未来信息泄露）
- 每城市训练 50 个 epoch

## 流程要求

本 demo 必须按顺序执行，每一步在独立构建目录中完成：

1. 配置并编译 `generate`
2. 运行 `generate`
3. 配置并编译 `train`
4. 运行 `train`
5. 配置并编译 `infer`
6. 运行 `infer`

可使用统一脚本一键运行：

```bash
# 完整流程
bash scripts/run_demo.sh weather

# 或分阶段
bash scripts/run_demo.sh weather generate
bash scripts/run_demo.sh weather train
bash scripts/run_demo.sh weather infer
```

Windows 下也可使用本地脚本：

```cmd
demo\weather\run_demo.bat
```

## 运行时生成物位置

运行时生成的代码、头文件、权重等，不写回源码目录。

统一生成到：

```text
build/demo/weather/data/
```

权重文件按城市命名：

- `weights_beijing.bin`
- `weights_shanghai.bin`
- `weights_new_york.bin`

## 预期输出

### Train 阶段

训练过程会打印每 epoch 的 MSE 损失，损失应持续下降：

```
=== Training: Beijing ===
Loaded 3653 weather records
Train samples: 2916, Test samples: 730
  Epoch 1/50 - avg loss: 0.031373
  Epoch 2/50 - avg loss: 0.013560
  ...
  Epoch 50/50 - avg loss: 0.005672
Weights saved: ../data/weights_beijing.bin
```

### Infer 阶段

推理阶段会打印前 5 条预测对比和整体评估指标：

```
=== Inference: Beijing ===
Test samples: 730
Sample 0:
  TempMax: pred=3.12 actual=2.60
  TempMin: pred=-6.19 actual=-6.00
  Precip: pred=-0.01 actual=0.00
  WindSpd: pred=17.67 actual=17.30
...

Metrics on 730 test samples:
  MAE:  5.2714
  RMSE: 7.5995
```

## 文件结构

```
demo/weather/
├── fetch_data.py          # 数据获取脚本（Open-Meteo API）
├── dataset/               # 已提交的静态 CSV 数据
│   ├── beijing.csv
│   ├── shanghai.csv
│   └── new_york.csv
├── weather_dataset.h      # 数据加载 API 声明
├── weather_dataset.c      # CSV 解析 + 滑窗构建 + 归一化
├── generate_main.c        # 网络定义（MLP 28→8→4）
├── train_main.c           # 训练入口（三城循环）
├── infer_main.c           # 推理入口（加载权重 + 评估指标）
├── generate/CMakeLists.txt
├── train/CMakeLists.txt
├── infer/CMakeLists.txt
├── run_demo.sh            # 本地 Shell 运行脚本
└── run_demo.bat           # 本地 Windows 批处理
```
