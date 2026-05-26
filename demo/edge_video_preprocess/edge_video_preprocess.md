# Edge Video Preprocessing Demo

演示边缘端智能预处理管道：帧差运动检测 + BnConvNet CNN识别分类 + 带宽节省统计。

## 概述

在边缘计算场景中，带宽和算力是稀缺资源。本demo展示用BnConvNet在边缘端进行智能预处理：

```
监控视频帧 → 帧差(SAD)检测 + 自适应阈值 + 时序平滑 → 静态? → 跳过（省算力+带宽）
                                                      → 有运动? → CNN推理 → 分类日志 → 统计报告
```

## 架构

### 网络结构（BnConvNet v47，5个叶子节点）

卷积层全部使用 stride=1，通过 3×3 卷积自然缩小特征图尺寸，配合 GAP (Global Average Pooling) 替代全连接前的展平，大幅减少参数量。所有卷积层使用 BN + LeakyReLU(alpha=0.01) 激活。

```
输入: 32×32×3 RGB

Conv1:    3×3,  3→32,   s=1, BN+LeakyReLU(0.01), POOL_NONE → 30×30×32
Conv2:    3×3,  32→64,  s=1, BN+LeakyReLU(0.01), POOL_NONE → 28×28×64
Conv3:    3×3,  64→128, s=1, BN+LeakyReLU(0.01), POOL_NONE → 26×26×128
Conv4:    3×3,  128→256,s=1, BN+LeakyReLU(0.01), POOL_NONE → 24×24×256
GAP:      1×1,  256→256, POOL_AVG → 256 scalars

MLP Head: 256 → [256(LeakyReLU)] → 10, ADAM + Cross-Entropy

总计: 5个叶子节点 (4 CNN + 1 MLP)
```

### 关键技术

| 特性 | 说明 |
|------|------|
| 运动检测 | SAD (Sum of Absolute Differences) + 自适应阈值 + 时序平滑防抖 |
| CNN后端 | stride=1, BN, LeakyReLU(alpha=0.01), He初始化, ADAM |
| 网络架构 | BnConvNet v47: 4CNN + GAP + 1MLP, 通道扩展 3→32→64→128→256 |
| 数据格式 | CIFAR-10平面RGB → 交错RGB, uint8 → float32 [0,1] |
| 训练参数 | batch_size=4, lr=0.001, momentum=0.0, step decay (every 3 epochs) |
| 运行环境 | 纯C11实现, 无外部依赖, CPU运行 |
| 代码生成 | profiler范围合并连接优化 |
| 输出 | 实时分类日志 + 最终统计报告 (带宽节省/分类分布/延迟) |

## 构建与运行

### 前置要求

- Python 3.7+ (数据准备)
- CMake 3.20+, MSVC 2022 (Windows) 或 Clang (Linux)
- Visual Studio 2022 Community (Windows) 或 Ninja (Linux)
- 可选: ffmpeg (用于真实视频帧提取)

### 完整管道

**Windows (MSVC):**
```batch
cd demo\edge_video_preprocess
REM 1. 数据准备
python data_prep.py [--video path/to/video.mp4]

REM 2. 构建生成器
cmake -S generate -B ..\..\build\demo\edge_video_preprocess\generate
cmake --build ..\..\build\demo\edge_video_preprocess\generate --config Release

REM 3. 生成网络代码
..\..\build\demo\edge_video_preprocess\generate\edge_video_preprocess_generate.exe

REM 4. 构建训练器
cmake -S train -B ..\..\build\demo\edge_video_preprocess\train
cmake --build ..\..\build\demo\edge_video_preprocess\train --config Release

REM 5. 训练
..\..\build\demo\edge_video_preprocess\train\edge_video_preprocess_train.exe

REM 6. 构建推理器
cmake -S infer -B ..\..\build\demo\edge_video_preprocess\infer
cmake --build ..\..\build\demo\edge_video_preprocess\infer --config Release

REM 7. 推理
..\..\build\demo\edge_video_preprocess\infer\edge_video_preprocess_infer.exe
```

**Linux (Ninja+Clang):**
```bash
cd demo/edge_video_preprocess
chmod +x run_demo.sh
./run_demo.sh [--video path/to/video.mp4]
```

### 快速验证 (200样本训练)

如果完整训练太慢（50K样本 × 多轮可能需数小时），可用快速模式验证管道：

```bash
# 在项目根目录
cmake -S verify -B build/verify
cmake --build build/verify --config Release --target quick_train
build/verify/Release/quick_train.exe          # 200样本 × 5轮 → weights.bin

# 然后运行推理
build/demo/edge_video_preprocess/infer/edge_video_preprocess_infer.exe
```

### 单元测试

```bash
cmake --build build/verify --config Release --target test_stride
cmake --build build/verify --config Release --target test_network_e2e
cmake --build build/verify --config Release --target test_motion_detect
cmake --build build/verify --config Release --target test_cnn_full
cmake --build build/verify --config Release --target test_cnn_backend
cmake --build build/verify --config Release --target test_bn_consistency

build/verify/Release/test_stride.exe          # stride功能测试
build/verify/Release/test_network_e2e.exe     # 端到端网络测试
build/verify/Release/test_motion_detect.exe   # 运动检测单元测试
build/verify/Release/test_cnn_full.exe        # CNN综合测试 (24项)
build/verify/Release/test_cnn_backend.exe     # CNN后端训练测试
build/verify/Release/test_bn_consistency.exe  # BN前向/反向一致性测试
```

## 实际运行结果

> **注:** 以下结果来自 BnConvNet v47 (4CNN+GAP+MLP) 架构。完整 50K 训练需要数小时 CPU 时间。Batch Normalization 修复后，训练稳定性显著提升。

### 单元测试 (stride验证)

```
=== CNN Stride Support Verification ===

Test 1: Forward pass stride=1  = OK
Test 2: Forward pass stride=2  = OK
Test 3: Reject stride=0        = OK
Test 4: Weight save/load       = OK
Test 5: Training step stride=2 = OK

Results: 0 failures
```

### 运动检测测试

```
=== Motion Detection Test ===

Test 1: Adaptive threshold initialization = OK
Test 2: Adaptive threshold update         = OK
Test 3: Temporal smoother state machine   = OK
Test 4: Edge case (all zero frames)       = OK

Results: 0 failures
```

### 推理管道 (7200帧监控视频)

```
=== Edge Video Preprocessing — Inference Pipeline ===

Video: 7200 frames @ 2.0 fps (3600 sec equivalent)
Resolution: 32x32x3 (RGB) | BnConvNet v47 (4 CNN + GAP + 1 MLP) → CIFAR-10

--- Motion Detection ---
Frames with motion:    5015 ( 69.7%)  <- 运行了CNN推理
Frames skipped:        2185 ( 30.3%)  <- 节省了算力+带宽

--- Inference Performance ---
Avg latency per inference: ~36 ms
Total inference time:       ~3 min

--- Bandwidth Savings ---
Raw upload (all frames):     84.38 MB
Edge filtered (motion only): 58.77 MB
Bandwidth saved:             25.61 MB ( 30.3%)
```

## 性能说明

| 指标 | 值 | 说明 |
|------|-----|------|
| 推理延迟 | ~36ms/帧 | 5子网CPU前向传播 |
| 静态帧跳过率 | ~30% | 取决于视频内容, 自适应SAD阈值 |
| 模型大小 | ~2.3MB | 序列化权重文件 |
| 子网数量 | 5 | 4 CNN + 1 MLP |
| 参数量 | ~600K | He (Kaiming) 均匀初始化 |
| 单步训练 | ~0.1s | 5子网前向+反向+ADAM |
| 快速训练 | ~3分钟 | 200样本 × 5轮 |
| 完整训练 | 数小时 | 50K样本 × 多轮 (CPU) |

## 数据来源

- **CIFAR-10**: Krizhevsky, 2009 (学术公开数据集)
- **监控视频**: MEVA数据集 (CC-BY-4.0, 328+小时真实CCTV), 也可使用本地视频文件
- **无视频时**: 自动生成合成帧用于管道测试

## 验证标准

1. CNN功能验证: 24项单元测试全部通过
2. BN一致性: 前向/反向缓存一致性测试通过
3. 端到端训练: loss持续下降
4. 运动检测: 自适应阈值和时序平滑功能正常
5. 生成+构建: generate/train/infer 三步构建成功 (WX strict warnings)
6. 推理管道: 7200帧跑完不崩溃, 运动检测正常工作
7. 权重持久化: save/load round-trip 输出一致
8. 带宽节省: 静态帧正确跳过, 统计数据合理
9. CIFAR-10准确率 > 80% (需完整50K样本 × 多轮训练, BN+LeakyReLU架构)

## 文件结构

```
demo/edge_video_preprocess/
├── edge_video_preprocess.md  # 本文档
├── data_prep.py              # Python数据准备 (CIFAR-10+视频帧)
├── cifar10_dataset.h/c       # CIFAR-10二进制加载器 + 数据增强
├── video_processor.h/c       # 视频帧读取+自适应运动检测+时序平滑
├── generate_main.c           # 网络架构定义 (BnConvNet v47)
├── train_main.c              # CIFAR-10训练循环 (LR调度+数据增强+恢复)
├── infer_main.c              # 监控视频推理+统计报告
├── generate/CMakeLists.txt
├── train/CMakeLists.txt
├── infer/CMakeLists.txt
├── run_demo.sh               # Linux构建运行脚本
├── run_demo.bat              # Windows构建运行脚本
├── dataset/                  # CIFAR-10数据 (下载后)
└── video_frames/             # 视频帧数据
    ├── video_meta.txt
    └── video_frames.dat

src/nn/types/cnn/             # CNN后端
├── cnn_config.h              # 配置结构
├── cnn_infer_ops.c/h         # 推理算子 (He uniform init, LeakyReLU)
└── cnn_train_ops.c/h         # 训练算子 (ADAM, BN fix)

verify/                       # 验证测试
├── CMakeLists.txt
├── test_stride.c             # stride单元测试
├── test_network_e2e.c        # 端到端网络测试
├── test_cnn_full.c           # CNN综合测试 (24项)
├── test_cnn_backend.c        # CNN后端训练测试
├── test_bn_consistency.c     # BN一致性测试
├── test_motion_detect.c      # 运动检测单元测试
└── quick_train.c             # 快速训练工具
```
