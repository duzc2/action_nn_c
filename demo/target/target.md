# Target Demo

## 功能说明

演示二维目标点移动控制网络，并在推理阶段输出字符地图，直观看到：

- 50x50 地图
- 起点 `S`
- 当前点 `C`
- 目标点 `T`
- 预测下一步 `P`
- 历史轨迹 `o`

默认推理演示为自动测试模式：

- 地图大小固定为 `50x50`
- 初始位置固定为 `(25, 25)`
- 最大速度固定为 `2`
- 不再写死固定轮数
- 终止条件为：**成功到达 `3` 次目标**
- 为防止异常长时间运行，额外设置安全上限：**最多 `200` 步**
- 每一轮都会把预测结果同步为新的当前位置
- 到达目标后自动随机一个新的目标点继续演示
- 每次运行都会重新生成随机种子，因此不同次运行的目标序列通常不同
- 推理开始时会打印本次运行使用的随机种子，方便复盘

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
build/demo/target/data/
```

## 6 步命令

### 1. configure + build generate

```powershell
cmake -S demo/target/generate -B build/demo/target/generate
cmake --build build/demo/target/generate --config Debug
```

### 2. run generate

```powershell
build/demo/target/generate/Debug/target_generate.exe
```

### 3. configure + build train

```powershell
cmake -S demo/target/train -B build/demo/target/train
cmake --build build/demo/target/train --config Debug
```

### 4. run train

```powershell
build/demo/target/train/Debug/target_train.exe
```

### 5. configure + build infer

```powershell
cmake -S demo/target/infer -B build/demo/target/infer
cmake --build build/demo/target/infer --config Debug
```

### 6. run infer

```powershell
build/demo/target/infer/Debug/target_infer.exe
```

## 说明

- `train` 与 `infer` 的依赖代码都从 `build/demo/target/data/` 读取
- 不应回到 `demo/target/` 源码目录寻找运行时生成物
- `infer` 默认不再等待手工输入，而是自动执行“达到 3 次目标或到达 200 步上限”为止的可视化地图推理
