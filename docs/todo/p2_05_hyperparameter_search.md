# P2-05: 轻量超参数搜索器

> 阶段: Phase 2 | 工作量: 中 | 前置依赖: 无 (纯 Python 工具, 调用 profiler + train 二进制, 可立即启动)

---

## 1. 是什么

一个 Python 脚本工具，在用户定义的搜索空间内自动寻找最优超参数组合（网络结构 + 训练参数），替代手动反复试错。不是完整 NAS（不搜索网络类型、不发明新结构），仅在指定网络类型内搜索超参数。

```
工作流:
  hyper_search.py <demo_name> [--trials 50]
    → 读取搜索空间配置 (.yaml)
    → 循环:
        ① 选下一组超参数 → ② 调 profiler 生成代码
        → ③ 调 train 训练 → ④ 调 eval 评估得分
        → ⑤ 更新 surrogate model
    → 输出最优配置
```

## 2. 为什么需要

- 当前每个 demo 手动试网络结构（"8→6→4 不好，改成 8→12→4 试试"），反复手工劳动
- snake 的 82 参数 MLP 试 20 种配置仅需几分钟，但不自动化就是纯体力活
- 贝叶斯优化用 20-50 次试验找到近最优配置，远优于随机搜索
- Hyperband 策略自动 early-stop 差配置，节省算力
- 非完整 NAS — 不搜索新网络类型或新算子，只在现有结构内优化超参数

## 3. 详细子任务

### 3.1 搜索算法

- [ ] **贝叶斯优化 (推荐)**
  ```python
  # 使用 scikit-learn 的 GaussianProcessRegressor 作为 surrogate model
  # 从 num_trials 次试验中找最优配置

  from sklearn.gaussian_process import GaussianProcessRegressor

  # Surrogate model: GP(observed_configs, observed_scores)
  # Acquisition function: Expected Improvement (EI)
  # 每轮: fit GP → maximize EI → 得到下一个候选配置
  ```

- [ ] **Hyperband 变体 (备选)**
  ```python
  # 多个早停层级, 快速淘汰差配置
  # 更适合试验成本低的场景 (小 demo 训练 < 1 分钟)

  def hyperband_search(space, max_iter, eta=3):
      # 第 k 层: 分配 budget = max_iter / eta^k 次迭代
      # 每层训练 n = eta^k * s 个配置
      # 保留 top 1/eta 进入下一层
  ```

- [ ] **简单随机搜索 (fallback)**
  ```python
  # 无 scikit-learn 依赖时的纯 Python 方案
  # 随机采样 N 个配置, 全量训练, 选最优
  ```

### 3.2 搜索空间定义

- [ ] **YAML 配置格式**
  ```yaml
  # demo/snake/search_space.yaml
  search:
    algorithm: bayesian        # bayesian | hyperband | random
    max_trials: 50

  model:
    mlp:
      hidden_layers:
        type: int
        range: [2, 5]
      hidden_dim:
        type: choice
        values: [4, 8, 12, 16, 24, 32]
      activation:
        type: categorical
        values: ["relu", "swiglu", "leaky_relu", "tanh"]

  train:
    learning_rate:
      type: loguniform
      range: [0.0001, 0.1]
    batch_size:
      type: choice
      values: [16, 32, 64]
    optimizer:
      type: categorical
      values: ["adam", "muon"]
    l2_reg:
      type: loguniform
      range: [1e-6, 1e-2]

  # 约束条件 (可选)
  constraints:
    - "hidden_dim * hidden_layers <= 64"    # 总参数上限
    - "learning_rate < 0.01 if optimizer == 'muon'"  # Muon 通常用更大的 lr

  metric:
    name: accuracy                           # 或 loss, score
    direction: maximize                      # maximize 或 minimize
    eval_window: 5                           # 最后 N 个 epoch 的滑动平均
  ```

- [ ] **支持的数据类型**
  - `int`: 整数范围 `[min, max]`
  - `choice`: 离散选项 `[v1, v2, ...]`
  - `categorical`: 字符串选项
  - `loguniform`: 对数均匀分布（学习率、正则化系数）
  - `float`: 浮点范围 `[min, max]`

### 3.3 命令行接口

- [ ] **主命令**
  ```bash
  # 基本用法
  python tools/hyper_search/hyper_search.py snake

  # 指定试验次数和搜索算法
  python tools/hyper_search/hyper_search.py snake --trials 30 --algo bayesian

  # 从已有结果继续搜索
  python tools/hyper_search/hyper_search.py snake --resume results/snake_search.json

  # 指定搜索空间文件
  python tools/hyper_search/hyper_search.py snake --space my_space.yaml

  # 列出可用的 demo
  python tools/hyper_search/hyper_search.py --list-demos
  ```

- [ ] **输出**
  ```json
  // results/snake_search_result.json
  {
    "demo": "snake",
    "algorithm": "bayesian",
    "total_trials": 30,
    "best_trial": {
      "id": 17,
      "config": {
        "mlp.hidden_layers": 2,
        "mlp.hidden_dim": 12,
        "mlp.activation": "leaky_relu",
        "train.learning_rate": 0.003,
        "train.batch_size": 32,
        "train.optimizer": "adam"
      },
      "score": 0.923,
      "training_time_sec": 45.2
    },
    "top5_trials": [...],
    "convergence_plot": "results/snake_search.png"
  }
  ```

### 3.4 与 profiler 集成

- [ ] **自动生成网络 JSON**
  ```python
  def config_to_network_json(config):
      # 将搜索配置转换为 profiler 可识别的网络定义 JSON
      return {
          "type": "mlp",
          "hidden_layers": [
              {"size": config["mlp.hidden_dim"], "activation": config["mlp.activation"]}
              for _ in range(config["mlp.hidden_layers"])
          ],
          ...
      }
  ```

- [ ] **调用 profiler + train**
  ```python
  def run_trial(config, demo_name, trial_id):
      # 1. 生成网络 JSON
      network_json = config_to_network_json(config)
      # 2. 调用 profiler
      subprocess.run(["build/profiler/profiler", demo_name,
                       "--network", json.dumps(network_json)])
      # 3. 编译
      subprocess.run(["cmake", "--build", "build", "--target", f"{demo_name}_train"])
      # 4. 训练 + 评估
      result = subprocess.run(["build/train/{demo_name}_train".format(demo_name=demo_name),
                               "--lr", str(config["train.learning_rate"]),
                               ...], capture_output=True)
      # 5. 解析输出中的得分
      return parse_score(result.stdout)
  ```

### 3.5 可视化 (可选)

- [ ] **收敛曲线** — matplotlib 绘制 score vs trial
- [ ] **参数重要性** — 部分依赖图 (partial dependence plot)
- [ ] **并行坐标图** — 可视化高维搜索空间中的好配置

### 3.6 刻意不做的事

- **不做 DARTS** — 可微分 NAS 太重，需要修改训练循环
- **不做 RL NAS** — 搜索成本太高，需要数 GPU-days
- **不搜索网络类型** — MLP vs CNN vs Transformer 由开发者指定
- **不耦合训练代码** — 超参数搜索只管选择参数并调用外部二进制，不修改 C 源码

### 3.7 测试

- [ ] **单元测试** — 搜索空间解析: YAML → 合法配置列表
- [ ] **单元测试** — 贝叶斯优化在已知函数上的收敛 (如 Branin 函数)
- [ ] **单元测试** — Hyperband early-stop 比率正确
- [ ] **集成测试** — snake: 20 次试验, 最优配置得分 ≥ 手动调参得分
- [ ] **集成测试** — 随机搜索 vs 贝叶斯优化: 50 次试验内贝叶斯显著优于随机

## 4. API 设计

```python
# tools/hyper_search/hyper_search.py 的核心类

class SearchSpace:
    """搜索空间定义"""
    def __init__(self, yaml_path: str): ...
    def sample(self) -> dict: ...           # 随机采样一个配置
    def contains(self, config: dict) -> bool: ...  # 检查配置是否在空间内

class BayesianOptimizer:
    """贝叶斯优化"""
    def __init__(self, space: SearchSpace, n_initial: int = 10): ...
    def suggest(self) -> dict: ...          # 建议下一个配置
    def observe(self, config: dict, score: float): ...  # 记录结果
    def best(self) -> tuple[dict, float]: ...  # 当前最优

class HyperbandOptimizer:
    """Hyperband 搜索"""
    def __init__(self, space: SearchSpace, max_iter: int, eta: int = 3): ...
    def suggest(self, budget: int) -> list[dict]: ...  # 返回该预算下的配置列表
    def observe(self, config: dict, score: float, budget: int): ...

class TrialRunner:
    """单次试验执行器"""
    def __init__(self, demo_name: str, build_dir: str): ...
    def run(self, config: dict) -> float: ...  # 返回得分
    def run_async(self, configs: list[dict]) -> list[float]: ...  # 并行 (可选)

# 配置文件: tools/hyper_search/search_spaces/<demo_name>.yaml
# 结果输出: results/<demo_name>_search.json + .png
```

## 5. 涉及文件清单

| 操作 | 文件 |
|---|---|
| **新建** | `tools/hyper_search/hyper_search.py` |
| **新建** | `tools/hyper_search/search_spaces/snake.yaml` |
| **新建** | `tools/hyper_search/search_spaces/mnist.yaml` |
| **新建** | `tools/hyper_search/search_spaces/` — 各 demo 的搜索空间配置 |
| **新建** | `tools/hyper_search/README.md` — 使用说明 |
| 无 C 源文件修改 | (纯 Python 工具, 通过 subprocess 调用外部二进制) |

## 6. 验收标准

1. 贝叶斯优化在已知函数上 50 次试验内收敛到全局最优 ±5%
2. snake 搜索: 20 次试验后最优配置得分 ≥ 手动调参最优得分
3. Hyperband 可以在试验数相同的情况下节省 ≥ 30% 总训练时间
4. `--resume` 可以从先前中断的搜索中继续
5. 搜索空间解析正确拒绝非法配置
6. `--list-demos` 正确列出所有 demo

---

**参考论文:** Snoek et al., Practical Bayesian Optimization (2012); Li et al., Hyperband (2018); PrototypeNAS, arXiv 2603.15106
