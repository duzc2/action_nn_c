# action_c 架构修复实施步骤（含全量测试）

**关联报告**: `report/static_analysis_report.md` (发现) → `report/remediation_plan.md` (方案)
**设计原则**: 流程驱动 · 根因消除 · 零向后兼容 · **每步必测**

---

## 文档导航

本文档为总索引。具体实施步骤拆分为以下子文档：

| 文档 | 内容 | 步骤数 |
|------|------|--------|
| [impl_phase0_infra.md](impl_phase0_infra.md) | **阶段0**: P0 准备 + P0.5 测试基础设施 | 2 |
| [impl_phase1_track_a.md](impl_phase1_track_a.md) | **轨A**: 基础设施与内存 (A1-A9) | 9 |
| [impl_phase1_track_b.md](impl_phase1_track_b.md) | **轨B**: 接口与去重 (B1-B10) | 10 |
| [impl_phase1_track_c.md](impl_phase1_track_c.md) | **轨C**: 构建系统配置化 (C1-C13) | 13 |
| [impl_phase2_integration.md](impl_phase2_integration.md) | **阶段2**: Profiler 流水线集成 (I1-I6) | 6 |
| [impl_phase3_4_finish.md](impl_phase3_4_finish.md) | **阶段3+4**: 前端修复 (D1-D8) + 验证清理 (V1-V5) | 13 |

---

## 实施总览

```
阶段 0: 准备工作 + 测试基础设施（所有人，串行）
  步骤 P0:  创建分支 + 新目录骨架
  步骤 P0.5: 创建测试框架 + CTest 集成

阶段 1: 三轨并行（3 人同时推进，各自独立测试）
  轨 A (步骤 A1-A9):  基础设施 + 内存    [RC6 → RC1]
  轨 B (步骤 B1-B10): 接口 + 去重        [RC3 → RC5]
  轨 C (步骤 C1-C13): 构建系统配置化     [RC4]

阶段 2: 集成（轨 A/B/C 全部完成后）
  步骤 I1-I6: Profiler 流水线            [RC2]

阶段 3: 前端 + 独立修复（可与阶段 1-2 并行）
  步骤 D1-D8: Web Editor + Wasm + 独立问题

阶段 4: 清理 + 验证（最后）
  步骤 V1-V5: 删除死代码 + 全量验证
```

---

## 根因追溯

| 根因 | 影响的步骤 | 消除问题数 |
|------|-----------|-----------|
| RC1 无统一内存模型 | A1-A9 | ~25 |
| RC2 无 Pipeline 抽象 | I1-I6 | ~15 |
| RC3 无类型安全后端接口 | B1-B6 | ~20 |
| RC4 构建为脚本非配置 | C1-C13 | ~35 |
| RC5 代码重复为架构常态 | B7-B10 | ~25 |
| RC6 无错误处理基础设施 | A1-A5, I5 | ~20 |

---

## 测试工作流规范（每步强制执行）

### 测试框架

项目测试基于 **最小自定义 harness + CTest**，零外部依赖。

- **Harness 文件**: `tests/test_harness.h`（约 50 行，定义 `TEST()` `ASSERT_EQ()` `ASSERT_TRUE()` 等宏）
- **测试目录结构**:
  ```
  tests/
    CMakeLists.txt
    test_harness.h
    utils/          # src/utils/ 单元测试
    nn/             # src/nn/ 单元测试
    profiler/       # src/profiler/ 单元测试
    demo/           # demo 回归测试 (.cmake 脚本)
    integration/    # 跨模块集成测试
  ```
- **运行器**: CTest (`ctest`)，每个测试文件对应一个 `add_test()` 注册

### 每步标准流程

```
┌─────────────────────────────────────────────────────┐
│ 步骤 N: 标题                                         │
│                                                      │
│  1. 编写代码变更（创建/修改源文件）                      │
│  2. 编写步骤测试（创建测试文件，含 3-10 个测试用例）       │
│  3. 单步验证（构建并运行本步骤测试）                      │
│     cmake --build build/ --target test_<name>         │
│     ctest -R test_<name> --output-on-failure          │
│  4. 全量回归（运行所有已累积的测试）                      │
│     ctest --output-on-failure                         │
│     cmake --build build/ --config Debug 2>&1 | grep error │
│  5. 门控：二者均通过 → git commit（检查点）              │
│     任一失败 → 修复后回到步骤 3                          │
│  6. 进入下一步                                         │
└─────────────────────────────────────────────────────┘
```

### 全量测试套件定义

"全量回归" 随进度增长：

| 阶段 | 全量测试包含 |
|------|------------|
| P0.5 后 | test_harness compile check |
| 轨A 每步后 | 已累积的所有 utils/ 单元测试 |
| 轨B 每步后 | utils/ 测试 + 已累积的 nn/ 单元测试 + 已注册的 demo 回归 |
| 轨C 每步后 | build 系统测试 (cmake configure + preset 验证) |
| 阶段2 每步后 | 全量单元测试 + profiler 测试 + 全量 demo 回归 |
| 阶段4 | 全量单元测试 + 全量 demo 回归 + sanitizer + 静态分析 |

### 测试用例数量估算

| 阶段 | 新增测试文件 | 新增测试用例 | 累积测试用例 |
|------|------------|------------|------------|
| P0.5 | 1 (smoke) | 3 | 3 |
| 轨A | 5 | ~35 | ~38 |
| 轨B | 6 | ~40 | ~78 |
| 轨C | 5 | ~25 | ~103 |
| 阶段2 | 5 | ~35 | ~138 |
| 阶段3+4 | 6 | ~30 | ~168 |

---

## 实施检查清单总表

### 阶段 0: 准备
- [ ] P0: 创建分支与目录骨架
- [ ] P0.5: 测试框架 + CTest 集成

### 轨 A（基础设施 + 内存）
- [ ] A1: ActionCError 枚举 + 测试
- [ ] A2: 日志 + 断言宏 + 测试
- [ ] A3: 安全整数运算 + 测试
- [ ] A4: Arena Allocator + 测试
- [ ] A5: 全工程错误码替换 + 回归
- [ ] A6: MLP Context 集成 Arena + 测试
- [ ] A7: MLP 后端 Arena 接入 + 测试
- [ ] A8: GNN 后端 Arena 接入 + 测试
- [ ] A9: Transformer + RNN Arena 接入 + 测试

### 轨 B（接口 + 去重）
- [ ] B1: nn_backend.h VTable 接口 + 测试
- [ ] B2: nn_weight_header.h + 测试
- [ ] B3: 注册表重写为 VTable + 测试
- [ ] B4: 删除旧桥接调度层 + 回归
- [ ] B5: MLP VTable 实现 + 测试
- [ ] B6: GNN/Transformer/CNN/RNN VTable + 测试
- [ ] B7: CNN + CNN_Dual_Pool 合并 + 回归
- [ ] B8: Transformer 前向合并 + 回归
- [ ] B9: MNIST 读取器合并 + 回归
- [ ] B10: restrict + ActivationFn + safe_math + 测试

### 轨 C（构建系统）
- [ ] C1: cmake/compiler_flags.cmake + 测试
- [ ] C2: cmake/version.cmake + 测试
- [ ] C3: src/ + demo/ 使用 cmake/ 模块 + 回归
- [ ] C4: CONFIGURE_DEPENDS + 测试
- [ ] C5: demo 构建依赖 + 测试
- [ ] C6: 路径计算修复 + 回归
- [ ] C7: CMakePresets.json + 测试
- [ ] C8: C_EXTENSIONS OFF 统一 + 回归
- [ ] C9: wasm CMakeLists.txt 修复 + 测试
- [ ] C10: build_wasm.sh 修复 + 测试
- [ ] C11: CS 工具路径修复 + 回归
- [ ] C12: build_demos.ps1 重写 + 测试
- [ ] C13: demo 脚本平台检测 + 回归

### 阶段 2: Profiler 集成
- [ ] I1: FlatNetwork 类型定义 + 测试
- [ ] I2: 验证函数接收 FlatNetwork + 测试
- [ ] I3: 管道中间结果类型 + 测试
- [ ] I4: profiler_generate_v2 管道化 + 测试
- [ ] I5: write_file 错误码 + 测试
- [ ] I6: StringHashMap 实现 + 测试

### 阶段 3: 前端与独立修复
- [ ] D1: Web Editor 遗留删除
- [ ] D2: Web Editor 安全加固
- [ ] D3: Web Editor 工程完善 + 测试
- [ ] D4: 连接代码生成 + 测试
- [ ] D5: Wasm 运行时补全 + 测试
- [ ] D6: 独立小修复 (6 项)
- [ ] D7: CS JSON 加固 + 测试
- [ ] D8: 脚本去重 + 测试

### 阶段 4: 验证
- [ ] V1: 删除死代码
- [ ] V2: 全量构建验证 + ctest 全量
- [ ] V3: Sanitizer 验证
- [ ] V4: 静态分析验证
- [ ] V5: 文档更新
