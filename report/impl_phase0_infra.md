# 阶段 0：准备与测试基础设施

**关联**: [implementation_steps.md](implementation_steps.md) (总索引)
**设计原则**: 流程驱动 · 根因消除 · 零向后兼容 · **每步必测**

---

## 步骤 P0 — 创建新分支与目录骨架

### 代码变更

```bash
git checkout -b architecture/root-cause-fix

# 创建新增目录
mkdir -p src/utils
mkdir -p cmake
mkdir -p tests/utils
mkdir -p tests/nn
mkdir -p tests/profiler
mkdir -p tests/demo
mkdir -p tests/integration
mkdir -p demo/mnist_common
```

### 步骤测试

本步骤仅有文件系统操作，无代码变更。

**测试用例**:

| # | 测试 | 命令 | 预期 |
|---|------|------|------|
| 1 | 分支已创建 | `git branch --show-current` | 输出 `architecture/root-cause-fix` |
| 2 | 目录存在 | `ls -d src/utils cmake tests/utils tests/nn tests/profiler tests/demo tests/integration demo/mnist_common` | 7 个目录均显示 |
| 3 | 工作区干净 | `git status --porcelain` | 无输出（未跟踪的目录不计） |

### 单步验证

```bash
git branch --show-current
ls -d src/utils/ cmake/ tests/utils/ tests/nn/ tests/profiler/ tests/demo/ tests/integration/ demo/mnist_common/
```

### 全量回归

本步骤无编译产物，跳过全量回归。

### 门控

- [x] 分支名正确
- [x] 7 个目录创建成功
- [ ] → 进入 P0.5

---

## 步骤 P0.5 — 创建测试基础设施

**目标**: 建立最小测试框架 + CTest 集成，为后续所有步骤提供测试能力。

### 代码变更

#### 新建: `tests/test_harness.h`

```c
#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int _tests_run   = 0;
static int _tests_pass  = 0;
static int _tests_fail  = 0;
static const char* _current_test = NULL;

/* 定义测试函数 */
#define TEST(name) \
    static void _test_##name(void)

/* 注册并运行一个测试 */
#define RUN_TEST(name) do { \
    _current_test = #name; \
    _tests_run++; \
    _test_##name(); \
} while(0)

/* 断言宏 */
#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s\n", \
                __FILE__, __LINE__, _current_test, msg); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_EQ_INT(expected, actual, msg) do { \
    int _e = (expected), _a = (actual); \
    if (_e != _a) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected %d, got %d)\n", \
                __FILE__, __LINE__, _current_test, msg, _e, _a); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_EQ_SIZE(expected, actual, msg) do { \
    size_t _e = (expected), _a = (actual); \
    if (_e != _a) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected %zu, got %zu)\n", \
                __FILE__, __LINE__, _current_test, msg, _e, _a); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_EQ_U64(expected, actual, msg) do { \
    uint64_t _e = (expected), _a = (actual); \
    if (_e != _a) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected %llu, got %llu)\n", \
                __FILE__, __LINE__, _current_test, msg, \
                (unsigned long long)_e, (unsigned long long)_a); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_STREQ(expected, actual, msg) do { \
    if (strcmp((expected), (actual)) != 0) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected '%s', got '%s')\n", \
                __FILE__, __LINE__, _current_test, msg, (expected), (actual)); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_NULL(ptr, msg) do { \
    if ((ptr) != NULL) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected NULL, got %p)\n", \
                __FILE__, __LINE__, _current_test, msg, (const void*)(ptr)); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_NOT_NULL(ptr, msg) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected non-NULL)\n", \
                __FILE__, __LINE__, _current_test, msg); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_FLOAT_EQ(expected, actual, epsilon, msg) do { \
    float _e = (expected), _a = (actual), _eps = (epsilon); \
    float _diff = (_e > _a) ? (_e - _a) : (_a - _e); \
    if (_diff > _eps) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected %f, got %f, diff %f)\n", \
                __FILE__, __LINE__, _current_test, msg, _e, _a, _diff); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

/* runner 入口宏 */
#define TEST_MAIN_BEGIN() \
int main(void) { \
    printf("1..<count>\n");

/*
   每个 RUN_TEST 后 _tests_run 递增。
   在 main 末尾手动写 printf 输出结果。

   使用模式:
   int main(void) {
       RUN_TEST(foo);
       RUN_TEST(bar);
       printf("1..%d\n", _tests_run);
       printf("Results: %d pass, %d fail, %d total\n",
              _tests_pass, _tests_fail, _tests_run);
       return _tests_fail > 0 ? 1 : 0;
   }
*/

#endif /* TEST_HARNESS_H */
```

#### 新建: `tests/smoke_test.c`

验证 harness 本身可正常工作：

```c
#include "test_harness.h"

TEST(harness_assert_true_works)  { ASSERT_TRUE(1 == 1, "1 equals 1"); }
TEST(harness_assert_eq_works)    { ASSERT_EQ_INT(42, 42, "42 equals 42"); }
TEST(harness_assert_streq_works) { ASSERT_STREQ("hello", "hello", "strings match"); }

int main(void) {
    RUN_TEST(harness_assert_true_works);
    RUN_TEST(harness_assert_eq_works);
    RUN_TEST(harness_assert_streq_works);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

#### 新建: `tests/CMakeLists.txt`

```cmake
# 启用 CTest
enable_testing()

# 添加测试目标辅助宏
function(action_c_add_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )
    target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(${TEST_NAME} PRIVATE)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Smoke test
action_c_add_test(test_harness_smoke smoke_test.c)
```

#### 修改: 顶层 `CMakeLists.txt`

在 `add_subdirectory(src)` 之后追加：

```cmake
# 添加测试（仅在启用时）
option(BUILD_TESTING "Enable CTest" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
```

### 步骤测试

**测试文件**: `tests/smoke_test.c`（已包含）

**测试用例**:

| # | 测试名 | 内容 | 预期 |
|---|--------|------|------|
| 1 | `harness_assert_true_works` | `ASSERT_TRUE(1 == 1, ...)` | PASS |
| 2 | `harness_assert_eq_works` | `ASSERT_EQ_INT(42, 42, ...)` | PASS |
| 3 | `harness_assert_streq_works` | `ASSERT_STREQ("hello", "hello", ...)` | PASS |

额外的编译期测试：

| # | 测试名 | 命令 | 预期 |
|---|--------|------|------|
| 4 | harness 语法 | `gcc -fsyntax-only tests/test_harness.h` | 无错误 |
| 5 | CTest 可发现 | `ctest -N` | 显示 1 个测试 `test_harness_smoke` |

### 单步验证

```bash
# 1. 编译检查 harness
gcc -fsyntax-only tests/test_harness.h

# 2. 构建测试目录
cmake -B build/ -S . -DBUILD_TESTING=ON
cmake --build build/ --target test_harness_smoke

# 3. 运行单步测试
ctest -R test_harness_smoke --output-on-failure
```

**预期输出**:
```
Test project .../build
    Start 1: test_harness_smoke
1/1 Test #1: test_harness_smoke ............... Passed

100% tests passed, 0 tests failed out of 1
```

### 全量回归

```bash
# 当前全量 = 仅一个测试
ctest --output-on-failure

# 确认项目现有 demo 不受影响（编译检查）
cmake -B build/ -S .
cmake --build build/ --config Debug 2>&1 | grep -i "error"
```

**预期**: ctest 1/1 通过。`grep -i error` 无输出。

### 门控

- [x] `test_harness_smoke` 通过（3/3 断言）
- [x] `ctest` 运行成功（1/1）
- [x] 项目全量编译无错误
- [ ] → **git commit** `"feat: add test infrastructure (harness + CTest)"`
- [ ] → 进入轨 A/B/C 并行开发
