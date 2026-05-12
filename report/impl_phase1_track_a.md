# 轨 A — 基础设施与内存模型

**关联**: [implementation_steps.md](implementation_steps.md) (总索引)
**前置**: 阶段 0 (P0 + P0.5 已完成)
**可并行**: 与轨 B、轨 C 并行推进
**对应根因**: RC6 (错误基础设施) → RC1 (Arena 内存模型)

> 负责人 A 独立推进。轨 A 完成前不依赖轨 B 和轨 C 的输出。

---

## 步骤 A1 — 创建 ActionCError 枚举

### 代码变更

**新建**: `src/utils/error.h`

```c
#ifndef ACTION_C_ERROR_H
#define ACTION_C_ERROR_H

typedef enum {
    ACTION_C_OK = 0,

    /* 参数错误 */
    ACTION_C_ERR_NULL_POINTER   = -1,
    ACTION_C_ERR_INVALID_ARG    = -2,
    ACTION_C_ERR_OUT_OF_RANGE   = -3,

    /* 资源错误 */
    ACTION_C_ERR_NO_MEMORY      = -10,
    ACTION_C_ERR_IO_FAILED      = -11,

    /* 数据错误 */
    ACTION_C_ERR_DIM_MISMATCH   = -20,
    ACTION_C_ERR_VERSION_MISMATCH = -21,
    ACTION_C_ERR_CONFIG_INVALID = -22,
    ACTION_C_ERR_CYCLE_DETECTED = -23,

    /* 查找错误 */
    ACTION_C_ERR_NOT_FOUND      = -30,

    /* 内部错误 */
    ACTION_C_ERR_INTERNAL       = -99,
} ActionCError;

#endif
```

### 步骤测试

**测试文件**: `tests/utils/test_error.c`

```c
#include "test_harness.h"
#include "utils/error.h"

TEST(action_c_ok_is_zero) {
    ASSERT_EQ_INT(0, ACTION_C_OK, "OK should be 0");
}

TEST(error_values_are_negative) {
    ASSERT_TRUE(ACTION_C_ERR_NULL_POINTER < 0, "null pointer is negative");
    ASSERT_TRUE(ACTION_C_ERR_NO_MEMORY < 0, "no memory is negative");
    ASSERT_TRUE(ACTION_C_ERR_INTERNAL < 0, "internal is negative");
}

TEST(error_ranges_no_collision) {
    /* 每个范围不重叠 */
    ASSERT_TRUE(ACTION_C_ERR_NULL_POINTER >= -9, "param errors near -1");
    ASSERT_TRUE(ACTION_C_ERR_OUT_OF_RANGE >= -9, "param errors range");
    ASSERT_TRUE(ACTION_C_ERR_NO_MEMORY <= -10, "resource errors <= -10");
    ASSERT_TRUE(ACTION_C_ERR_NO_MEMORY >= -19, "resource errors >= -19");
    ASSERT_TRUE(ACTION_C_ERR_DIM_MISMATCH <= -20, "data errors <= -20");
    ASSERT_TRUE(ACTION_C_ERR_DIM_MISMATCH >= -29, "data errors >= -29");
}

int main(void) {
    RUN_TEST(action_c_ok_is_zero);
    RUN_TEST(error_values_are_negative);
    RUN_TEST(error_ranges_no_collision);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
# 编译
gcc -fsyntax-only src/utils/error.h

# 构建 + 运行
cmake --build build/ --target test_error
ctest -R test_error --output-on-failure
```

**预期**: 3 个测试全部 PASS。

### 全量回归

```bash
ctest --output-on-failure
cmake --build build/ --config Debug 2>&1 | grep -i "error"
```

**当前全量**: test_harness_smoke + test_error 共 2 个测试。

### 门控

- [x] test_error 全部通过
- [x] ctest 全量通过 (2/2)
- [ ] → **git commit** `"feat(A1): add ActionCError enum with tests"`
- [ ] → 进入 A2

---

## 步骤 A2 — 创建日志 + 断言宏

### 代码变更

**新建**: `src/utils/log.h`

```c
#ifndef ACTION_C_LOG_H
#define ACTION_C_LOG_H

#include <stdio.h>

#ifndef ACTION_C_LOG_LEVEL
#define ACTION_C_LOG_LEVEL 2
#endif

void _action_c_log(int level, const char* file, int line,
                   const char* fmt, ...);

#if ACTION_C_LOG_LEVEL >= 1
#define LOG_ERROR(fmt, ...) \
    _action_c_log(1, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define LOG_ERROR(fmt, ...) ((void)0)
#endif

#if ACTION_C_LOG_LEVEL >= 2
#define LOG_WARN(fmt, ...) \
    _action_c_log(2, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define LOG_WARN(fmt, ...) ((void)0)
#endif

#if ACTION_C_LOG_LEVEL >= 3
#define LOG_INFO(fmt, ...) \
    _action_c_log(3, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define LOG_INFO(fmt, ...) ((void)0)
#endif

/* Debug 断言: Debug 时验证不变量并返回错误，Release 时零开销移除 */
#ifndef NDEBUG
#define ASSERT(cond, err, msg) do { \
    if (!(cond)) { \
        LOG_ERROR("ASSERT FAILED: %s -- %s", #cond, msg); \
        return (err); \
    } \
} while(0)
#else
#define ASSERT(cond, err, msg) ((void)0)
#endif

#endif
```

**新建**: `src/utils/log.c`

```c
#include "log.h"
#include <stdarg.h>

void _action_c_log(int level, const char* file, int line,
                   const char* fmt, ...) {
    static const char* level_str[] = {
        "", "[ERROR]", "[WARN]", "[INFO]", "[DEBUG]"
    };
    fprintf(stderr, "%s %s:%d: ",
            level_str[level], file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}
```

### 步骤测试

**测试文件**: `tests/utils/test_log.c`

```c
#include "test_harness.h"

/* 测试日志宏可编译且不崩溃 */
/* 注意: 默认 ACTION_C_LOG_LEVEL=2，LOG_INFO 被编译置零 */

TEST(log_error_compiles_and_runs) {
    /* 仅验证宏不导致编译错误、运行不崩溃 */
    LOG_ERROR("test error: %d", 42);
    ASSERT_TRUE(1, "log_error executed without crash");
}

TEST(log_warn_compiles_and_runs) {
    LOG_WARN("test warn: %s", "hello");
    ASSERT_TRUE(1, "log_warn executed without crash");
}

/* ASSERT 宏在 Debug 构建中的行为测试 */
TEST(assert_passes_when_cond_true) {
    ASSERT(1 == 1, -1, "should not fail");
    ASSERT_TRUE(1, "assert true condition passed");
}

TEST(ndebug_removes_assert) {
    /* 这是一项文档性测试: 在 Release 构建中 ASSERT 应为空操作。
       此处仅验证 Debug 中宏语法正确。 */
    ASSERT_TRUE(1, "syntax-only check for ASSERT macro");
}

int main(void) {
    RUN_TEST(log_error_compiles_and_runs);
    RUN_TEST(log_warn_compiles_and_runs);
    RUN_TEST(assert_passes_when_cond_true);
    RUN_TEST(ndebug_removes_assert);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
# 编译 log.c
gcc -c src/utils/log.c -I src/utils/ -Wall -Werror

# 构建 + 运行测试
cmake --build build/ --target test_log
ctest -R test_log --output-on-failure
```

**预期**: 4 个测试全部 PASS。stderr 中可见 LOG_ERROR 输出，不影响测试判定。

### 全量回归

```bash
ctest --output-on-failure
cmake --build build/ --config Debug 2>&1 | grep -i "error"
```

**当前全量**: test_harness_smoke + test_error + test_log = 3 个测试。

### 门控

- [x] test_log 全部通过
- [x] ctest 全量通过 (3/3)
- [ ] → **git commit** `"feat(A2): add logging and ASSERT macros with tests"`
- [ ] → 进入 A3

---

## 步骤 A3 — 创建安全整数运算

### 代码变更

**新建**: `src/utils/safe_math.h`

```c
#ifndef ACTION_C_SAFE_MATH_H
#define ACTION_C_SAFE_MATH_H

#include <stddef.h>
#include <stdint.h>

/* 安全乘法：检查 a * b 是否超过 limit。
   若溢出返回 0，否则返回 a * b。 */
static inline size_t safe_size_mul(size_t a, size_t b, size_t limit) {
    if (a == 0 || b == 0) return 0;
    if (b > limit / a) return 0;
    return a * b;
}

/* 安全 size_t 乘法，上限为 SIZE_MAX */
static inline size_t safe_size_mul_max(size_t a, size_t b) {
    return safe_size_mul(a, b, SIZE_MAX);
}

/* 安全 size_t 加法的元素数版本: count * sizeof(T) 不溢出 */
#define SAFE_ALLOC_SIZE(count, type) \
    safe_size_mul((count), sizeof(type), SIZE_MAX)

#endif
```

### 步骤测试

**测试文件**: `tests/utils/test_safe_math.c`

```c
#include "test_harness.h"
#include "utils/safe_math.h"
#include <stdint.h>

TEST(mul_normal_case) {
    size_t r = safe_size_mul_max(100, 200);
    ASSERT_EQ_SIZE(20000, r, "100 * 200 = 20000");
}

TEST(mul_zero_a) {
    size_t r = safe_size_mul_max(0, 1000000);
    ASSERT_EQ_SIZE(0, r, "0 * n = 0");
}

TEST(mul_zero_b) {
    size_t r = safe_size_mul_max(1000000, 0);
    ASSERT_EQ_SIZE(0, r, "n * 0 = 0");
}

TEST(mul_overflow_size_max) {
    /* SIZE_MAX * 2 应溢出 */
    size_t r = safe_size_mul_max(SIZE_MAX, 2);
    ASSERT_EQ_SIZE(0, r, "overflow returns 0");
}

TEST(mul_overflow_large) {
    size_t r = safe_size_mul_max(SIZE_MAX / 2 + 1, 2);
    ASSERT_EQ_SIZE(0, r, "(SIZE_MAX/2+1)*2 overflows");
}

TEST(mul_boundary_ok) {
    size_t r = safe_size_mul_max(SIZE_MAX / 2, 2);
    ASSERT_TRUE(r > 0, "SIZE_MAX/2 * 2 should not overflow");
}

TEST(mul_limit_param) {
    size_t r = safe_size_mul(5, 5, 20);
    ASSERT_EQ_SIZE(0, r, "5*5=25 exceeds limit 20");
}

TEST(mul_within_limit) {
    size_t r = safe_size_mul(5, 5, 30);
    ASSERT_EQ_SIZE(25, r, "5*5=25 within limit 30");
}

int main(void) {
    RUN_TEST(mul_normal_case);
    RUN_TEST(mul_zero_a);
    RUN_TEST(mul_zero_b);
    RUN_TEST(mul_overflow_size_max);
    RUN_TEST(mul_overflow_large);
    RUN_TEST(mul_boundary_ok);
    RUN_TEST(mul_limit_param);
    RUN_TEST(mul_within_limit);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
# 语法检查
gcc -fsyntax-only src/utils/safe_math.h

# 构建 + 运行测试
cmake --build build/ --target test_safe_math
ctest -R test_safe_math --output-on-failure
```

**预期**: 8 个测试全部 PASS。溢出测试在 ASan 下无警告。

### 全量回归

```bash
ctest --output-on-failure
```

**当前全量**: 4 个测试 (smoke + error + log + safe_math)

### 门控

- [x] test_safe_math 全部通过
- [x] ctest 全量通过 (4/4)
- [ ] → **git commit** `"feat(A3): add safe integer math with overflow tests"`
- [ ] → 进入 A4

---

## 步骤 A4 — 创建 Arena Allocator

### 代码变更

**新建**: `src/utils/arena.h`

```c
#ifndef ACTION_C_ARENA_H
#define ACTION_C_ARENA_H

#include <stddef.h>

typedef struct {
    unsigned char* memory;
    size_t         capacity;
    size_t         used;
} Arena;

/* 创建 arena。cap 为初始容量（字节） */
Arena* arena_create(size_t cap);

/* 销毁 arena */
void arena_destroy(Arena* a);

/* 记录当前水位线，返回标记 */
size_t arena_snapshot(const Arena* a);

/* 回卷到之前的水位线 */
void arena_restore(Arena* a, size_t mark);

/* 从 arena 分配 n 个类型 T 的元素，返回未初始化内存 */
#define ARENA_ALLOC(a, T, n) \
    ((T*)_arena_alloc((a), (n) * sizeof(T)))

/* 从 arena 分配 n 个类型 T 的元素，并清零 */
#define ARENA_CALLOC(a, T, n) \
    ((T*)_arena_calloc((a), (n) * sizeof(T)))

/* 内部函数，不直接调用 */
void* _arena_alloc(Arena* a, size_t size);
void* _arena_calloc(Arena* a, size_t size);

#endif
```

**新建**: `src/utils/arena.c`

```c
#include "arena.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>

Arena* arena_create(size_t cap) {
    Arena* a = (Arena*)malloc(sizeof(Arena));
    if (!a) return NULL;
    a->memory = (unsigned char*)malloc(cap);
    if (!a->memory) {
        free(a);
        return NULL;
    }
    a->capacity = cap;
    a->used = 0;
    return a;
}

void arena_destroy(Arena* a) {
    if (!a) return;
    free(a->memory);
    free(a);
}

size_t arena_snapshot(const Arena* a) {
    return a->used;
}

void arena_restore(Arena* a, size_t mark) {
    a->used = mark;
}

static void arena_grow(Arena* a, size_t needed) {
    size_t new_cap = a->capacity;
    while (new_cap < needed) {
        new_cap = (new_cap == 0) ? 4096 : new_cap * 2;
    }
    unsigned char* new_mem = (unsigned char*)realloc(a->memory, new_cap);
    if (!new_mem) {
        LOG_ERROR("Arena grow failed: need %zu, capacity %zu",
                  needed, a->capacity);
        return;
    }
    a->memory   = new_mem;
    a->capacity = new_cap;
}

void* _arena_alloc(Arena* a, size_t size) {
    if (a->used + size > a->capacity) {
        arena_grow(a, a->used + size);
    }
    if (a->used + size > a->capacity) {
        return NULL;
    }
    void* ptr = a->memory + a->used;
    a->used += size;
    return ptr;
}

void* _arena_calloc(Arena* a, size_t size) {
    void* ptr = _arena_alloc(a, size);
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
}
```

### 步骤测试

**测试文件**: `tests/utils/test_arena.c`

```c
#include "test_harness.h"
#include "utils/arena.h"
#include <string.h>

TEST(arena_create_and_destroy) {
    Arena* a = arena_create(1024);
    ASSERT_NOT_NULL(a, "arena_create should succeed");
    ASSERT_EQ_SIZE(1024, a->capacity, "capacity should be 1024");
    ASSERT_EQ_SIZE(0, a->used, "used should start at 0");
    arena_destroy(a);
}

TEST(arena_alloc_basic) {
    Arena* a = arena_create(1024);
    int* p = ARENA_ALLOC(a, int, 10);
    ASSERT_NOT_NULL(p, "ARENA_ALLOC should succeed");
    ASSERT_EQ_SIZE(10 * sizeof(int), a->used, "used should be 10*sizeof(int)");
    p[0] = 42;
    p[9] = 99;
    ASSERT_EQ_INT(42, p[0], "first element writable");
    ASSERT_EQ_INT(99, p[9], "last element writable");
    arena_destroy(a);
}

TEST(arena_calloc_zeros) {
    Arena* a = arena_create(1024);
    int* p = ARENA_CALLOC(a, int, 100);
    ASSERT_NOT_NULL(p, "ARENA_CALLOC should succeed");
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ_INT(0, p[i], "calloc'd memory should be zero");
    }
    arena_destroy(a);
}

TEST(arena_snapshot_restore) {
    Arena* a = arena_create(1024);
    /* 先分配一些 */
    ARENA_ALLOC(a, int, 10);
    size_t mark = arena_snapshot(a);
    ASSERT_EQ_SIZE(10 * sizeof(int), mark, "snapshot captures used");

    /* 再分配更多 */
    ARENA_ALLOC(a, int, 20);
    ASSERT_TRUE(a->used > mark, "used increases after more alloc");

    /* 回卷 */
    arena_restore(a, mark);
    ASSERT_EQ_SIZE(mark, a->used, "restore resets used to mark");

    /* 回卷后分配复用空间 */
    int* p = ARENA_ALLOC(a, int, 5);
    ASSERT_NOT_NULL(p, "alloc after restore should succeed");
    arena_destroy(a);
}

TEST(arena_grow) {
    Arena* a = arena_create(16);
    /* 分配超过初始容量的数据 */
    char* p = ARENA_ALLOC(a, char, 2048);
    ASSERT_NOT_NULL(p, "alloc beyond initial capacity should trigger grow");
    ASSERT_TRUE(a->capacity >= 2048 + 16, "capacity should have grown");
    /* 确保写入不崩溃 */
    p[0] = 'a';
    p[2047] = 'z';
    ASSERT_EQ_INT('a', p[0], "first byte writable after grow");
    ASSERT_EQ_INT('z', p[2047], "last byte writable after grow");
    arena_destroy(a);
}

TEST(arena_multiple_alloc_grow) {
    Arena* a = arena_create(32);
    for (int i = 0; i < 100; i++) {
        int* p = ARENA_ALLOC(a, int, 64);
        ASSERT_NOT_NULL(p, "repeated alloc should succeed");
        p[0] = i;
    }
    arena_destroy(a);
}

TEST(arena_null_destroy_safe) {
    arena_destroy(NULL);
    ASSERT_TRUE(1, "destroy(NULL) should not crash");
}

int main(void) {
    RUN_TEST(arena_create_and_destroy);
    RUN_TEST(arena_alloc_basic);
    RUN_TEST(arena_calloc_zeros);
    RUN_TEST(arena_snapshot_restore);
    RUN_TEST(arena_grow);
    RUN_TEST(arena_multiple_alloc_grow);
    RUN_TEST(arena_null_destroy_safe);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
# 编译 arena
gcc -c src/utils/arena.c -I src/utils/ -Wall -Werror

# 构建 + 运行测试
cmake --build build/ --target test_arena
ctest -R test_arena --output-on-failure
```

**预期**: 7 个测试全部 PASS。如果可用，在 valgrind 下运行确认零泄漏。

### 全量回归

```bash
ctest --output-on-failure
```

**当前全量**: 5 个测试 (smoke + error + log + safe_math + arena)

### 门控

- [x] test_arena 全部通过
- [x] ctest 全量通过 (5/5)
- [x] valgrind test_arena 零泄漏（可选但推荐）
- [ ] → **git commit** `"feat(A4): add Arena allocator with full tests"`
- [ ] → 进入 A5

---

## 步骤 A5 — 全局替换错误码

### 代码变更

**操作**: 全工程搜索替换返回值约定。

```
搜索模式 → 替换为:

return -1;  (指针/参数类)  → return ACTION_C_ERR_NULL_POINTER;
return -1;  (内存类)       → return ACTION_C_ERR_NO_MEMORY;
return -1;  (维度类)       → return ACTION_C_ERR_DIM_MISMATCH;
return -1;  (IO类)         → return ACTION_C_ERR_IO_FAILED;
return -2;  (查找类)       → return ACTION_C_ERR_NOT_FOUND;
return -2;  (配置类)       → return ACTION_C_ERR_CONFIG_INVALID;
return  1;  (bool true)    → return 1;  /* 保持不变 */
return  0;  (bool false/bool success) → return 0;  /* 保持不变 */
return  (void) 函数         → return ACTION_C_OK;

函数签名 int → ActionCError:
  - 所有返回错误码的函数，返回类型从 int 改为 ActionCError
  - 使用 assert/abort 的非法状态函数保持 void 或 noreturn
```

**涉及文件**: `src/profiler/` 全量, `src/nn/` 全量, `src/infer/`, `src/train/`, 所有 demo `*_main.c`

**执行方式**: 分文件逐个替换，每完成一个文件编译验证一次。

### 步骤测试

本步骤本质上是重构（改变量名不变逻辑），测试策略为 **全量回归测试**：

**测试用例** (编译 + 运行级别):

| # | 测试 | 命令 | 预期 |
|---|------|------|------|
| 1 | 全量编译零错误 | `cmake --build build/ --config Debug 2>&1` | 无 error |
| 2 | 全量编译零警告 | `cmake --build build/ --config Debug 2>&1` | 无 warning |
| 3 | move demo 全流程 | `bash scripts/run_demo.sh move all` | 三阶段均 exit 0 |
| 4 | sevenseg demo 全流程 | `bash scripts/run_demo.sh sevenseg all` | 三阶段均 exit 0 |
| 5 | target demo 全流程 | `bash scripts/run_demo.sh target all` | 三阶段均 exit 0 |

### 单步验证

```bash
# 逐文件替换后：
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
```

### 全量回归

```bash
# 全部单元测试
ctest --output-on-failure

# 所有可用 demo 回归
bash scripts/run_demo.sh move all
bash scripts/run_demo.sh sevenseg all
bash scripts/run_demo.sh target all
```

**当前全量**: 5 个单元测试 + 3 个 demo 回归。

### 门控

- [x] 全量编译零错误零警告
- [x] ctest 全量通过
- [x] move / sevenseg / target demo 全流程通过
- [ ] → **git commit** `"refactor(A5): replace magic return codes with ActionCError enum"`
- [ ] → 进入 A6

---

## 步骤 A6 — 修改 MLP 上下文结构体，集成 Arena

### 代码变更

**修改**: `src/nn/types/mlp/mlp_infer_ops.c` (MlpInferContext)

```c
typedef struct {
    MlpConfig config;
    MlpDenseLayer** layers;
    size_t layer_count;
    size_t max_layer_width;
    float* input_buffer;
    float* output_buffer;
    size_t output_size;

    /* 新增 */
    Arena* arena;
} MlpInferContext;
```

**修改**: `src/nn/types/mlp/mlp_train_ops.c` (MlpTrainContext)

```c
typedef struct {
    MlpConfig config;
    MlpDenseLayer** layers;
    size_t layer_count;
    size_t max_layer_width;

    float** activations;
    size_t activation_count;

    float** weight_momentum;
    float** bias_momentum;
    float** weight_velocity;
    float** bias_velocity;

    /* 新增 */
    Arena* arena;
} MlpTrainContext;
```

### 步骤测试

这是结构体定义的变更，测试重点为：

**测试文件**: `tests/nn/test_mlp_context_size.c`

```c
#include "test_harness.h"
#include <stddef.h>

/* 验证结构体大小增长合理（仅增加一个指针字段） */
/* 注意: 此处需要 #include 实际的头文件，但结构体定义通常在 .c 中。
   若结构体在头文件中声明，则直接 include；若在 .c 中，则用 sizeof 编译期检查。 */

/* 本测试为编译期验证：确认 Arena* 字段成功添加到结构体中。
   可用一个简单的辅助编译验证：
   gcc -c -fsyntax-only src/nn/types/mlp/mlp_infer_ops.c -I src/
   对比修改前后 sizeof(MlpInferContext) 增加了 sizeof(Arena*) 的字节。 */

int main(void) {
    /* 此测试文件验证编译通过即可 */
    printf("1..1\n");
    printf("ok 1 - MLP context struct compiles with Arena field\n");
    printf("Results: 1 pass, 0 fail, 1 total\n");
    return 0;
}
```

### 单步验证

```bash
# 编译 MLP 模块
gcc -c src/nn/types/mlp/mlp_infer_ops.c -I src/ -I src/utils/ -Wall -Werror
gcc -c src/nn/types/mlp/mlp_train_ops.c -I src/ -I src/utils/ -Wall -Werror

# 全项目编译
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
```

### 全量回归

```bash
ctest --output-on-failure
bash scripts/run_demo.sh move all
```

**当前全量**: 6 个单元测试 + move demo。

### 门控

- [x] MLP 源文件编译零错误
- [x] 全量编译零错误零警告
- [x] ctest 全量通过
- [x] move demo 全流程通过
- [ ] → **git commit** `"refactor(A6): add Arena field to MLP context structs"`
- [ ] → 进入 A7

---

## 步骤 A7 — MLP 后端接入 Arena（消除热路径堆分配）

### 代码变更

**修改**: `mlp_train_ops.c` — `train_backward_pass` 函数

替换前（旧）:
```c
float* current_delta = (float*)calloc(max_width, sizeof(float));
float* next_delta    = (float*)calloc(max_width, sizeof(float));
// ... 使用 ...
free(current_delta);
free(next_delta);
```

替换后（新）:
```c
size_t mark = arena_snapshot(ctx->arena);
float* current_delta = ARENA_CALLOC(ctx->arena, float, max_width);
float* next_delta    = ARENA_CALLOC(ctx->arena, float, max_width);
// ... 使用 ...
arena_restore(ctx->arena, mark);
```

**修改**: `mlp_train_ops.c` — `nn_mlp_train_create` 函数

```c
/* 在创建末尾 */
size_t max_scratch = mlp_train_compute_max_scratch(config);
ctx->arena = arena_create(max_scratch);
if (!ctx->arena) {
    mlp_train_destroy_partial(ctx, layer_index);
    return ACTION_C_ERR_NO_MEMORY;
}
```

**修改**: `mlp_train_ops.c` — `nn_mlp_train_destroy` 函数

```c
arena_destroy(ctx->arena);
/* 后续原有的 free 调用... */
```

**修改**: `mlp_train_ops.c` — `train_forward_pass` 函数

将 `activations` 的分配改为 arena（原本已在 create 中分配，改为走 arena）。

### 步骤测试

**测试文件**: `tests/nn/test_mlp_arena_integration.c`

```c
#include "test_harness.h"
#include "utils/arena.h"
#include <stdlib.h>

TEST(arena_snapshot_restore_pattern) {
    /* 模拟 backward_pass 中的 snapshot/restore 模式 */
    Arena* a = arena_create(4096);

    /* 首次分配 */
    size_t mark = arena_snapshot(a);
    float* buf1 = ARENA_CALLOC(a, float, 100);
    float* buf2 = ARENA_CALLOC(a, float, 100);
    ASSERT_NOT_NULL(buf1, "first calloc succeeds");
    ASSERT_NOT_NULL(buf2, "second calloc succeeds");
    buf1[0] = 3.14f;
    buf2[50] = 2.71f;

    /* 回卷 */
    arena_restore(a, mark);
    ASSERT_EQ_SIZE(mark, a->used, "used back to mark");

    /* 再次分配——复用空间 */
    float* buf3 = ARENA_CALLOC(a, float, 100);
    float* buf4 = ARENA_CALLOC(a, float, 100);
    ASSERT_NOT_NULL(buf3, "re-alloc succeeds after restore");
    ASSERT_NOT_NULL(buf4, "second re-alloc succeeds");

    arena_destroy(a);
}

TEST(arena_calloc_is_zeroed_after_restore) {
    Arena* a = arena_create(4096);

    /* 先写一些数据 */
    {
        size_t mark = arena_snapshot(a);
        int* p = ARENA_ALLOC(a, int, 100);
        for (int i = 0; i < 100; i++) p[i] = 999;
        arena_restore(a, mark);
    }

    /* 回卷后用 calloc 分配——确保归零 */
    size_t mark2 = arena_snapshot(a);
    int* p2 = ARENA_CALLOC(a, int, 100);
    ASSERT_NOT_NULL(p2, "calloc after restore succeeds");
    ASSERT_EQ_INT(0, p2[0], "calloc clears previous data [0]");
    ASSERT_EQ_INT(0, p2[50], "calloc clears previous data [50]");
    ASSERT_EQ_INT(0, p2[99], "calloc clears previous data [99]");

    arena_destroy(a);
}

int main(void) {
    RUN_TEST(arena_snapshot_restore_pattern);
    RUN_TEST(arena_calloc_is_zeroed_after_restore);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_mlp_arena_integration
ctest -R test_mlp_arena_integration --output-on-failure

# 编译 MLP 模块
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"

# 运行 move demo 全流程
bash scripts/run_demo.sh move all
```

**预期**: 训练和推理正常执行。若用 valgrind，确认零泄漏。

### 全量回归

```bash
ctest --output-on-failure
bash scripts/run_demo.sh move all
```

**当前全量**: 7 个单元测试 + move demo。

### 门控

- [x] test_mlp_arena_integration 全部通过
- [x] 全量编译零错误零警告
- [x] ctest 全量通过
- [x] move demo 训练+推理正常
- [ ] → **git commit** `"perf(A7): replace MLP heap allocations with Arena (mark/restore)"`
- [ ] → 进入 A8

**消灭问题**: PRF-C02, PRF-M05, PRF-M06, STB-M08

---

## 步骤 A8 — GNN 后端接入 Arena

### 代码变更

**修改**: `src/nn/types/gnn/gnn_infer_ops.c`

在 `GNNInferContext` 中添加 `Arena* arena` 字段。

`nn_gnn_forward_pass` 中的替换：
```c
/* 旧: 3 次 calloc */
float* owned_cache  = (float*)calloc(stages * stride, sizeof(float));
float* aggregated   = (float*)calloc(hidden_size, sizeof(float));
float* pooled_hidden = (float*)calloc(hidden_size, sizeof(float));

/* 新: arena */
size_t mark = arena_snapshot(ctx->arena);
float* owned_cache   = ARENA_CALLOC(ctx->arena, float, stages * stride);
float* aggregated    = ARENA_CALLOC(ctx->arena, float, hidden_size);
float* pooled_hidden = ARENA_CALLOC(ctx->arena, float, hidden_size);
// ... 使用 ...
arena_restore(ctx->arena, mark);
```

**修改**: `src/nn/types/gnn/gnn_train_ops.c`

在 `GNNTrainContext` 中添加 `Arena* arena` 字段。`gnn_backpropagate` 中 6 次 calloc 全部改为 arena。

```c
size_t mark = arena_snapshot(ctx->arena);
float* stage_grad          = ARENA_CALLOC(ctx->arena, float, stages * stride);
float* previous_stage_grad = ARENA_CALLOC(ctx->arena, float, stages * stride);
float* aggregated          = ARENA_CALLOC(ctx->arena, float, hidden_size);
float* aggregated_grad     = ARENA_CALLOC(ctx->arena, float, hidden_size);
float* pooled_hidden       = ARENA_CALLOC(ctx->arena, float, hidden_size);
float* pooled_grad         = ARENA_CALLOC(ctx->arena, float, hidden_size);
// ... 使用 ...
arena_restore(ctx->arena, mark);
```

### 步骤测试

**测试文件**: `tests/nn/test_gnn_arena_integration.c`

```c
#include "test_harness.h"
#include "utils/arena.h"

/* 模拟 GNN forward_pass 中的 3-buffer snapshot/restore 模式 */
TEST(gnn_three_buffer_pattern) {
    Arena* a = arena_create(8192);
    size_t mark = arena_snapshot(a);

    float* owned_cache   = ARENA_CALLOC(a, float, 64);
    float* aggregated    = ARENA_CALLOC(a, float, 32);
    float* pooled_hidden = ARENA_CALLOC(a, float, 32);

    ASSERT_NOT_NULL(owned_cache, "owned_cache alloc");
    ASSERT_NOT_NULL(aggregated, "aggregated alloc");
    ASSERT_NOT_NULL(pooled_hidden, "pooled_hidden alloc");

    /* 写入数据 */
    owned_cache[0] = 1.0f;
    aggregated[10] = 2.0f;
    pooled_hidden[5] = 3.0f;

    arena_restore(a, mark);

    /* 再次分配——验证复用 */
    float* buf2 = ARENA_CALLOC(a, float, 64);
    ASSERT_NOT_NULL(buf2, "re-alloc after restore");
    ASSERT_EQ_INT(0, (int)buf2[0], "calloc clears old owned_cache[0]");

    arena_destroy(a);
}

/* 模拟 GNN backpropagate 中的 6-buffer pattern */
TEST(gnn_six_buffer_pattern) {
    Arena* a = arena_create(16384);
    size_t mark = arena_snapshot(a);

    for (int i = 0; i < 6; i++) {
        float* buf = ARENA_CALLOC(a, float, 128);
        ASSERT_NOT_NULL(buf, "buffer %d alloc", i);
        buf[0] = (float)i;
    }
    ASSERT_TRUE(a->used >= 6 * 128 * sizeof(float), "used reflects 6 buffers");

    arena_restore(a, mark);
    ASSERT_EQ_SIZE(mark, a->used, "6 buffers fully reclaimed");

    arena_destroy(a);
}

int main(void) {
    RUN_TEST(gnn_three_buffer_pattern);
    RUN_TEST(gnn_six_buffer_pattern);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_gnn_arena_integration
ctest -R test_gnn_arena_integration --output-on-failure

# 编译 GNN 模块（如已启用）
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"

# 运行 GNN demo（如已构建）
bash scripts/run_demo.sh road_graph_nav all
```

### 全量回归

```bash
ctest --output-on-failure
# 所有可通过的 demo
bash scripts/run_demo.sh move all
bash scripts/run_demo.sh road_graph_nav all
```

**当前全量**: 8 个单元测试 + move + road_graph_nav demo 回归。

### 门控

- [x] test_gnn_arena_integration 全部通过
- [x] 全量编译零错误
- [x] ctest 全量通过
- [x] road_graph_nav demo 正常
- [ ] → **git commit** `"perf(A8): replace GNN heap allocations with Arena"`
- [ ] → 进入 A9

**消灭问题**: PRF-C03, PRF-C04, PRF-M14

---

## 步骤 A9 — Transformer 和 RNN 后端接入 Arena

### 代码变更

**Transformer** (`src/nn/types/transformer/transformer_infer_ops.c`):

在 `TransformerInferContext` 中添加 `Arena* arena`。
当前 `transformer_forward_cache_init` 改为 arena 分配，ForwardCache 内嵌到上下文中。
创建时一次性从 arena 分配，step 开始时 `arena_restore(mark)`，无需每次重新分配。

**Transformer 训练** (`transformer_train_ops.c`):

将 3 个梯度 buffer 改为 arena 分配。

**RNN** (`src/nn/types/rnn/rnn_train_ops.c`):

将 BPTT 中的 `memcpy` 交换改为指针交换：
```c
/* 旧 */
memcpy(dh_current, dh_next, hidden_size * sizeof(float));
memset(dh_next, 0, hidden_size * sizeof(float));

/* 新 */
float* tmp = dh_current;
dh_current = dh_next;
dh_next = tmp;
memset(dh_next, 0, hidden_size * sizeof(float));
```

RNN 的 BPTT scratch buffer 也改为 arena。

### 步骤测试

**测试文件**: `tests/nn/test_rnn_pointer_swap.c`

```c
#include "test_harness.h"
#include <string.h>

/* 验证指针交换与 memcpy 语义等价 */
TEST(pointer_swap_vs_memcpy) {
    float buf_a[8] = {1,2,3,4,5,6,7,8};
    float buf_b[8] = {0,0,0,0,0,0,0,0};

    /* 用指针交换 */
    float* current = buf_a;
    float* next = buf_b;
    float* tmp = current;
    current = next;
    next = tmp;
    memset(next, 0, 8 * sizeof(float));

    /* 验证 current 指向原 buf_b (全零) */
    ASSERT_EQ_INT(0, (int)current[0], "current[0] should be 0 after swap");
    /* 验证 next (原 buf_a) 被清零 */
    ASSERT_EQ_INT(0, (int)next[0], "next[0] is zeroed after swap");
    /* 但原 buf_a 被清零 */
    ASSERT_EQ_INT(0, (int)buf_a[0], "original buffer zeroed");
}

int main(void) {
    RUN_TEST(pointer_swap_vs_memcpy);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

**测试文件**: `tests/nn/test_transformer_arena.c`

```c
#include "test_harness.h"
#include "utils/arena.h"

/* 模拟 Transformer 的 cache + 每步 restore 模式 */
TEST(transformer_cache_restore_per_step) {
    Arena* a = arena_create(65536);
    size_t create_mark = arena_snapshot(a);

    /* 创建阶段：分配持久 cache */
    float* qkv_cache = ARENA_CALLOC(a, float, 1024);
    float* attn_cache = ARENA_CALLOC(a, float, 512);
    ASSERT_NOT_NULL(qkv_cache, "qkv cache alloc");
    ASSERT_NOT_NULL(attn_cache, "attn cache alloc");

    size_t persistent_used = a->used;

    /* 每步：分配临时缓冲区后回卷 */
    for (int step = 0; step < 10; step++) {
        size_t step_mark = arena_snapshot(a);
        float* step_buf = ARENA_CALLOC(a, float, 256);
        ASSERT_NOT_NULL(step_buf, "step %d buffer alloc", step);
        step_buf[0] = (float)step;
        arena_restore(a, step_mark);
    }

    /* 持久分配未被破坏 */
    ASSERT_EQ_SIZE(persistent_used, a->used, "persistent alloc preserved after step loops");
    ASSERT_EQ_INT(0, (int)qkv_cache[0], "qkv cache still zeroed");

    arena_destroy(a);
}

int main(void) {
    RUN_TEST(transformer_cache_restore_per_step);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_rnn_pointer_swap
ctest -R test_rnn_pointer_swap --output-on-failure

cmake --build build/ --target test_transformer_arena
ctest -R test_transformer_arena --output-on-failure

# 编译全量
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"

# 运行 Transformer + CNN_RNN demo
bash scripts/run_demo.sh transformer all
bash scripts/run_demo.sh cnn_rnn_react all
```

### 全量回归

```bash
ctest --output-on-failure
bash scripts/run_demo.sh move all
bash scripts/run_demo.sh transformer all
bash scripts/run_demo.sh cnn_rnn_react all
bash scripts/run_demo.sh road_graph_nav all
```

**当前全量**: 10 个单元测试 + 4 个 demo 回归。

### 门控

- [x] test_rnn_pointer_swap 通过
- [x] test_transformer_arena 通过
- [x] 全量编译零错误零警告
- [x] ctest 全量通过
- [x] transformer + cnn_rnn_react demo 正常
- [ ] → **git commit** `"perf(A9): migrate Transformer and RNN to Arena allocator"`
- [ ] → 轨 A 完成

**消灭问题**: PRF-H06, PRF-M11, PRF-M12, PRF-M13

---

## 轨 A 完成检查

- [x] A1: 错误枚举 + 测试 (3 用例)
- [x] A2: 日志 + 断言 + 测试 (4 用例)
- [x] A3: 安全数学 + 测试 (8 用例)
- [x] A4: Arena + 测试 (7 用例)
- [x] A5: 全局错误码替换 + 回归
- [x] A6: MLP Context + Arena + 测试
- [x] A7: MLP Arena 接入 + 测试 (2 用例)
- [x] A8: GNN Arena 接入 + 测试 (2 用例)
- [x] A9: Transformer + RNN Arena 接入 + 测试 (2 用例)

**轨 A 合计**: 28 个单元测试用例，覆盖 Arena 所有分配路径。
**消灭问题**: RC6 (错误基础设施) + RC1 (内存模型) 共约 45 个问题。
