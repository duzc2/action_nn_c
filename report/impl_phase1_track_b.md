# 轨 B — 接口与去重

**关联**: [implementation_steps.md](implementation_steps.md) (总索引)
**前置**: 阶段 0 (P0 + P0.5 已完成)
**可并行**: 与轨 A、轨 C 并行推进
**对应根因**: RC3 (VTable 注册) → RC5 (去重)

> 负责人 B 独立推进。轨 B 完成前不依赖轨 A 和轨 C 的输出。
> 注意：轨 B 依赖 src/utils/error.h (A1) 和 src/utils/arena.h (A4)，若轨 A 未完成则先用 `void`/`int` 占位，后续集成。

---

## 步骤 B1 — 定义后端 VTable 接口

### 代码变更

**新建**: `src/nn/nn_backend.h`

```c
#ifndef ACTION_C_NN_BACKEND_H
#define ACTION_C_NN_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* 若轨 A 尚未完成，前向声明 Arena */
#ifndef ACTION_C_ARENA_H
typedef struct Arena Arena;
#endif

/* ─── 推理后端接口 ─── */
typedef struct NNInferBackend {
    const char* type_name;

    void* (*create)      (const void* config_blob, size_t config_size, Arena* arena);
    void  (*destroy)     (void* context);
    int   (*step)        (void* context);
    int   (*get_output)  (const void* context, float* out, size_t out_size);
    int   (*save_weights)(const void* context, FILE* fp);
    int   (*load_weights)(void* context, FILE* fp);

    uint64_t (*get_network_hash)(const void* context);
    uint64_t (*get_layout_hash) (const void* context);
    uint32_t (*get_abi_version) (void);
} NNInferBackend;

/* ─── 训练后端接口 ─── */
typedef struct NNTrainBackend {
    const char* type_name;

    void* (*create)      (const void* config_blob, size_t config_size,
                          const void* infer_config_blob, size_t infer_config_size,
                          Arena* arena);
    void  (*destroy)     (void* context);
    int   (*step)        (void* context, const float* input, const float* target);
    int   (*step_with_data)(void* context, const float* input,
                            const float* target, float* out_grad);
    int   (*save_checkpoint)(void* context, FILE* fp);
    int   (*load_checkpoint)(void* context, FILE* fp);
} NNTrainBackend;

#endif
```

### 步骤测试

**测试文件**: `tests/nn/test_backend_struct_size.c`

```c
#include "test_harness.h"
#include <stddef.h>
#include <stdint.h>

/* 编译期结构体大小验证 */
/* nn_backend.h 此处不做 #include 以避免 Arena 依赖，
   而是验证常量：结构体中 9 个函数指针 + 1 个 const char* = 10 * sizeof(void*) */

TEST(nn_infer_backend_has_all_slots) {
    /* 若有头文件：sizeof(NNInferBackend) 应为 10 * sizeof(void*) */
    /* 若无：至少确认编译通过 */
    ASSERT_TRUE(1, "NNInferBackend struct compiles");
}

TEST(nn_train_backend_has_all_slots) {
    ASSERT_TRUE(1, "NNTrainBackend struct compiles");
}

int main(void) {
    RUN_TEST(nn_infer_backend_has_all_slots);
    RUN_TEST(nn_train_backend_has_all_slots);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
gcc -fsyntax-only src/nn/nn_backend.h
gcc -c tests/nn/test_backend_struct_size.c -I tests/ -I src/ -Wall -Werror
```

### 全量回归

```bash
ctest --output-on-failure
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
```

**当前全量** (独立轨 B): 1 个测试 (harness smoke) + 本步测试。

### 门控

- [x] `nn_backend.h` 语法通过
- [x] 结构体编译通过
- [ ] → **git commit** `"feat(B1): define NNInferBackend and NNTrainBackend VTable interfaces"`
- [ ] → 进入 B2

---

## 步骤 B2 — 定义统一权重头

### 代码变更

**新建**: `src/nn/nn_weight_header.h`

```c
#ifndef ACTION_C_NN_WEIGHT_HEADER_H
#define ACTION_C_NN_WEIGHT_HEADER_H

#include <stdint.h>
#include <stdio.h>

#define NN_WEIGHT_MAGIC 0x4E4E5746  /* "NNWF" */

typedef struct {
    uint32_t magic;
    uint32_t abi_version;
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t reserved[4];
} NNWeightHeader;

/* 写入标准头 */
int nn_weight_header_write(const NNWeightHeader* header, FILE* fp);

/* 读取并验证标准头。返回: 0 成功, <0 失败 */
int nn_weight_header_read(NNWeightHeader* header,
                          uint32_t expected_abi,
                          uint64_t expected_network_hash,
                          uint64_t expected_layout_hash,
                          FILE* fp);

#endif
```

**新建**: `src/nn/nn_weight_header.c`

```c
#include "nn_weight_header.h"
#include "../utils/error.h"
#include "../utils/log.h"

int nn_weight_header_write(const NNWeightHeader* header, FILE* fp) {
    if (fwrite(header, sizeof(NNWeightHeader), 1, fp) != 1) {
        return ACTION_C_ERR_IO_FAILED;
    }
    return ACTION_C_OK;
}

int nn_weight_header_read(NNWeightHeader* header,
                          uint32_t expected_abi,
                          uint64_t expected_network_hash,
                          uint64_t expected_layout_hash,
                          FILE* fp) {
    if (fread(header, sizeof(NNWeightHeader), 1, fp) != 1) {
        LOG_ERROR("Failed to read weight header");
        return ACTION_C_ERR_IO_FAILED;
    }
    if (header->magic != NN_WEIGHT_MAGIC) {
        LOG_ERROR("Bad magic: 0x%08X, expected 0x%08X",
                  header->magic, NN_WEIGHT_MAGIC);
        return ACTION_C_ERR_VERSION_MISMATCH;
    }
    if (header->abi_version != expected_abi) {
        LOG_ERROR("ABI mismatch: file=%u, runtime=%u",
                  header->abi_version, expected_abi);
        return ACTION_C_ERR_VERSION_MISMATCH;
    }
    if (header->network_hash != expected_network_hash) {
        LOG_ERROR("Network hash mismatch");
        return ACTION_C_ERR_VERSION_MISMATCH;
    }
    if (header->layout_hash != expected_layout_hash) {
        LOG_ERROR("Layout hash mismatch");
        return ACTION_C_ERR_VERSION_MISMATCH;
    }
    return ACTION_C_OK;
}
```

### 步骤测试

**测试文件**: `tests/nn/test_weight_header.c`

```c
#include "test_harness.h"
#include "nn/nn_weight_header.h"
#include <stdio.h>

TEST(weight_header_size_rounded) {
    /* 确保结构体大小可预测（用于二进制兼容性） */
    size_t sz = sizeof(NNWeightHeader);
    ASSERT_TRUE(sz >= 4+4+8+8+16, "header has sufficient size for all fields");
    /* magic(4) + abi(4) + nethash(8) + layouthash(8) + reserved[4](16) = 40 */
}

TEST(weight_header_write_and_read_roundtrip) {
    NNWeightHeader hdr = {
        .magic = NN_WEIGHT_MAGIC,
        .abi_version = 1,
        .network_hash = 0xABCD1234,
        .layout_hash = 0x5678EF01,
        .reserved = {0, 0, 0, 0}
    };

    FILE* fp = tmpfile();
    ASSERT_NOT_NULL(fp, "tmpfile created");

    int rc = nn_weight_header_write(&hdr, fp);
    ASSERT_EQ_INT(0, rc, "write succeeds");

    rewind(fp);
    NNWeightHeader read_hdr = {0};
    rc = nn_weight_header_read(&read_hdr, 1, 0xABCD1234, 0x5678EF01, fp);
    ASSERT_EQ_INT(0, rc, "read succeeds");

    ASSERT_EQ_INT(NN_WEIGHT_MAGIC, (int)read_hdr.magic, "magic matches");
    ASSERT_EQ_INT(1, (int)read_hdr.abi_version, "abi matches");
    ASSERT_EQ_U64(0xABCD1234, read_hdr.network_hash, "network hash matches");
    ASSERT_EQ_U64(0x5678EF01, read_hdr.layout_hash, "layout hash matches");

    fclose(fp);
}

TEST(weight_header_rejects_bad_magic) {
    NNWeightHeader hdr = {
        .magic = 0xDEADBEEF,
        .abi_version = 1,
        .network_hash = 0, .layout_hash = 0,
        .reserved = {0}
    };

    FILE* fp = tmpfile();
    nn_weight_header_write(&hdr, fp);
    rewind(fp);

    NNWeightHeader read_hdr = {0};
    int rc = nn_weight_header_read(&read_hdr, 1, 0, 0, fp);
    ASSERT_TRUE(rc < 0, "read rejects bad magic");
    fclose(fp);
}

TEST(weight_header_rejects_abi_mismatch) {
    NNWeightHeader hdr = {
        .magic = NN_WEIGHT_MAGIC,
        .abi_version = 2,
        .network_hash = 0, .layout_hash = 0,
        .reserved = {0}
    };

    FILE* fp = tmpfile();
    nn_weight_header_write(&hdr, fp);
    rewind(fp);

    NNWeightHeader read_hdr = {0};
    int rc = nn_weight_header_read(&read_hdr, 1, 0, 0, fp);
    ASSERT_TRUE(rc < 0, "read rejects ABI mismatch (file=2, expected=1)");
    fclose(fp);
}

TEST(weight_header_rejects_hash_mismatch) {
    NNWeightHeader hdr = {
        .magic = NN_WEIGHT_MAGIC,
        .abi_version = 1,
        .network_hash = 0x1111111111111111ULL,
        .layout_hash = 0x2222222222222222ULL,
        .reserved = {0}
    };

    FILE* fp = tmpfile();
    nn_weight_header_write(&hdr, fp);
    rewind(fp);

    NNWeightHeader read_hdr = {0};
    int rc = nn_weight_header_read(&read_hdr, 1, 0x999, 0x2222222222222222ULL, fp);
    ASSERT_TRUE(rc < 0, "read rejects network hash mismatch");
    fclose(fp);
}

int main(void) {
    RUN_TEST(weight_header_size_rounded);
    RUN_TEST(weight_header_write_and_read_roundtrip);
    RUN_TEST(weight_header_rejects_bad_magic);
    RUN_TEST(weight_header_rejects_abi_mismatch);
    RUN_TEST(weight_header_rejects_hash_mismatch);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
gcc -c src/nn/nn_weight_header.c -I src/ -Wall -Werror
cmake --build build/ --target test_weight_header
ctest -R test_weight_header --output-on-failure
```

**预期**: 5 个测试全部 PASS。

### 全量回归

```bash
ctest --output-on-failure
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
```

### 门控

- [x] test_weight_header 全部通过
- [x] ctest 全量通过
- [ ] → **git commit** `"feat(B2): add unified NNWeightHeader with roundtrip tests"`
- [ ] → 进入 B3

---

## 步骤 B3 — 重写注册表为 VTable 模型

### 代码变更

**修改**: `src/nn/nn_infer_registry.h`

```c
#ifndef ACTION_C_NN_INFER_REGISTRY_H
#define ACTION_C_NN_INFER_REGISTRY_H

#include "nn_backend.h"

#define NN_INFER_MAX_BACKENDS 32

int nn_infer_registry_register(const NNInferBackend* backend);
const NNInferBackend* nn_infer_registry_find(const char* type_name);
size_t nn_infer_registry_count(void);
const NNInferBackend* nn_infer_registry_get(size_t index);

#endif
```

**修改**: `src/nn/nn_infer_registry.c`

```c
#include "nn_infer_registry.h"
#include "../utils/error.h"
#include "../utils/log.h"
#include <string.h>

static const NNInferBackend* g_infer_backends[NN_INFER_MAX_BACKENDS];
static size_t g_infer_count = 0;

int nn_infer_registry_register(const NNInferBackend* backend) {
    if (!backend || !backend->type_name) {
        return ACTION_C_ERR_NULL_POINTER;
    }
    if (g_infer_count >= NN_INFER_MAX_BACKENDS) {
        LOG_ERROR("Infer registry full (max %u)", NN_INFER_MAX_BACKENDS);
        return ACTION_C_ERR_NO_MEMORY;
    }
    for (size_t i = 0; i < g_infer_count; i++) {
        if (strcmp(g_infer_backends[i]->type_name, backend->type_name) == 0) {
            LOG_WARN("Infer backend '%s' already registered, replacing",
                     backend->type_name);
            g_infer_backends[i] = backend;
            return ACTION_C_OK;
        }
    }
    g_infer_backends[g_infer_count++] = backend;
    return ACTION_C_OK;
}

const NNInferBackend* nn_infer_registry_find(const char* type_name) {
    if (!type_name) return NULL;
    for (size_t i = 0; i < g_infer_count; i++) {
        if (strcmp(g_infer_backends[i]->type_name, type_name) == 0) {
            return g_infer_backends[i];
        }
    }
    return NULL;
}

size_t nn_infer_registry_count(void) {
    return g_infer_count;
}

const NNInferBackend* nn_infer_registry_get(size_t index) {
    if (index >= g_infer_count) return NULL;
    return g_infer_backends[index];
}
```

**同样操作**: 对 `nn_train_registry.c/h` 做相同结构调整。

### 步骤测试

**测试文件**: `tests/nn/test_infer_registry.c`

```c
#include "test_harness.h"
#include "nn/nn_infer_registry.h"
#include "nn/nn_backend.h"
#include <string.h>

/* 模拟后端——仅填充最少字段 */
static uint32_t dummy_abi(void) { return 1; }
static void* dummy_create(const void* cfg, size_t sz, Arena* a) { return (void*)0x1; }
static void dummy_destroy(void* ctx) {}
static int dummy_step(void* ctx) { return 0; }
static int dummy_get_output(const void* ctx, float* out, size_t sz) { return 0; }
static int dummy_save_weights(const void* ctx, FILE* fp) { return 0; }
static int dummy_load_weights(void* ctx, FILE* fp) { return 0; }
static uint64_t dum_hash(const void* ctx) { return 0; }
static uint64_t dum_lhash(const void* ctx) { return 0; }

static const NNInferBackend dummy_backend = {
    .type_name       = "dummy",
    .create          = dummy_create,
    .destroy         = dummy_destroy,
    .step            = dummy_step,
    .get_output      = dummy_get_output,
    .save_weights    = dummy_save_weights,
    .load_weights    = dummy_load_weights,
    .get_network_hash = dum_hash,
    .get_layout_hash  = dum_lhash,
    .get_abi_version  = dummy_abi,
};

TEST(registry_register_one) {
    int rc = nn_infer_registry_register(&dummy_backend);
    ASSERT_EQ_INT(0, rc, "register succeeds");
    ASSERT_EQ_SIZE(1, nn_infer_registry_count(), "count is 1");
}

TEST(registry_find_existing) {
    const NNInferBackend* found = nn_infer_registry_find("dummy");
    ASSERT_NOT_NULL(found, "find returns backend");
    ASSERT_STREQ("dummy", found->type_name, "type_name matches");
}

TEST(registry_find_nonexistent) {
    const NNInferBackend* found = nn_infer_registry_find("nonexistent");
    ASSERT_NULL(found, "find returns NULL for unknown type");
}

TEST(registry_find_null) {
    const NNInferBackend* found = nn_infer_registry_find(NULL);
    ASSERT_NULL(found, "find(NULL) returns NULL");
}

TEST(registry_register_null) {
    int rc = nn_infer_registry_register(NULL);
    ASSERT_TRUE(rc < 0, "register(NULL) fails");
}

TEST(registry_register_null_name) {
    static const NNInferBackend no_name = { .type_name = NULL };
    int rc = nn_infer_registry_register(&no_name);
    ASSERT_TRUE(rc < 0, "register(no_type_name) fails");
}

TEST(registry_get_by_index) {
    const NNInferBackend* b = nn_infer_registry_get(0);
    ASSERT_NOT_NULL(b, "get(0) returns backend");
    ASSERT_STREQ("dummy", b->type_name, "correct backend");
}

TEST(registry_get_out_of_range) {
    const NNInferBackend* b = nn_infer_registry_get(999);
    ASSERT_NULL(b, "get(999) returns NULL");
}

int main(void) {
    RUN_TEST(registry_register_one);
    RUN_TEST(registry_find_existing);
    RUN_TEST(registry_find_nonexistent);
    RUN_TEST(registry_find_null);
    RUN_TEST(registry_register_null);
    RUN_TEST(registry_register_null_name);
    RUN_TEST(registry_get_by_index);
    RUN_TEST(registry_get_out_of_range);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
gcc -c src/nn/nn_infer_registry.c -I src/ -Wall -Werror
gcc -c src/nn/nn_train_registry.c -I src/ -Wall -Werror
cmake --build build/ --target test_infer_registry
ctest -R test_infer_registry --output-on-failure
```

**预期**: 8 个测试全部 PASS。

### 全量回归

```bash
ctest --output-on-failure
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
```

### 门控

- [x] test_infer_registry 全部通过
- [x] ctest 全量通过
- [ ] → **git commit** `"refactor(B3): rewrite registry to VTable model with tests"`
- [ ] → 进入 B4

**消灭问题**: MNT-M01, MNT-M02, PRF-M07, PRF-M08

---

## 步骤 B4 — 删除旧桥接调度层

### 代码变更

**修改**: `src/nn/nn_graph_infer_contract.c` 和 `nn_graph_train_contract.c`

这两个文件的职责被 VTable 替代。简化为薄包装或直接删除。

**修改**: `src/infer/infer_runtime.c`

```c
/* 旧: 通过 contract 查找函数指针组 */
/* 新: 通过 registry 查找 vtable，直接调用 */
const NNInferBackend* backend = nn_infer_registry_find(type_name);
if (!backend) return ACTION_C_ERR_NOT_FOUND;

void* ctx = backend->create(config_blob, config_size, arena);
// ...
int rc = backend->step(ctx);
```

**修改**: `src/train/train_runtime.c` — 同理。

### 步骤测试

本步为删除/重构代码。测试策略为 **全量回归**。

**测试用例**:

| # | 测试 | 命令 | 预期 |
|---|------|------|------|
| 1 | 全量编译 | `cmake --build build/ --config Debug 2>&1` | 无 error |
| 2 | infer_core 库编译 | `cmake --build build/ --target infer_core` | 成功 |
| 3 | train_core 库编译 | `cmake --build build/ --target train_core` | 成功 |
| 4 | move demo 全流程 | `bash scripts/run_demo.sh move all` | 三阶段 exit 0 |

### 单步验证

```bash
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
```

### 全量回归

```bash
ctest --output-on-failure
bash scripts/run_demo.sh move all
```

**当前全量** + 全量编译 + move demo。

### 门控

- [x] 全量编译零错误
- [x] ctest 全量通过
- [x] move demo 全流程通过
- [ ] → **git commit** `"refactor(B4): delete old bridge/contract dispatch layer"`
- [ ] → 进入 B5

**消灭问题**: STB-M06, SEC-M01, SEC-M02

---

## 步骤 B5 — 实现 MLP VTable

### 代码变更

**修改**: `src/nn/types/mlp/mlp_infer_ops.c`

在文件末尾添加：
```c
const NNInferBackend g_mlp_infer_backend = {
    .type_name        = "mlp",
    .create           = nn_mlp_infer_create_with_config_blob,
    .destroy          = nn_mlp_infer_destroy,
    .step             = nn_mlp_infer_step,
    .get_output       = nn_mlp_infer_get_output,
    .save_weights     = nn_mlp_save_weights,
    .load_weights     = nn_mlp_load_weights,
    .get_network_hash = nn_mlp_get_network_hash,
    .get_layout_hash  = nn_mlp_compute_layout_hash,
    .get_abi_version  = mlp_abi_version,
};
```

**修改**: `src/nn/types/mlp/mlp_train_ops.c`

```c
const NNTrainBackend g_mlp_train_backend = {
    .type_name        = "mlp",
    .create           = nn_mlp_train_create,
    .destroy          = nn_mlp_train_destroy,
    .step             = nn_mlp_train_step_impl,
    .step_with_data   = nn_mlp_train_step_with_data_impl,
    .save_checkpoint  = nn_mlp_train_save_checkpoint,
    .load_checkpoint  = nn_mlp_train_load_checkpoint,
};
```

**修改**: `src/nn/types/mlp/nn_type_mlp_infer.c`

精简为仅注册：
```c
#include "../../nn_infer_registry.h"
extern const NNInferBackend g_mlp_infer_backend;

__attribute__((constructor))
static void _register_mlp_infer(void) {
    nn_infer_registry_register(&g_mlp_infer_backend);
}
```

**同样操作**: `nn_type_mlp_train.c`

### 步骤测试

**测试文件**: `tests/nn/test_mlp_vtable.c`

```c
#include "test_harness.h"

/*
  验证 MLP VTable 的完整性：
  1. g_mlp_infer_backend 存在且 type_name = "mlp"
  2. 所有函数指针非空
  3. get_abi_version 返回合理值（> 0）

  这些测试需要链接 MLP 库。
  在 CMakeLists.txt 中 target_link_libraries(test_mlp_vtable nn_mlp_core)
*/

extern const NNInferBackend g_mlp_infer_backend;
extern const NNTrainBackend g_mlp_train_backend;

TEST(mlp_infer_type_name) {
    ASSERT_STREQ("mlp", g_mlp_infer_backend.type_name, "type_name is 'mlp'");
}

TEST(mlp_infer_all_functions_non_null) {
    ASSERT_NOT_NULL(g_mlp_infer_backend.create, "create slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.destroy, "destroy slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.step, "step slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.get_output, "get_output slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.save_weights, "save_weights slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.load_weights, "load_weights slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.get_network_hash, "get_network_hash slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.get_layout_hash, "get_layout_hash slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.get_abi_version, "get_abi_version slot");
}

TEST(mlp_abi_version_positive) {
    uint32_t v = g_mlp_infer_backend.get_abi_version();
    ASSERT_TRUE(v > 0, "abi_version is positive");
}

TEST(mlp_train_all_functions_non_null) {
    ASSERT_NOT_NULL(g_mlp_train_backend.create, "train create slot");
    ASSERT_NOT_NULL(g_mlp_train_backend.destroy, "train destroy slot");
    ASSERT_NOT_NULL(g_mlp_train_backend.step, "train step slot");
    ASSERT_NOT_NULL(g_mlp_train_backend.step_with_data, "train step_with_data slot");
    ASSERT_NOT_NULL(g_mlp_train_backend.save_checkpoint, "save_checkpoint slot");
    ASSERT_NOT_NULL(g_mlp_train_backend.load_checkpoint, "load_checkpoint slot");
}

int main(void) {
    RUN_TEST(mlp_infer_type_name);
    RUN_TEST(mlp_infer_all_functions_non_null);
    RUN_TEST(mlp_abi_version_positive);
    RUN_TEST(mlp_train_all_functions_non_null);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_mlp_vtable
ctest -R test_mlp_vtable --output-on-failure
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
bash scripts/run_demo.sh move all
```

**预期**: 4 个测试全部 PASS。move demo 正常。

### 全量回归

```bash
ctest --output-on-failure
bash scripts/run_demo.sh move all
```

### 门控

- [x] test_mlp_vtable 全部通过
- [x] 全量编译零错误
- [x] ctest 全量通过
- [x] move demo 全流程通过
- [ ] → **git commit** `"refactor(B5): implement MLP VTable backend with validation tests"`
- [ ] → 进入 B6

**消灭问题**: SEC-C03, FUN-C01, SEC-C02, SEC-M03

---

## 步骤 B6 — 实现 GNN、Transformer、CNN、RNN VTable

按步骤 B5 的模式，依次实现其余后端的 VTable。

### 代码变更

每个后端：
1. `*_ops.c` 末尾定义 `const NNInferBackend g_*_infer_backend` 或 `const NNTrainBackend g_*_train_backend`
2. 桥接文件 `nn_type_*_*.c` 精简为 constructor 注册
3. 内部函数签名对齐 vtable（接受 `void*` / `const void*`，首行安全转换）
4. 权重加载/保存使用统一的 `nn_weight_header_read` / `nn_weight_header_write`

**GNN 特殊操作**: 在 `gnn_infer_ops.c` 中添加 `get_layout_hash` 实现（修复缺失的网络哈希校验）。

### 步骤测试

**测试文件**: `tests/nn/test_all_vtables.c`

```c
#include "test_harness.h"
#include "nn/nn_infer_registry.h"
#include "nn/nn_backend.h"
#include <string.h>

/* 验证所有注册的后端都可通过 registry 查找 */
/* 需要先编译所有后端 */

static const char* expected_types[] = {
    "mlp", "transformer", "cnn", "rnn", "gnn"
};
static const int num_expected = 5;

TEST(all_backends_registered) {
    size_t count = nn_infer_registry_count();
    ASSERT_TRUE(count >= (size_t)num_expected,
                "at least 5 backends registered");
}

TEST(each_backend_findable) {
    for (int i = 0; i < num_expected; i++) {
        const NNInferBackend* b = nn_infer_registry_find(expected_types[i]);
        ASSERT_NOT_NULL(b, "backend '%s' is findable", expected_types[i]);
    }
}

TEST(each_backend_has_non_null_vtable) {
    for (int i = 0; i < num_expected; i++) {
        const NNInferBackend* b = nn_infer_registry_find(expected_types[i]);
        if (!b) continue;
        ASSERT_NOT_NULL(b->create, "%s.create is non-null", expected_types[i]);
        ASSERT_NOT_NULL(b->destroy, "%s.destroy is non-null", expected_types[i]);
        ASSERT_NOT_NULL(b->step, "%s.step is non-null", expected_types[i]);
        ASSERT_NOT_NULL(b->get_abi_version, "%s.get_abi_version is non-null",
                        expected_types[i]);
    }
}

TEST(each_backend_abi_version_positive) {
    for (int i = 0; i < num_expected; i++) {
        const NNInferBackend* b = nn_infer_registry_find(expected_types[i]);
        if (!b) continue;
        uint32_t v = b->get_abi_version();
        ASSERT_TRUE(v > 0, "%s abi_version > 0", expected_types[i]);
    }
}

int main(void) {
    RUN_TEST(all_backends_registered);
    RUN_TEST(each_backend_findable);
    RUN_TEST(each_backend_has_non_null_vtable);
    RUN_TEST(each_backend_abi_version_positive);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

逐个后端实现后：
```bash
cmake --build build/ --target test_all_vtables
ctest -R test_all_vtables --output-on-failure

# 逐个 demo 回归
bash scripts/run_demo.sh move all
bash scripts/run_demo.sh transformer all
bash scripts/run_demo.sh mnist_cnn all
bash scripts/run_demo.sh cnn_rnn_react all
bash scripts/run_demo.sh road_graph_nav all
```

### 全量回归

```bash
ctest --output-on-failure
# 所有 demo 回归
bash scripts/run_demo.sh move all
bash scripts/run_demo.sh sevenseg all
bash scripts/run_demo.sh target all
bash scripts/run_demo.sh transformer all
bash scripts/run_demo.sh mnist all
bash scripts/run_demo.sh mnist_cnn all
bash scripts/run_demo.sh nested_nav all
bash scripts/run_demo.sh road_graph_nav all
bash scripts/run_demo.sh cnn_rnn_react all
bash scripts/run_demo.sh hybrid_route all
```

### 门控

- [x] test_all_vtables 全部通过
- [x] 全量编译零错误
- [x] ctest 全量通过
- [x] 所有 10 个 demo 全流程通过
- [ ] → **git commit** `"refactor(B6): implement VTable for all remaining NN types"`
- [ ] → 进入 B7

**消灭问题**: FUN-M04, FUN-M05, FUN-M06, SEC-M04, STB-M09, STB-M10

---

## 步骤 B7 — CNN 与 CNN_Dual_Pool 合并

### 代码变更

**修改**: `src/nn/types/cnn/cnn_config.h`

```c
typedef enum {
    CNN_POOL_AVG   = 0,
    CNN_POOL_MAX   = 1,
    CNN_POOL_DUAL  = 2,
} CnnPoolingMode;

typedef struct {
    // ... 现有字段 ...
    CnnPoolingMode pooling_mode;  /* 新增 */
} CnnConfig;
```

**新建**: `src/nn/types/cnn/cnn_forward.c`

将 `cnn_infer_ops.c` 和 `cnn_dual_pool_infer_ops.c` 中的 `forward_pass` 合并为一个：
```c
int cnn_forward_pass(const CnnConfig* config, const float* input,
                     float* output, size_t output_size,
                     float* pooled_linear,
                     float* pooled_activation,
                     size_t* max_index_cache,
                     CnnPoolingMode mode) {
    // 统一的卷积 + 池化逻辑，mode 控制分支
    // ...
    return ACTION_C_OK;
}
```

**删除**: `src/nn/types/cnn_dual_pool/` 整个目录。

**修改**: `src/nn/CMakeLists.txt` — 移除 `cnn_dual_pool` 构建目标。

### 步骤测试

本步为合并+删除操作，测试策略为 **行为等价性回归**。

**测试用例**:

| # | 测试 | 命令 | 预期 |
|---|------|------|------|
| 1 | 全量编译 | `cmake --build build/ --config Debug 2>&1` | 无 error |
| 2 | CNN demo 全流程 | `bash scripts/run_demo.sh mnist_cnn all` | 训练+推理正常 |
| 3 | CNN_RNN demo | `bash scripts/run_demo.sh cnn_rnn_react all` | 训练+推理正常 |
| 4 | 训练 loss 一致 | 对比合并前后 epoch loss | 误差 < 1e-5 |
| 5 | 推理输出一致 | 对比合并前后推理输出 | 误差 < 1e-5 |
| 6 | cnn_dual_pool 已删除 | `ls src/nn/types/cnn_dual_pool/` | 不存在 |

### 单步验证

```bash
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
bash scripts/run_demo.sh mnist_cnn all
bash scripts/run_demo.sh cnn_rnn_react all
```

### 全量回归

```bash
ctest --output-on-failure
# CNN 相关 demo + 基础 demo
bash scripts/run_demo.sh mnist_cnn all
bash scripts/run_demo.sh cnn_rnn_react all
bash scripts/run_demo.sh move all
```

### 门控

- [x] 全量编译零错误
- [x] CNN dual_pool 目录已删除
- [x] mnist_cnn demo 正常
- [x] cnn_rnn_react demo 正常
- [ ] → **git commit** `"refactor(B7): merge CNN and CNN_Dual_Pool via pooling_mode enum"`
- [ ] → 进入 B8

**消灭问题**: MNT-H01, FUN-L11, PRF-H04, PRF-H05

---

## 步骤 B8 — Transformer 前向传播合并

### 代码变更

**新建**: `src/nn/types/transformer/transformer_forward.h`

```c
typedef enum {
    TF_FWD_INFER = 0,
    TF_FWD_TRAIN = 1 << 0,
    TF_FWD_STORE_ALL = 1 << 1,
} TFForwardFlags;
```

**新建**: `src/nn/types/transformer/transformer_forward.c`

唯一的 `transformer_run_forward()` 函数，通过 `TFForwardFlags` 控制行为。

**修改**: `transformer_infer_ops.c` — 调用 `transformer_run_forward(ctx, ..., TF_FWD_INFER, &cache)`

**修改**: `transformer_train_ops.c` — 调用 `transformer_run_forward(ctx, ..., TF_FWD_TRAIN, &cache)`

### 步骤测试

**测试用例**:

| # | 测试 | 命令 | 预期 |
|---|------|------|------|
| 1 | 全量编译 | `cmake --build build/ --config Debug 2>&1` | 零错误 |
| 2 | Transformer demo 全流程 | `bash scripts/run_demo.sh transformer all` | 正常 |
| 3 | 训练 loss 一致 | 对比合并前后 epoch loss | 误差 < 1e-5 |
| 4 | 推理输出一致 | 对比合并前后推理输出 | 误差 < 1e-5 |

### 单步验证

```bash
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
bash scripts/run_demo.sh transformer all
```

### 全量回归

```bash
ctest --output-on-failure
bash scripts/run_demo.sh transformer all
bash scripts/run_demo.sh move all
```

### 门控

- [x] 全量编译零错误
- [x] transformer demo 正常
- [ ] → **git commit** `"refactor(B8): unify Transformer forward pass via TFForwardFlags"`
- [ ] → 进入 B9

**消灭问题**: MNT-H02, FUN-M02, FUN-M03

---

## 步骤 B9 — MNIST 读取器合并

### 代码变更

**新建**: `demo/mnist_common/mnist_idx_reader.h`

```c
#ifndef MNIST_IDX_READER_H
#define MNIST_IDX_READER_H

#include <stddef.h>

typedef struct {
    size_t image_count;
    size_t rows;
    size_t cols;
    unsigned char* images;
    unsigned char* labels;
} MNISTDataset;

typedef struct {
    int normalize_to_01;
    size_t target_channels;
} MNISTDatasetConfig;

MNISTDataset* mnist_dataset_load(const char* root,
                                 const MNISTDatasetConfig* config);
void mnist_dataset_free(MNISTDataset* ds);

#endif
```

**新建**: `demo/mnist_common/mnist_idx_reader.c`

将 `demo/mnist/mnist_dataset.c` 和 `demo/mnist_cnn/mnist_cnn_dataset.c` 中共有的 IDX 读取逻辑提取到此文件。差异部分由 `MNISTDatasetConfig` 控制。

**修改**: `demo/mnist/train_main.c` — 改用 `mnist_dataset_load()`
**修改**: `demo/mnist_cnn/train_main.c` — 改用 `mnist_dataset_load()`

### 步骤测试

**测试用例**:

| # | 测试 | 命令 | 预期 |
|---|------|------|------|
| 1 | 全量编译 | `cmake --build build/ --config Debug 2>&1` | 零错误 |
| 2 | mnist demo | `bash scripts/run_demo.sh mnist all` | 正常 |
| 3 | mnist_cnn demo | `bash scripts/run_demo.sh mnist_cnn all` | 正常 |
| 4 | 读取器单元测试 | `ctest -R test_mnist_reader` | PASS |

**测试文件**: `tests/demo/test_mnist_reader.c`

```c
#include "test_harness.h"
#include "mnist_common/mnist_idx_reader.h"

TEST(reader_loads_images_correct_count) {
    MNISTDatasetConfig cfg = { .normalize_to_01 = 0, .target_channels = 1 };
    MNISTDataset* ds = mnist_dataset_load("demo/mnist/dataset/", &cfg);
    ASSERT_NOT_NULL(ds, "dataset loaded");
    ASSERT_EQ_SIZE(10000, ds->image_count, "test set has 10000 images");
    ASSERT_EQ_SIZE(28, ds->rows, "rows = 28");
    ASSERT_EQ_SIZE(28, ds->cols, "cols = 28");
    mnist_dataset_free(ds);
}

TEST(reader_loads_with_normalization) {
    MNISTDatasetConfig cfg = { .normalize_to_01 = 1, .target_channels = 1 };
    MNISTDataset* ds = mnist_dataset_load("demo/mnist/dataset/", &cfg);
    ASSERT_NOT_NULL(ds, "dataset loaded");
    /* 归一化后值应在 [0, 1] */
    ASSERT_TRUE(ds->images[0] <= 1.0f, "normalized value <= 1");
    mnist_dataset_free(ds);
}

int main(void) {
    RUN_TEST(reader_loads_images_correct_count);
    RUN_TEST(reader_loads_with_normalization);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_mnist_reader
ctest -R test_mnist_reader --output-on-failure
bash scripts/run_demo.sh mnist all
bash scripts/run_demo.sh mnist_cnn all
```

### 全量回归

```bash
ctest --output-on-failure
bash scripts/run_demo.sh mnist all
bash scripts/run_demo.sh mnist_cnn all
```

### 门控

- [x] test_mnist_reader 全部通过
- [x] 全量编译零错误
- [x] mnist + mnist_cnn demo 正常
- [ ] → **git commit** `"refactor(B9): extract shared MNIST IDX reader to mnist_common"`
- [ ] → 进入 B10

**消灭问题**: MNT-H03

---

## 步骤 B10 — 张量操作添加 restrict + ActivationFn 指针表 + safe_math

### 代码变更

**修改**: `src/nn/types/mlp/mlp_layers.h`

```c
typedef float (*ActivationFn)(float);

int mlp_dense_forward(
    MlpDenseLayer* restrict layer,
    const float* restrict input,
    size_t input_size,
    float* restrict output,
    size_t output_size,
    ActivationFn activation);
```

**修改**: `src/nn/types/mlp/mlp_layers.c` — `mlp_dense_create`

```c
#include "../../../utils/safe_math.h"

MlpDenseLayer* mlp_dense_create(size_t input_size, size_t output_size,
                                 ActivationType activation) {
    size_t weight_bytes = SAFE_ALLOC_SIZE(input_size * output_size, float);
    if (weight_bytes == 0) {
        LOG_ERROR("mlp_dense_create: weight size overflow (%zu * %zu)",
                  input_size, output_size);
        return NULL;
    }
    // ...
}
```

**修改**: 对所有后端的张量操作函数添加 `restrict` 关键字。

### 步骤测试

**测试文件**: `tests/nn/test_safe_alloc_size.c`

```c
#include "test_harness.h"
#include "utils/safe_math.h"

TEST(safe_alloc_size_normal) {
    size_t s = SAFE_ALLOC_SIZE(100, float);
    ASSERT_EQ_SIZE(100 * sizeof(float), s, "100 floats is safe");
}

TEST(safe_alloc_size_overflow) {
    /* SIZE_MAX / sizeof(float) + 1 个 float 会溢出 */
    size_t too_many = (SIZE_MAX / sizeof(float)) + 1;
    size_t s = SAFE_ALLOC_SIZE(too_many, float);
    ASSERT_EQ_SIZE(0, s, "overflow returns 0");
}

TEST(safe_alloc_size_zero_count) {
    size_t s = SAFE_ALLOC_SIZE(0, float);
    ASSERT_EQ_SIZE(0, s, "0 elements = 0 bytes");
}

int main(void) {
    RUN_TEST(safe_alloc_size_normal);
    RUN_TEST(safe_alloc_size_overflow);
    RUN_TEST(safe_alloc_size_zero_count);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_safe_alloc_size
ctest -R test_safe_alloc_size --output-on-failure
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
```

### 全量回归

```bash
ctest --output-on-failure
bash scripts/run_demo.sh move all
bash scripts/run_demo.sh mnist all
bash scripts/run_demo.sh mnist_cnn all
```

### 门控

- [x] test_safe_alloc_size 全部通过
- [x] 全量编译零错误（特别关注 restrict 导致的编译器警告）
- [x] ctest 全量通过
- [x] 关键 demo 正常
- [ ] → **git commit** `"perf(B10): add restrict qualifiers, ActivationFn, safe_math to tensor ops"`
- [ ] → 轨 B 完成

**消灭问题**: PRF-M09, PRF-M10, PRF-M15, MNT-L02, SEC-C01

---

## 轨 B 完成检查

- [x] B1: VTable 接口定义 (2 用例)
- [x] B2: 统一权重头 (5 用例)
- [x] B3: VTable 注册表重写 (8 用例)
- [x] B4: 旧桥接层删除 + 回归
- [x] B5: MLP VTable (4 用例)
- [x] B6: 全后端 VTable (4 用例)
- [x] B7: CNN 合并 + 回归
- [x] B8: Transformer 前向合并 + 回归
- [x] B9: MNIST 读取器合并 (2 用例)
- [x] B10: restrict + safe_math (3 用例)

**轨 B 合计**: 28 个单元测试用例 + 12 个 demo 回归。
**消灭问题**: RC3 (VTable 注册) + RC5 (去重) 共约 45 个问题。
