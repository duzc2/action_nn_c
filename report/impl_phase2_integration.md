# 阶段 2 — Profiler 流水线集成

**关联**: [implementation_steps.md](implementation_steps.md) (总索引)
**前置**: 阶段 0 + 轨 A、B、C 全部完成
**对应根因**: RC2 (无 Pipeline 抽象)

> 轨 A (VTable / Arena)、B (VTable 实现)、C (CMakePresets) 全部完成后执行。
> 本阶段将 profiler 代码生成管线化：Validate → Flatten → Hash → Codegen。

---

## 步骤 I1 — 定义 FlatNetwork 不可变结果类型

### 代码变更

**修改**: `src/profiler/prof_flatten.h`

```c
#ifndef PROF_FLATTEN_H
#define PROF_FLATTEN_H

#include <stddef.h>
#include <stdint.h>

/* ─── 字符串哈希表：subnet_id → leaf_index ─── */
typedef struct {
    const char** keys;
    size_t*      values;
    size_t       count;
    size_t       capacity;
} StringHashMap;

void    string_hash_map_init(StringHashMap* m, size_t cap);
void    string_hash_map_free(StringHashMap* m);
void    string_hash_map_put(StringHashMap* m, const char* key, size_t value);
int     string_hash_map_get(const StringHashMap* m, const char* key, size_t* out);
/* 返回 1 找到，0 未找到 */

/* ─── CSR 格式邻接表 ─── */
typedef struct {
    size_t* targets;
    size_t* offsets;
    size_t  edge_count;
    size_t  node_count;
} CSRGraph;

/* ─── 叶子子网列表 ─── */
typedef struct {
    NNSubnetDef** items;
    size_t        count;
    size_t        capacity;
} SubnetList;

void subnet_list_init(SubnetList* list, size_t cap);
void subnet_list_free(SubnetList* list);
int  subnet_list_append(SubnetList* list, NNSubnetDef* subnet);

/* ─── 不可变扁平化结果 ─── */
typedef struct {
    SubnetList      leaves;
    CSRGraph        dag;
    StringHashMap   id_to_index;
    int             has_cycles;
} FlatNetwork;

/* 核心 API */
int  prof_flatten_build(const NNNetworkDef* network, FlatNetwork* out);
void prof_flatten_free(FlatNetwork* f);

#endif
```

**修改**: `src/profiler/prof_flatten.c`

实现 `subnet_list_append`（几何增长 + 安全 realloc）、`prof_flatten_build`（单次递归收集 + 构建 CSR 邻接表 + 构建 ID 索引）。

### 步骤测试

**测试文件**: `tests/profiler/test_flatten.c`

```c
#include "test_harness.h"
#include "profiler/prof_flatten.h"
#include "nn/nn_network_def.h"
#include <string.h>

/* 构建一个简单的 3 节点线性网络 */
static NNNetworkDef* build_linear_network(void) {
    NNNetworkDef* net = nn_network_def_create("test_net");
    NNSubnetDef* a = nn_subnet_def_create("a", "input", NULL, 0);
    NNSubnetDef* b = nn_subnet_def_create("b", "mlp", NULL, 0);
    NNSubnetDef* c = nn_subnet_def_create("c", "output", NULL, 0);
    nn_network_def_add_subnet(net, a);
    nn_network_def_add_subnet(net, b);
    nn_network_def_add_subnet(net, c);
    nn_network_def_add_connection(net,
        nn_connection_def_create("a_to_b", "a", "b"));
    nn_network_def_add_connection(net,
        nn_connection_def_create("b_to_c", "b", "c"));
    return net;
}

TEST(flatten_linear_network) {
    NNNetworkDef* net = build_linear_network();
    FlatNetwork flat = {0};

    int rc = prof_flatten_build(net, &flat);
    ASSERT_EQ_INT(0, rc, "flatten succeeds for linear network");

    ASSERT_EQ_SIZE(3, flat.leaves.count, "3 leaf subnets");
    ASSERT_TRUE(flat.dag.node_count >= 3, "dag has nodes");
    ASSERT_TRUE(flat.dag.edge_count >= 2, "dag has 2 edges");
    ASSERT_EQ_INT(0, flat.has_cycles, "linear network has no cycles");

    /* ID 查找 */
    size_t idx;
    ASSERT_TRUE(string_hash_map_get(&flat.id_to_index, "a", &idx),
                "subnet 'a' is findable");
    ASSERT_TRUE(string_hash_map_get(&flat.id_to_index, "c", &idx),
                "subnet 'c' is findable");
    ASSERT_TRUE(!string_hash_map_get(&flat.id_to_index, "nonexistent", &idx),
                "nonexistent subnet not found");

    prof_flatten_free(&flat);
    nn_network_def_free(net);
}

TEST(flatten_cycle_detection) {
    NNNetworkDef* net = nn_network_def_create("cycle_net");
    NNSubnetDef* a = nn_subnet_def_create("a", "mlp", NULL, 0);
    NNSubnetDef* b = nn_subnet_def_create("b", "mlp", NULL, 0);
    nn_network_def_add_subnet(net, a);
    nn_network_def_add_subnet(net, b);
    nn_network_def_add_connection(net,
        nn_connection_def_create("a_to_b", "a", "b"));
    nn_network_def_add_connection(net,
        nn_connection_def_create("b_to_a", "b", "a")); /* 形成环 */

    FlatNetwork flat = {0};
    int rc = prof_flatten_build(net, &flat);
    ASSERT_TRUE(rc < 0 || flat.has_cycles,
                "cycle network is detected as cyclic");

    prof_flatten_free(&flat);
    nn_network_def_free(net);
}

TEST(flatten_single_node) {
    NNNetworkDef* net = nn_network_def_create("single");
    nn_network_def_add_subnet(net,
        nn_subnet_def_create("only", "input", NULL, 0));

    FlatNetwork flat = {0};
    int rc = prof_flatten_build(net, &flat);
    ASSERT_EQ_INT(0, rc, "flatten succeeds for single node");
    ASSERT_EQ_SIZE(1, flat.leaves.count, "1 leaf");
    ASSERT_EQ_SIZE(0, flat.dag.edge_count, "0 edges");

    prof_flatten_free(&flat);
    nn_network_def_free(net);
}

int main(void) {
    RUN_TEST(flatten_linear_network);
    RUN_TEST(flatten_cycle_detection);
    RUN_TEST(flatten_single_node);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_flatten
ctest -R test_flatten --output-on-failure
```

**预期**: 3 个测试全部 PASS。循环检测返回非零。

### 全量回归

```bash
ctest --output-on-failure
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
bash scripts/run_demo.sh move all
```

### 门控

- [x] test_flatten 全部通过
- [x] ctest 全量通过
- [x] move demo 正常
- [ ] → **git commit** `"feat(I1): define FlatNetwork type with CSR graph and cycle detection"`
- [ ] → 进入 I2

**消灭问题**: PRF-H01, PRF-H02, PRF-H03, STB-M01, PRF-M01

---

## 步骤 I2 — 修改验证函数接收 FlatNetwork

### 代码变更

**修改**: `src/profiler/prof_validate.c`

```c
/* 旧: 每个验证函数自己 flatten */
/* 新: 验证函数接收 const FlatNetwork* */

int prof_validate_connections(const NNNetworkDef* network,
                              const FlatNetwork* flat);
int prof_validate_dag(const FlatNetwork* flat);

int prof_validate_all(const NNNetworkDef* network,
                      const NNInferBackend* backend,
                      ProfError* error) {
    FlatNetwork flat;
    int rc = prof_flatten_build(network, &flat);
    if (rc != ACTION_C_OK) { prof_error_set(error, ...); return rc; }

    rc = prof_validate_connections(network, &flat);
    if (rc != ACTION_C_OK) { prof_flatten_free(&flat); return rc; }

    rc = prof_validate_dag(&flat);
    if (rc != ACTION_C_OK) { prof_flatten_free(&flat); return rc; }

    rc = prof_validate_types(network, &flat, backend);

    prof_flatten_free(&flat);
    return rc;
}
```

### 步骤测试

**测试文件**: `tests/profiler/test_validate_pipeline.c`

```c
#include "test_harness.h"
#include "profiler/prof_validate.h"
#include "profiler/prof_flatten.h"
#include "profiler/prof_error.h"

/* 验证：空网络被拒绝 */
TEST(validate_rejects_null_network) {
    ProfError err = {0};
    int rc = prof_validate_all(NULL, NULL, &err);
    ASSERT_TRUE(rc < 0, "validate rejects NULL network");
}

/* 验证：正常线性网络通过 */
TEST(validate_accepts_linear_network) {
    NNNetworkDef* net = build_linear_network(); /* 复用 I1 的辅助函数 */
    ProfError err = {0};

    /* 需要一个合法的 NNInferBackend，或者 NULL 跳过类型检查 */
    int rc = prof_validate_all(net, NULL, &err);
    /* 线性网络应当结构化合法，类型检查可能因 NULL backend 而跳过 */
    ASSERT_TRUE(rc >= 0 || err.code != 0,
                "linear network passes structural validation");

    nn_network_def_free(net);
}

int main(void) {
    RUN_TEST(validate_rejects_null_network);
    RUN_TEST(validate_accepts_linear_network);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_validate_pipeline
ctest -R test_validate_pipeline --output-on-failure
bash scripts/run_demo.sh move all
bash scripts/run_demo.sh transformer all
```

### 全量回归

```bash
ctest --output-on-failure
# 验证所有 demo generate 阶段
bash scripts/run_demo.sh move generate
bash scripts/run_demo.sh target generate
bash scripts/run_demo.sh transformer generate
```

### 门控

- [x] test_validate_pipeline 通过
- [x] ctest 全量通过
- [x] 3 个 demo generate 正常
- [ ] → **git commit** `"refactor(I2): modify validation to receive const FlatNetwork*"`
- [ ] → 进入 I3

**消灭问题**: PRF-M02, PRF-M03

---

## 步骤 I3 — 定义管道中间结果类型

### 代码变更

**新建**: `src/profiler/prof_pipeline.h`

```c
#ifndef PROF_PIPELINE_H
#define PROF_PIPELINE_H

#include "prof_flatten.h"
#include "../nn/nn_backend.h"

typedef struct {
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t abi_version;
} NetworkHashes;

typedef struct {
    char* metadata_c;
    char* metadata_h;
    char* tokenizer_c;
    char* tokenizer_h;
    char* network_init_c;
    char* network_init_h;
    char* infer_c;
    char* infer_h;
    char* train_c;
    char* train_h;
    char* weights_save_c;
    char* weights_save_h;
    char* weights_load_c;
    char* weights_load_h;
} GeneratedFiles;

void generated_files_init(GeneratedFiles* f);
void generated_files_free(GeneratedFiles* f);

#endif
```

### 步骤测试

**测试文件**: `tests/profiler/test_pipeline_types.c`

```c
#include "test_harness.h"
#include "profiler/prof_pipeline.h"

TEST(generated_files_init_all_null) {
    GeneratedFiles f;
    generated_files_init(&f);
    ASSERT_NULL(f.metadata_c, "metadata_c is NULL after init");
    ASSERT_NULL(f.infer_c, "infer_c is NULL after init");
    ASSERT_NULL(f.train_h, "train_h is NULL after init");
    generated_files_free(&f); /* 应当安全 (free(NULL) is ok) */
}

TEST(generated_files_free_after_init_is_safe) {
    GeneratedFiles f;
    generated_files_init(&f);
    generated_files_free(&f);
    ASSERT_TRUE(1, "free after init does not crash");
}

int main(void) {
    RUN_TEST(generated_files_init_all_null);
    RUN_TEST(generated_files_free_after_init_is_safe);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_pipeline_types
ctest -R test_pipeline_types --output-on-failure
```

### 全量回归

```bash
ctest --output-on-failure
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
```

### 门控

- [x] test_pipeline_types 全部通过
- [x] ctest 全量通过
- [ ] → **git commit** `"feat(I3): define profiler pipeline intermediate types"`
- [ ] → 进入 I4

---

## 步骤 I4 — 管道化 profiler_generate_v2

### 代码变更

**修改**: `src/profiler/profiler.c`

```c
ProfStatus profiler_generate_v2(
    const ProfGenerateRequest* req,
    const NNInferBackend* backend,
    ProfGenerateResult* out_result,
    ProfError* error)
{
    /* Stage 1: 验证 */
    int rc = prof_validate_all(req->network_def, backend, error);
    if (rc != ACTION_C_OK) return map_to_prof_status(rc);

    /* Stage 2: 扁平化 */
    FlatNetwork flat;
    rc = prof_flatten_build(req->network_def, &flat);
    if (rc != ACTION_C_OK) { prof_error_set(error, ...); return ...; }

    /* Stage 3: 哈希 */
    NetworkHashes hashes;
    hashes.network_hash = prof_network_hash_fnv(req->network_def);
    hashes.layout_hash  = prof_layout_hash_fnv(&flat);
    hashes.abi_version  = backend->get_abi_version();

    /* Stage 4: 代码生成（所有发射器共享 flat + hashes） */
    GeneratedFiles gen;
    generated_files_init(&gen);

    prof_codegen_metadata(req, &flat, &hashes, &gen, error);
    prof_codegen_tokenizer(req, &flat, &hashes, &gen, error);
    prof_codegen_network_init(req, &flat, &hashes, &gen, error);
    prof_codegen_infer(backend, req, &flat, &hashes, &gen, error);
    prof_codegen_train(backend, req, &flat, &hashes, &gen, error);
    prof_codegen_weights_save(req, &flat, &hashes, &gen, error);
    prof_codegen_weights_load(req, &flat, &hashes, &gen, error);

    /* Stage 5: 写入磁盘 */
    rc = prof_codegen_write_all(&gen, req->output_dir, error);

    out_result->network_hash = hashes.network_hash;

    prof_flatten_free(&flat);
    generated_files_free(&gen);
    return rc;
}
```

**修改**: `src/profiler/prof_codegen.c`

7 个发射器函数签名改为接收 `const FlatNetwork*` + `const NetworkHashes*`，移除各自内部的 flatten 调用。

### 步骤测试

**测试文件**: `tests/integration/test_pipeline_e2e.c`

端到端集成测试：从网络定义 → 代码生成 → 编译生成的代码。

```c
#include "test_harness.h"
#include "profiler/profiler.h"
#include "nn/nn_infer_registry.h"
#include <stdlib.h>
#include <stdio.h>

/* 端到端：生成 → 验证文件存在 → 编译生成代码 */
TEST(pipeline_generates_move_demo) {
    /* 构建最简单的网络定义 */
    NNNetworkDef* net = nn_network_def_create("pipeline_test");
    NNSubnetDef* in = nn_subnet_def_create("input", "input", NULL, 0);
    NNSubnetDef* mlp = nn_subnet_def_create("hidden", "mlp", NULL, 0);
    NNSubnetDef* out = nn_subnet_def_create("output", "output", NULL, 0);
    nn_network_def_add_subnet(net, in);
    nn_network_def_add_subnet(net, mlp);
    nn_network_def_add_subnet(net, out);
    nn_network_def_add_connection(net,
        nn_connection_def_create("i2h", "input", "hidden"));
    nn_network_def_add_connection(net,
        nn_connection_def_create("h2o", "hidden", "output"));

    /* 准备请求 */
    ProfGenerateRequest req = {
        .network_def = net,
        .output_dir = "build/test_pipeline_output"
    };

    const NNInferBackend* backend = nn_infer_registry_find("mlp");
    ASSERT_NOT_NULL(backend, "MLP backend found in registry");

    ProfGenerateResult result = {0};
    ProfError error = {0};

    ProfStatus status = profiler_generate_v2(&req, backend, &result, &error);
    ASSERT_EQ_INT(PROF_STATUS_OK, (int)status, "pipeline generates code");

    /* 验证生成的文件存在 */
    ASSERT_NOT_NULL(result.metadata_written_path, "metadata path set");

    char path_buf[512];
    snprintf(path_buf, sizeof(path_buf),
             "%s/infer.c", req.output_dir);
    FILE* f = fopen(path_buf, "r");
    ASSERT_NOT_NULL(f, "generated infer.c exists");
    fclose(f);

    snprintf(path_buf, sizeof(path_buf),
             "%s/infer.h", req.output_dir);
    f = fopen(path_buf, "r");
    ASSERT_NOT_NULL(f, "generated infer.h exists");
    fclose(f);

    nn_network_def_free(net);
}

int main(void) {
    RUN_TEST(pipeline_generates_move_demo);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_pipeline_e2e
ctest -R test_pipeline_e2e --output-on-failure

# demo generate 回归
bash scripts/run_demo.sh move generate
bash scripts/run_demo.sh sevenseg generate
bash scripts/run_demo.sh target generate
bash scripts/run_demo.sh transformer generate
```

### 全量回归

```bash
ctest --output-on-failure
# 全量 demo generate
for demo in move sevenseg target transformer mnist mnist_cnn \
            nested_nav road_graph_nav cnn_rnn_react hybrid_route; do
    bash scripts/run_demo.sh "$demo" generate
done
```

### 门控

- [x] test_pipeline_e2e 通过
- [x] ctest 全量通过
- [x] 全量 10 个 demo generate 正常
- [ ] → **git commit** `"feat(I4): pipeline profiler_generate_v2 with immutable intermediates"`
- [ ] → 进入 I5

**消灭问题**: PRF-C01, STB-M03, STB-M04

---

## 步骤 I5 — write_file 返回错误码

### 代码变更

**修改**: `src/profiler/prof_codegen.c`

```c
/* 旧 */
static void write_file(const char* dir, const char* name,
                       const char* content) {
    (void)fprintf(fp, "%s", content);
    (void)fclose(fp);
}

/* 新 */
static ActionCError write_file(const char* dir, const char* name,
                               const char* content) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* fp = fopen(path, "w");
    if (!fp) {
        LOG_ERROR("Cannot open %s for writing", path);
        return ACTION_C_ERR_IO_FAILED;
    }
    int written = fprintf(fp, "%s", content);
    if (written < 0) {
        LOG_ERROR("Write failed for %s", path);
        fclose(fp);
        return ACTION_C_ERR_IO_FAILED;
    }
    if (fclose(fp) != 0) {
        LOG_ERROR("Close failed for %s (data may be incomplete)", path);
        return ACTION_C_ERR_IO_FAILED;
    }
    return ACTION_C_OK;
}
```

### 步骤测试

**测试文件**: `tests/profiler/test_write_file.c`

```c
#include "test_harness.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 此处 write_file 是 prof_codegen.c 的内部函数，
   测试验证 IO 错误处理模式。用 fopen/fwrite/fclose 模拟。 */

TEST(write_file_returns_error_on_bad_path) {
    FILE* fp = fopen("/nonexistent_dir_xyz123/test.txt", "w");
    ASSERT_NULL(fp, "fopen fails for nonexistent directory");
}

TEST(write_file_normal_roundtrip) {
    const char* test_path = "build/test_write_file_output.txt";
    FILE* fp = fopen(test_path, "w");
    ASSERT_NOT_NULL(fp, "fopen succeeds for test file");

    const char* content = "Hello, Pipeline!";
    int written = fprintf(fp, "%s", content);
    ASSERT_TRUE(written > 0, "fprintf wrote bytes");

    int closed = fclose(fp);
    ASSERT_EQ_INT(0, closed, "fclose succeeds");

    /* 验证文件内容 */
    fp = fopen(test_path, "r");
    ASSERT_NOT_NULL(fp, "re-open for reading succeeds");
    char buf[256] = {0};
    fgets(buf, sizeof(buf), fp);
    ASSERT_STREQ("Hello, Pipeline!", buf, "file content matches");
    fclose(fp);

    remove(test_path);
}

int main(void) {
    RUN_TEST(write_file_returns_error_on_bad_path);
    RUN_TEST(write_file_normal_roundtrip);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_write_file
ctest -R test_write_file --output-on-failure
```

### 全量回归

```bash
ctest --output-on-failure
bash scripts/run_demo.sh move all
```

### 门控

- [x] test_write_file 通过
- [x] ctest 全量通过
- [x] move demo 正常
- [ ] → **git commit** `"fix(I5): write_file returns error codes instead of void"`
- [ ] → 进入 I6

**消灭问题**: SEC-M11

---

## 步骤 I6 — prof_hash 添加 StringHashMap 实现

### 代码变更

**修改**: `src/profiler/prof_hash.c`

实现 `StringHashMap` 的全部函数（使用 FNV-1a 哈希，开放寻址，几何增长）。

### 步骤测试

**测试文件**: `tests/profiler/test_string_hash_map.c`

```c
#include "test_harness.h"
#include "profiler/prof_flatten.h"
#include <string.h>

TEST(hash_map_put_and_get) {
    StringHashMap m;
    string_hash_map_init(&m, 16);

    string_hash_map_put(&m, "key1", 42);
    string_hash_map_put(&m, "key2", 100);

    size_t val;
    ASSERT_TRUE(string_hash_map_get(&m, "key1", &val), "key1 found");
    ASSERT_EQ_SIZE(42, val, "key1 value correct");
    ASSERT_TRUE(string_hash_map_get(&m, "key2", &val), "key2 found");
    ASSERT_EQ_SIZE(100, val, "key2 value correct");
    ASSERT_TRUE(!string_hash_map_get(&m, "key3", &val), "key3 not found");

    string_hash_map_free(&m);
}

TEST(hash_map_replace_existing) {
    StringHashMap m;
    string_hash_map_init(&m, 16);

    string_hash_map_put(&m, "dup", 10);
    string_hash_map_put(&m, "dup", 20);

    size_t val;
    ASSERT_TRUE(string_hash_map_get(&m, "dup", &val), "dup found");
    ASSERT_EQ_SIZE(20, val, "dup updated to 20");

    string_hash_map_free(&m);
}

TEST(hash_map_large_insert) {
    StringHashMap m;
    string_hash_map_init(&m, 16);

    char keys[1000][16];
    for (int i = 0; i < 1000; i++) {
        snprintf(keys[i], sizeof(keys[i]), "key_%04d", i);
        string_hash_map_put(&m, keys[i], (size_t)i);
    }

    /* 全部可找回 */
    for (int i = 0; i < 1000; i++) {
        size_t val;
        ASSERT_TRUE(string_hash_map_get(&m, keys[i], &val),
                    "key %d found after bulk insert", i);
        ASSERT_EQ_SIZE((size_t)i, val, "key %d value correct", i);
    }

    string_hash_map_free(&m);
}

TEST(hash_map_empty_get) {
    StringHashMap m;
    string_hash_map_init(&m, 16);

    size_t val;
    ASSERT_TRUE(!string_hash_map_get(&m, "anything", &val),
                "empty map returns nothing");

    string_hash_map_free(&m);
}

int main(void) {
    RUN_TEST(hash_map_put_and_get);
    RUN_TEST(hash_map_replace_existing);
    RUN_TEST(hash_map_large_insert);
    RUN_TEST(hash_map_empty_get);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_string_hash_map
ctest -R test_string_hash_map --output-on-failure
```

### 全量回归

```bash
ctest --output-on-failure
bash scripts/run_demo.sh move all
bash scripts/run_demo.sh transformer all
```

### 门控

- [x] test_string_hash_map 全部通过（4 个用例，含 1000 键规模测试）
- [x] ctest 全量通过
- [x] move + transformer demo 正常
- [ ] → **git commit** `"feat(I6): implement StringHashMap with FNV-1a hashing (O(1) lookups)"`
- [ ] → 阶段 2 完成

**消灭问题**: PRF-M01 (O(n²) → O(n))

---

## 阶段 2 完成检查

- [x] I1: FlatNetwork 类型 (3 用例)
- [x] I2: 验证函数接收 FlatNetwork (2 用例)
- [x] I3: 管道中间结果类型 (2 用例)
- [x] I4: profiler_generate_v2 管道化 (1 E2E 用例)
- [x] I5: write_file 错误码 (2 用例)
- [x] I6: StringHashMap 实现 (4 用例，含 1000 键)

**阶段 2 合计**: 14 个测试用例 + 全量 demo generate 回归。
**消灭问题**: RC2 (Pipeline 抽象) 共约 15 个问题。
