#include "test_harness.h"
#include "nn/nn_infer_registry.h"
#include "nn/nn_backend.h"
#include "utils/arena.h"
#include <string.h>

/* ─── Dummy VTable backend for testing ─── */
static uint32_t dummy_abi(void) { return 1; }
static void*  dummy_create(const void* cfg, size_t sz, struct Arena* a) { (void)cfg; (void)sz; (void)a; return (void*)0x1; }
static void   dummy_destroy(void* ctx) { (void)ctx; }
static int    dummy_step(void* ctx) { (void)ctx; return 0; }
static int    dummy_get_output(const void* ctx, float* out, size_t sz) { (void)ctx; (void)out; (void)sz; return 0; }
static int    dummy_save_weights(const void* ctx, FILE* fp) { (void)ctx; (void)fp; return 0; }
static int    dummy_load_weights(void* ctx, FILE* fp) { (void)ctx; (void)fp; return 0; }
static uint64_t dum_hash(const void* ctx) { (void)ctx; return 0; }
static uint64_t dum_lhash(const void* ctx) { (void)ctx; return 0; }

static const NNInferBackend dummy_backend = {
    .type_name        = "dummy",
    .create           = dummy_create,
    .destroy          = dummy_destroy,
    .step             = dummy_step,
    .get_output       = dummy_get_output,
    .save_weights     = dummy_save_weights,
    .load_weights     = dummy_load_weights,
    .get_network_hash = dum_hash,
    .get_layout_hash  = dum_lhash,
    .get_abi_version  = dummy_abi,
};

/* Second dummy for duplicate/overflow testing. */
static const NNInferBackend dummy2_backend = {
    .type_name        = "dummy2",
    .create           = dummy_create,
    .destroy          = dummy_destroy,
    .step             = dummy_step,
    .get_output       = dummy_get_output,
    .save_weights     = dummy_save_weights,
    .load_weights     = dummy_load_weights,
    .get_network_hash = dum_hash,
    .get_layout_hash  = dum_lhash,
    .get_abi_version  = dummy_abi,
};

TEST(vtable_register_one) {
    int rc = nn_infer_vtable_register(&dummy_backend);
    ASSERT_EQ_INT(0, rc, "register succeeds");
    ASSERT_EQ_SIZE(1, nn_infer_vtable_count(), "count is 1");
}

TEST(vtable_find_existing) {
    const NNInferBackend* found = nn_infer_vtable_find("dummy");
    ASSERT_NOT_NULL(found, "find returns backend");
    ASSERT_STREQ("dummy", found->type_name, "type_name matches");
}

TEST(vtable_find_nonexistent) {
    const NNInferBackend* found = nn_infer_vtable_find("nonexistent");
    ASSERT_NULL(found, "find returns NULL for unknown type");
}

TEST(vtable_find_null) {
    const NNInferBackend* found = nn_infer_vtable_find(NULL);
    ASSERT_NULL(found, "find(NULL) returns NULL");
}

TEST(vtable_register_null) {
    int rc = nn_infer_vtable_register(NULL);
    ASSERT_TRUE(rc < 0, "register(NULL) fails");
}

TEST(vtable_register_null_name) {
    static const NNInferBackend no_name = { .type_name = NULL };
    int rc = nn_infer_vtable_register(&no_name);
    ASSERT_TRUE(rc < 0, "register(no_type_name) fails");
}

TEST(vtable_get_by_index) {
    const NNInferBackend* b = nn_infer_vtable_get(0);
    ASSERT_NOT_NULL(b, "get(0) returns backend");
    ASSERT_STREQ("dummy", b->type_name, "correct backend at index 0");
}

TEST(vtable_get_out_of_range) {
    const NNInferBackend* b = nn_infer_vtable_get(999);
    ASSERT_NULL(b, "get(999) returns NULL");
}

TEST(vtable_duplicate_replaces) {
    /* Re-registering "dummy" should replace and count stays at 1. */
    int rc = nn_infer_vtable_register(&dummy_backend);
    ASSERT_EQ_INT(0, rc, "re-register succeeds");
    ASSERT_EQ_SIZE(1, nn_infer_vtable_count(), "count still 1 after re-register");
}

int main(void) {
    RUN_TEST(vtable_register_one);
    RUN_TEST(vtable_find_existing);
    RUN_TEST(vtable_find_nonexistent);
    RUN_TEST(vtable_find_null);
    RUN_TEST(vtable_register_null);
    RUN_TEST(vtable_register_null_name);
    RUN_TEST(vtable_get_by_index);
    RUN_TEST(vtable_get_out_of_range);
    RUN_TEST(vtable_duplicate_replaces);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
