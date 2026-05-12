#include "test_harness.h"
#include "utils/arena.h"

/* Simulate Transformer's: alloc persistent cache once, then per-step temp + restore. */
TEST(transformer_cache_restore_per_step) {
    Arena* a = arena_create(65536);
    ASSERT_NOT_NULL(a, "arena_create succeeds");

    /* Creation: allocate persistent cache buffers (QKV, attention, etc.) */
    (void)arena_snapshot(a);
    float* qkv_cache = ARENA_CALLOC(a, float, 1024);
    float* attn_cache = ARENA_CALLOC(a, float, 512);
    ASSERT_NOT_NULL(qkv_cache, "qkv cache alloc");
    ASSERT_NOT_NULL(attn_cache, "attn cache alloc");

    size_t persistent_used = a->used;

    /* Per forward step: allocate temp buffers, then restore. */
    int step;
    for (step = 0; step < 10; step++) {
        size_t step_mark = arena_snapshot(a);
        float* step_buf = ARENA_CALLOC(a, float, 256);
        ASSERT_NOT_NULL(step_buf, "step temp buf alloc");
        step_buf[0] = (float)step;
        arena_restore(a, step_mark);
    }

    /* Persistent allocations are preserved after step loops. */
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
