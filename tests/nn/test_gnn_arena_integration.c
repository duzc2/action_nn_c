#include "test_harness.h"
#include "utils/arena.h"

/* Simulate GNN forward_pass 3-buffer snapshot/restore pattern. */
TEST(gnn_three_buffer_pattern) {
    Arena* a = arena_create(8192);
    ASSERT_NOT_NULL(a, "arena_create succeeds");

    size_t mark = arena_snapshot(a);
    float* owned_cache   = ARENA_CALLOC(a, float, 64);
    float* aggregated    = ARENA_CALLOC(a, float, 32);
    float* pooled_hidden = ARENA_CALLOC(a, float, 32);

    ASSERT_NOT_NULL(owned_cache, "owned_cache alloc");
    ASSERT_NOT_NULL(aggregated, "aggregated alloc");
    ASSERT_NOT_NULL(pooled_hidden, "pooled_hidden alloc");

    owned_cache[0] = 1.0f;
    aggregated[10] = 2.0f;
    pooled_hidden[5] = 3.0f;

    arena_restore(a, mark);

    /* Re-allocate: verify memory reuse and zeroing after restore. */
    float* buf2 = ARENA_CALLOC(a, float, 64);
    ASSERT_NOT_NULL(buf2, "re-alloc after restore succeeds");
    ASSERT_EQ_INT(0, (int)buf2[0], "calloc clears old owned_cache[0]");

    arena_destroy(a);
}

/* Simulate GNN backpropagate 6-buffer pattern. */
TEST(gnn_six_buffer_pattern) {
    Arena* a = arena_create(16384);
    ASSERT_NOT_NULL(a, "arena_create succeeds");

    size_t mark = arena_snapshot(a);
    int i;
    for (i = 0; i < 6; i++) {
        float* buf = ARENA_CALLOC(a, float, 128);
        ASSERT_NOT_NULL(buf, "buffer alloc");
        buf[0] = (float)i;
    }
    ASSERT_TRUE(a->used >= 6 * 128 * (int)sizeof(float), "used reflects 6 buffers");

    arena_restore(a, mark);
    ASSERT_EQ_SIZE(mark, a->used, "6 buffers fully reclaimed after restore");

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
