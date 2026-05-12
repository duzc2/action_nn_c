#include "test_harness.h"
#include "utils/arena.h"

TEST(arena_snapshot_restore_pattern) {
    /* Simulate the backward_pass snapshot/restore pattern. */
    Arena* a = arena_create(4096);
    ASSERT_NOT_NULL(a, "arena_create succeeds");

    /* First allocate — these simulate current_delta + next_delta. */
    size_t mark = arena_snapshot(a);
    float* buf1 = ARENA_CALLOC(a, float, 100);
    float* buf2 = ARENA_CALLOC(a, float, 100);
    ASSERT_NOT_NULL(buf1, "first calloc succeeds");
    ASSERT_NOT_NULL(buf2, "second calloc succeeds");
    buf1[0] = 3.14f;
    buf2[50] = 2.71f;

    /* Rollback to mark. */
    arena_restore(a, mark);
    ASSERT_EQ_SIZE(mark, a->used, "used back to mark after restore");

    /* Re-allocate — should reuse the same memory region. */
    float* buf3 = ARENA_CALLOC(a, float, 100);
    float* buf4 = ARENA_CALLOC(a, float, 100);
    ASSERT_NOT_NULL(buf3, "re-alloc after restore succeeds");
    ASSERT_NOT_NULL(buf4, "second re-alloc after restore succeeds");

    arena_destroy(a);
}

TEST(arena_calloc_is_zeroed_after_restore) {
    Arena* a = arena_create(4096);
    ASSERT_NOT_NULL(a, "arena_create succeeds");

    /* Write non-zero data, then rollback. */
    {
        size_t mark = arena_snapshot(a);
        int* p = ARENA_ALLOC(a, int, 100);
        ASSERT_NOT_NULL(p, "alloc for dirty write");
        for (int i = 0; i < 100; i++) p[i] = 999;
        arena_restore(a, mark);
    }

    /* After restore, a fresh calloc should give zero-initialized memory. */
    (void)arena_snapshot(a);
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
