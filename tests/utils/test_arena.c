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
        if (p[i] != 0) {
            ASSERT_EQ_INT(0, p[i], "calloc'd memory should be zero");
            break; /* unreachable if ASSERT passes; breaks out on first fail */
        }
    }
    ASSERT_TRUE(1, "calloc zeros verification done");
    arena_destroy(a);
}

TEST(arena_snapshot_restore) {
    Arena* a = arena_create(1024);
    ARENA_ALLOC(a, int, 10);
    size_t mark = arena_snapshot(a);
    ASSERT_EQ_SIZE(10 * sizeof(int), mark, "snapshot captures used");

    ARENA_ALLOC(a, int, 20);
    ASSERT_TRUE(a->used > mark, "used increases after more alloc");

    arena_restore(a, mark);
    ASSERT_EQ_SIZE(mark, a->used, "restore resets used to mark");

    int* p = ARENA_ALLOC(a, int, 5);
    ASSERT_NOT_NULL(p, "alloc after restore should succeed");
    arena_destroy(a);
}

TEST(arena_grow) {
    Arena* a = arena_create(16);
    char* p = ARENA_ALLOC(a, char, 2048);
    ASSERT_NOT_NULL(p, "alloc beyond initial capacity should trigger grow");
    ASSERT_TRUE(a->capacity >= 2048, "capacity should have grown");
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
