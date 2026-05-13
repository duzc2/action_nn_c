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
    int i;
    for (i = 0; i < 1000; i++) {
        snprintf(keys[i], sizeof(keys[i]), "key_%04d", i);
        string_hash_map_put(&m, keys[i], (size_t)i);
    }

    /* All entries retrievable after bulk insert (triggers resize). */
    for (i = 0; i < 1000; i++) {
        size_t val;
        ASSERT_TRUE(string_hash_map_get(&m, keys[i], &val),
                    "key found after bulk insert");
        ASSERT_EQ_SIZE((size_t)i, val, "value correct after bulk insert");
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

TEST(hash_map_min_capacity) {
    StringHashMap m;
    string_hash_map_init(&m, 2); /* less than minimum */
    ASSERT_TRUE(m.capacity >= 8, "capacity clamped to minimum 8");

    string_hash_map_put(&m, "a", 1);
    size_t val;
    ASSERT_TRUE(string_hash_map_get(&m, "a", &val), "works at min capacity");

    string_hash_map_free(&m);
}

int main(void) {
    RUN_TEST(hash_map_put_and_get);
    RUN_TEST(hash_map_replace_existing);
    RUN_TEST(hash_map_large_insert);
    RUN_TEST(hash_map_empty_get);
    RUN_TEST(hash_map_min_capacity);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
