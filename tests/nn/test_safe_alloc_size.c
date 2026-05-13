/**
 * @file test_safe_alloc_size.c
 * @brief Unit tests for SAFE_ALLOC_SIZE overflow detection (B10).
 */

#include "test_harness.h"
#include "utils/safe_math.h"

TEST(safe_alloc_size_normal) {
    /* 100 floats = 100 * 4 = 400 bytes */
    size_t n = SAFE_ALLOC_SIZE(100, float);
    ASSERT_EQ_SIZE(100 * sizeof(float), n, "SAFE_ALLOC_SIZE(100, float) == 400");
}

TEST(safe_alloc_size_large_ok) {
    /* SIZE_MAX / 4 floats should fit */
    size_t count = SIZE_MAX / sizeof(float);
    size_t n = SAFE_ALLOC_SIZE(count, float);
    ASSERT_TRUE(n > 0, "SAFE_ALLOC_SIZE(SIZE_MAX/sizeof(float), float) should not overflow");
    ASSERT_EQ_SIZE(count * sizeof(float), n, "result equals expected bytes");
}

TEST(safe_alloc_size_overflow) {
    /* SIZE_MAX / sizeof(float) + 1 should overflow to 0 */
    size_t count = SIZE_MAX / sizeof(float);
    if (count < SIZE_MAX) {
        count++;
        size_t n = SAFE_ALLOC_SIZE(count, float);
        ASSERT_EQ_SIZE(0, n, "SAFE_ALLOC_SIZE(SIZE_MAX/sizeof(float)+1, float) returns 0 on overflow");
    } else {
        /* Can't test overflow on this platform (sizeof(float) == 1 unlikely) */
        ASSERT_TRUE(1, "overflow test skipped — sizeof(float) == 1");
    }
}

TEST(safe_alloc_size_zero_count) {
    size_t n = SAFE_ALLOC_SIZE(0, float);
    ASSERT_EQ_SIZE(0, n, "SAFE_ALLOC_SIZE(0, float) returns 0");
}

TEST(safe_alloc_size_zero_count_int) {
    size_t n = SAFE_ALLOC_SIZE(0, int);
    ASSERT_EQ_SIZE(0, n, "SAFE_ALLOC_SIZE(0, int) returns 0");
}

TEST(safe_alloc_size_single_element) {
    size_t n = SAFE_ALLOC_SIZE(1, double);
    ASSERT_EQ_SIZE(sizeof(double), n, "SAFE_ALLOC_SIZE(1, double) == sizeof(double)");
}

TEST(safe_size_mul_large_ok) {
    size_t n = safe_size_mul(SIZE_MAX / 2, 2, SIZE_MAX);
    ASSERT_TRUE(n > 0, "(SIZE_MAX/2)*2 should not overflow");
    ASSERT_EQ_SIZE(SIZE_MAX / 2 * 2, n, "result is correct");
}

TEST(safe_size_mul_overflow_limit) {
    size_t n = safe_size_mul(100, 200, 19999);
    ASSERT_EQ_SIZE(0, n, "100*200 > 19999 limit returns 0");
}

int main(void) {
    RUN_TEST(safe_alloc_size_normal);
    RUN_TEST(safe_alloc_size_large_ok);
    RUN_TEST(safe_alloc_size_overflow);
    RUN_TEST(safe_alloc_size_zero_count);
    RUN_TEST(safe_alloc_size_zero_count_int);
    RUN_TEST(safe_alloc_size_single_element);
    RUN_TEST(safe_size_mul_large_ok);
    RUN_TEST(safe_size_mul_overflow_limit);

    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
