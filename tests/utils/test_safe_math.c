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
    /* SIZE_MAX * 2 should overflow */
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
