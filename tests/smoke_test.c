#include "test_harness.h"

TEST(harness_assert_true_works)  { ASSERT_TRUE(1 == 1, "1 equals 1"); }
TEST(harness_assert_eq_works)    { ASSERT_EQ_INT(42, 42, "42 equals 42"); }
TEST(harness_assert_streq_works) { ASSERT_STREQ("hello", "hello", "strings match"); }

int main(void) {
    RUN_TEST(harness_assert_true_works);
    RUN_TEST(harness_assert_eq_works);
    RUN_TEST(harness_assert_streq_works);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
