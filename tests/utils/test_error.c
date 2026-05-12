#include "test_harness.h"
#include "utils/error.h"

TEST(action_c_ok_is_zero) {
    ASSERT_EQ_INT(0, ACTION_C_OK, "OK should be 0");
}

TEST(error_values_are_negative) {
    ASSERT_TRUE(ACTION_C_ERR_NULL_POINTER < 0, "null pointer is negative");
    ASSERT_TRUE(ACTION_C_ERR_NO_MEMORY < 0, "no memory is negative");
    ASSERT_TRUE(ACTION_C_ERR_INTERNAL < 0, "internal is negative");
}

TEST(error_ranges_no_collision) {
    /* null / invalid / out_of_range in [-9, -1] */
    ASSERT_TRUE(ACTION_C_ERR_NULL_POINTER >= -9, "param errors >= -9");
    ASSERT_TRUE(ACTION_C_ERR_OUT_OF_RANGE >= -9, "param errors >= -9");
    /* no_memory / io in [-19, -10] */
    ASSERT_TRUE(ACTION_C_ERR_NO_MEMORY <= -10, "resource errors <= -10");
    ASSERT_TRUE(ACTION_C_ERR_NO_MEMORY >= -19, "resource errors >= -19");
    /* dim_mismatch / version / config / cycle in [-29, -20] */
    ASSERT_TRUE(ACTION_C_ERR_DIM_MISMATCH <= -20, "data errors <= -20");
    ASSERT_TRUE(ACTION_C_ERR_CYCLE_DETECTED >= -29, "data errors >= -29");
    /* not_found near -30 */
    ASSERT_TRUE(ACTION_C_ERR_NOT_FOUND <= -30, "not_found <= -30");
    ASSERT_TRUE(ACTION_C_ERR_NOT_FOUND >= -39, "not_found >= -39");
}

int main(void) {
    RUN_TEST(action_c_ok_is_zero);
    RUN_TEST(error_values_are_negative);
    RUN_TEST(error_ranges_no_collision);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
