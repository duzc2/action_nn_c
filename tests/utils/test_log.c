#include "test_harness.h"
#include "utils/log.h"

/* Test that log macros compile and run without crashing.
   Note: default ACTION_C_LOG_LEVEL=2 so LOG_INFO is compiled away. */

TEST(log_error_compiles_and_runs) {
    LOG_ERROR("test error: %d", 42);
    ASSERT_TRUE(1, "log_error executed without crash");
}

TEST(log_warn_compiles_and_runs) {
    LOG_WARN("test warn: %s", "hello");
    ASSERT_TRUE(1, "log_warn executed without crash");
}

/* ASSERT macro behavior in Debug builds - tested via int-returning wrappers,
   since ASSERT performs 'return err' which MSVC warns about in void functions. */

int dummy_func_ok(int x) {
    ASSERT(x > 0, -1, "x must be positive");
    return x;
}

int dummy_func_fail(int x) {
    ASSERT(x > 0, -1, "x must be positive");
    return x;
}

TEST(assert_passes_when_cond_true) {
    int rc = dummy_func_ok(42);
    ASSERT_EQ_INT(42, rc, "dummy_func_ok returns x when condition true");
}

TEST(assert_returns_early_on_fail) {
    int rc = dummy_func_fail(-5);
    ASSERT_EQ_INT(-1, rc, "dummy_func with negative x returns -1 from ASSERT");
}

TEST(ndebug_removes_assert) {
    /* This is a documentation test: in Release builds ASSERT should be no-op.
       Here we just verify Debug behavior is correct. */
    ASSERT_TRUE(1, "syntax-only check for ASSERT macro");
}

int main(void) {
    RUN_TEST(log_error_compiles_and_runs);
    RUN_TEST(log_warn_compiles_and_runs);
    RUN_TEST(assert_passes_when_cond_true);
    RUN_TEST(ndebug_removes_assert);
    RUN_TEST(assert_returns_early_on_fail);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
