#include "test_harness.h"
#include <stdio.h>
#include <string.h>

/*
 * Verifies the I/O error-handling pattern used by write_file() in
 * prof_codegen.c: fopen checked, fprintf return checked, fclose return
 * checked. write_file itself is static and tested indirectly through
 * the full profiler pipeline. These tests confirm the pattern compiles
 * and works correctly for normal file operations.
 */

TEST(write_and_read_back) {
    FILE* fp = fopen("twf_rw.txt", "w");
    ASSERT_NOT_NULL(fp, "fopen succeeds");
    ASSERT_TRUE(fprintf(fp, "%s", "Hello, Pipeline!") > 0, "fprintf wrote bytes");
    ASSERT_EQ_INT(0, fclose(fp), "fclose succeeds");

    fp = fopen("twf_rw.txt", "r");
    ASSERT_NOT_NULL(fp, "re-open for reading succeeds");
    char buf[256] = {0};
    (void)!fgets(buf, sizeof(buf), fp);
    ASSERT_STREQ("Hello, Pipeline!", buf, "file content matches");
    (void)fclose(fp);
    (void)remove("twf_rw.txt");
}

int main(void) {
    RUN_TEST(write_and_read_back);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
