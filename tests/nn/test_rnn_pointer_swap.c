#include "test_harness.h"
#include <string.h>

/* Verify pointer swap + memset is equivalent to memcpy + memset. */
TEST(pointer_swap_vs_memcpy) {
    float buf_a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float buf_b[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    /* dh_next=grad_a, dh_current=grad_b; accumulate into dh_next.
     * Swap: dh_current gets the accumulated data, dh_next gets zeroed. */
    float* current = buf_b;
    float* next = buf_a;
    {
        float* tmp = current;
        current = next;
        next = tmp;
    }
    memset(next, 0, 8 * sizeof(float));

    /* current now points to buf_a (had accumulated data) */
    ASSERT_EQ_INT(1, (int)current[0], "current points to old next (preserved data)");
    ASSERT_EQ_INT(8, (int)current[7], "current[7] preserved");
    /* next now points to buf_b (old current), which has been zeroed */
    ASSERT_EQ_INT(0, (int)next[0], "next zeroed after swap");
    ASSERT_EQ_INT(0, (int)buf_b[0], "original buf_b zeroed");
}

int main(void) {
    RUN_TEST(pointer_swap_vs_memcpy);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
