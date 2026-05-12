#include "test_harness.h"
#include "nn/nn_backend.h"

#include <stddef.h>

/* Verify NNInferBackend has the correct number of function-pointer slots:
 * 1 const char* + 6 fn ptrs + 2 fn ptrs + 1 fn ptr = 10 pointers */
TEST(nn_infer_backend_size) {
    /* type_name(1) + create(2) + destroy(3) + step(4) + get_output(5)
     * + save_weights(6) + load_weights(7) + get_network_hash(8)
     * + get_layout_hash(9) + get_abi_version(10) = 10 pointer slots */
    ASSERT_EQ_SIZE(sizeof(void*) * 10, sizeof(NNInferBackend),
                   "NNInferBackend has 10 pointer-width slots");
}

/* Verify NNTrainBackend: 1 const char* + 5 fn ptrs + 1 fn ptr = 7 pointers */
TEST(nn_train_backend_size) {
    /* type_name(1) + create(2) + destroy(3) + step(4)
     * + step_with_data(5) + save_checkpoint(6) + load_checkpoint(7) */
    ASSERT_EQ_SIZE(sizeof(void*) * 7, sizeof(NNTrainBackend),
                   "NNTrainBackend has 7 pointer-width slots");
}

int main(void) {
    RUN_TEST(nn_infer_backend_size);
    RUN_TEST(nn_train_backend_size);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
