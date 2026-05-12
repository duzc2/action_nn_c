#include "test_harness.h"
#include "nn/nn_backend.h"
#include <string.h>

/*
  Verify MLP VTable integrity:
  1. g_mlp_infer_backend exists with type_name = "mlp"
  2. All infer function pointers are non-null
  3. get_abi_version returns a reasonable value (> 0)
  4. All train function pointers are non-null
*/

extern const NNInferBackend g_mlp_infer_backend;
extern const NNTrainBackend g_mlp_train_backend;

TEST(mlp_infer_type_name) {
    ASSERT_STREQ("mlp", g_mlp_infer_backend.type_name, "type_name is 'mlp'");
}

TEST(mlp_infer_all_functions_non_null) {
    ASSERT_NOT_NULL(g_mlp_infer_backend.create, "create slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.destroy, "destroy slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.step, "step slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.get_output, "get_output slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.save_weights, "save_weights slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.load_weights, "load_weights slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.get_network_hash, "get_network_hash slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.get_layout_hash, "get_layout_hash slot");
    ASSERT_NOT_NULL(g_mlp_infer_backend.get_abi_version, "get_abi_version slot");
}

TEST(mlp_abi_version_positive) {
    uint32_t v = g_mlp_infer_backend.get_abi_version();
    ASSERT_TRUE(v > 0, "abi_version is positive");
}

TEST(mlp_train_all_functions_non_null) {
    ASSERT_NOT_NULL(g_mlp_train_backend.create, "train create slot");
    ASSERT_NOT_NULL(g_mlp_train_backend.destroy, "train destroy slot");
    ASSERT_NOT_NULL(g_mlp_train_backend.step, "train step slot");
    ASSERT_NOT_NULL(g_mlp_train_backend.step_with_data, "train step_with_data slot");
    ASSERT_NOT_NULL(g_mlp_train_backend.save_checkpoint, "save_checkpoint slot");
    ASSERT_NOT_NULL(g_mlp_train_backend.load_checkpoint, "load_checkpoint slot");
}

TEST(mlp_train_type_name) {
    ASSERT_STREQ("mlp", g_mlp_train_backend.type_name, "train type_name is 'mlp'");
}

int main(void) {
    RUN_TEST(mlp_infer_type_name);
    RUN_TEST(mlp_infer_all_functions_non_null);
    RUN_TEST(mlp_abi_version_positive);
    RUN_TEST(mlp_train_all_functions_non_null);
    RUN_TEST(mlp_train_type_name);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
