#include "test_harness.h"
#include "nn/nn_backend.h"
#include "nn/nn_infer_registry.h"
#include "nn/nn_train_registry.h"
#include <string.h>

/*
  Verify that all enabled NN backends expose a valid VTable.
  Each VTable is registered manually so the test can validate
  function-pointer completeness without auto-generated code.
*/

/* All VTable instances (defined in each backend's *_ops.c). */
extern const NNInferBackend g_mlp_infer_backend;
extern const NNTrainBackend g_mlp_train_backend;

extern const NNInferBackend g_transformer_infer_backend;
extern const NNTrainBackend g_transformer_train_backend;

extern const NNInferBackend g_cnn_infer_backend;
extern const NNTrainBackend g_cnn_train_backend;

extern const NNInferBackend g_rnn_infer_backend;
extern const NNTrainBackend g_rnn_train_backend;

extern const NNInferBackend g_gnn_infer_backend;
extern const NNTrainBackend g_gnn_train_backend;

extern const NNInferBackend g_cnn_dual_pool_infer_backend;
extern const NNTrainBackend g_cnn_dual_pool_train_backend;

/* ─── Registration ─── */
static void register_all(void) {
    nn_infer_vtable_register(&g_mlp_infer_backend);
    nn_infer_vtable_register(&g_transformer_infer_backend);
    nn_infer_vtable_register(&g_cnn_infer_backend);
    nn_infer_vtable_register(&g_rnn_infer_backend);
    nn_infer_vtable_register(&g_gnn_infer_backend);
    nn_infer_vtable_register(&g_cnn_dual_pool_infer_backend);

    nn_train_vtable_register(&g_mlp_train_backend);
    nn_train_vtable_register(&g_transformer_train_backend);
    nn_train_vtable_register(&g_cnn_train_backend);
    nn_train_vtable_register(&g_rnn_train_backend);
    nn_train_vtable_register(&g_gnn_train_backend);
    nn_train_vtable_register(&g_cnn_dual_pool_train_backend);
}

TEST(all_backends_registered) {
    register_all();
    ASSERT_TRUE(nn_infer_vtable_count() >= 5,
                "at least 5 infer backends registered");
    ASSERT_TRUE(nn_train_vtable_count() >= 5,
                "at least 5 train backends registered");
}

/* ─── Infer VTable tests ─── */

TEST(mlp_infer_all_slots) {
    const NNInferBackend* b = nn_infer_vtable_find("mlp");
    ASSERT_NOT_NULL(b, "mlp found");
    if (!b) return;
    ASSERT_STREQ("mlp", b->type_name, "mlp type_name");
    ASSERT_NOT_NULL(b->create, "mlp create");
    ASSERT_NOT_NULL(b->destroy, "mlp destroy");
    ASSERT_NOT_NULL(b->step, "mlp step");
    ASSERT_NOT_NULL(b->get_output, "mlp get_output");
    ASSERT_NOT_NULL(b->save_weights, "mlp save_weights");
    ASSERT_NOT_NULL(b->load_weights, "mlp load_weights");
    ASSERT_NOT_NULL(b->get_network_hash, "mlp get_network_hash");
    ASSERT_NOT_NULL(b->get_layout_hash, "mlp get_layout_hash");
    ASSERT_NOT_NULL(b->get_abi_version, "mlp get_abi_version");
    ASSERT_TRUE(b->get_abi_version() > 0, "mlp abi > 0");
}

TEST(transformer_infer_all_slots) {
    const NNInferBackend* b = nn_infer_vtable_find("transformer");
    ASSERT_NOT_NULL(b, "transformer found");
    if (!b) return;
    ASSERT_STREQ("transformer", b->type_name, "transformer type_name");
    ASSERT_NOT_NULL(b->create, "transformer create");
    ASSERT_NOT_NULL(b->destroy, "transformer destroy");
    ASSERT_NOT_NULL(b->step, "transformer step");
    ASSERT_NOT_NULL(b->get_output, "transformer get_output");
    ASSERT_NOT_NULL(b->save_weights, "transformer save_weights");
    ASSERT_NOT_NULL(b->load_weights, "transformer load_weights");
    ASSERT_NOT_NULL(b->get_network_hash, "transformer get_network_hash");
    ASSERT_NOT_NULL(b->get_layout_hash, "transformer get_layout_hash");
    ASSERT_NOT_NULL(b->get_abi_version, "transformer get_abi_version");
    ASSERT_TRUE(b->get_abi_version() > 0, "transformer abi > 0");
}

TEST(cnn_infer_all_slots) {
    const NNInferBackend* b = nn_infer_vtable_find("cnn");
    ASSERT_NOT_NULL(b, "cnn found");
    if (!b) return;
    ASSERT_STREQ("cnn", b->type_name, "cnn type_name");
    ASSERT_NOT_NULL(b->create, "cnn create");
    ASSERT_NOT_NULL(b->destroy, "cnn destroy");
    ASSERT_NOT_NULL(b->step, "cnn step");
    ASSERT_NOT_NULL(b->get_output, "cnn get_output");
    ASSERT_NOT_NULL(b->save_weights, "cnn save_weights");
    ASSERT_NOT_NULL(b->load_weights, "cnn load_weights");
    ASSERT_NOT_NULL(b->get_network_hash, "cnn get_network_hash");
    ASSERT_NOT_NULL(b->get_layout_hash, "cnn get_layout_hash");
    ASSERT_NOT_NULL(b->get_abi_version, "cnn get_abi_version");
    ASSERT_TRUE(b->get_abi_version() > 0, "cnn abi > 0");
}

TEST(rnn_infer_all_slots) {
    const NNInferBackend* b = nn_infer_vtable_find("rnn");
    ASSERT_NOT_NULL(b, "rnn found");
    if (!b) return;
    ASSERT_STREQ("rnn", b->type_name, "rnn type_name");
    ASSERT_NOT_NULL(b->create, "rnn create");
    ASSERT_NOT_NULL(b->destroy, "rnn destroy");
    ASSERT_NOT_NULL(b->step, "rnn step");
    ASSERT_NOT_NULL(b->get_output, "rnn get_output");
    ASSERT_NOT_NULL(b->save_weights, "rnn save_weights");
    ASSERT_NOT_NULL(b->load_weights, "rnn load_weights");
    ASSERT_NOT_NULL(b->get_network_hash, "rnn get_network_hash");
    ASSERT_NOT_NULL(b->get_layout_hash, "rnn get_layout_hash");
    ASSERT_NOT_NULL(b->get_abi_version, "rnn get_abi_version");
    ASSERT_TRUE(b->get_abi_version() > 0, "rnn abi > 0");
}

TEST(gnn_infer_all_slots) {
    const NNInferBackend* b = nn_infer_vtable_find("gnn");
    ASSERT_NOT_NULL(b, "gnn found");
    if (!b) return;
    ASSERT_STREQ("gnn", b->type_name, "gnn type_name");
    ASSERT_NOT_NULL(b->create, "gnn create");
    ASSERT_NOT_NULL(b->destroy, "gnn destroy");
    ASSERT_NOT_NULL(b->step, "gnn step");
    ASSERT_NOT_NULL(b->get_output, "gnn get_output");
    ASSERT_NOT_NULL(b->save_weights, "gnn save_weights");
    ASSERT_NOT_NULL(b->load_weights, "gnn load_weights");
    ASSERT_NOT_NULL(b->get_network_hash, "gnn get_network_hash");
    ASSERT_NOT_NULL(b->get_layout_hash, "gnn get_layout_hash");
    ASSERT_NOT_NULL(b->get_abi_version, "gnn get_abi_version");
    ASSERT_TRUE(b->get_abi_version() > 0, "gnn abi > 0");
}

TEST(cnn_dual_pool_infer_all_slots) {
    const NNInferBackend* b = nn_infer_vtable_find("cnn_dual_pool");
    ASSERT_NOT_NULL(b, "cnn_dual_pool found");
    if (!b) return;
    ASSERT_STREQ("cnn_dual_pool", b->type_name, "cnn_dual_pool type_name");
    ASSERT_NOT_NULL(b->create, "cnn_dual_pool create");
    ASSERT_NOT_NULL(b->destroy, "cnn_dual_pool destroy");
    ASSERT_NOT_NULL(b->step, "cnn_dual_pool step");
    ASSERT_NOT_NULL(b->get_output, "cnn_dual_pool get_output");
    ASSERT_NOT_NULL(b->save_weights, "cnn_dual_pool save_weights");
    ASSERT_NOT_NULL(b->load_weights, "cnn_dual_pool load_weights");
    ASSERT_NOT_NULL(b->get_network_hash, "cnn_dual_pool get_network_hash");
    ASSERT_NOT_NULL(b->get_layout_hash, "cnn_dual_pool get_layout_hash");
    ASSERT_NOT_NULL(b->get_abi_version, "cnn_dual_pool get_abi_version");
    ASSERT_TRUE(b->get_abi_version() > 0, "cnn_dual_pool abi > 0");
}

/* ─── Train VTable tests ─── */

TEST(mlp_train_all_slots) {
    const NNTrainBackend* b = nn_train_vtable_find("mlp");
    ASSERT_NOT_NULL(b, "mlp train found");
    if (!b) return;
    ASSERT_STREQ("mlp", b->type_name, "mlp train type_name");
    ASSERT_NOT_NULL(b->create, "mlp train create");
    ASSERT_NOT_NULL(b->destroy, "mlp train destroy");
    ASSERT_NOT_NULL(b->step, "mlp train step");
    ASSERT_NOT_NULL(b->step_with_data, "mlp train step_with_data");
    ASSERT_NOT_NULL(b->save_checkpoint, "mlp train save_checkpoint");
    ASSERT_NOT_NULL(b->load_checkpoint, "mlp train load_checkpoint");
}

TEST(transformer_train_all_slots) {
    const NNTrainBackend* b = nn_train_vtable_find("transformer");
    ASSERT_NOT_NULL(b, "transformer train found");
    if (!b) return;
    ASSERT_STREQ("transformer", b->type_name, "transformer train type_name");
    ASSERT_NOT_NULL(b->create, "transformer train create");
    ASSERT_NOT_NULL(b->destroy, "transformer train destroy");
    ASSERT_NOT_NULL(b->step, "transformer train step");
    ASSERT_NOT_NULL(b->step_with_data, "transformer train step_with_data");
    ASSERT_NOT_NULL(b->save_checkpoint, "transformer train save_checkpoint");
    ASSERT_NOT_NULL(b->load_checkpoint, "transformer train load_checkpoint");
}

TEST(cnn_train_all_slots) {
    const NNTrainBackend* b = nn_train_vtable_find("cnn");
    ASSERT_NOT_NULL(b, "cnn train found");
    if (!b) return;
    ASSERT_STREQ("cnn", b->type_name, "cnn train type_name");
    ASSERT_NOT_NULL(b->create, "cnn train create");
    ASSERT_NOT_NULL(b->destroy, "cnn train destroy");
    ASSERT_NOT_NULL(b->step, "cnn train step");
    ASSERT_NOT_NULL(b->step_with_data, "cnn train step_with_data");
    ASSERT_NOT_NULL(b->save_checkpoint, "cnn train save_checkpoint");
    ASSERT_NOT_NULL(b->load_checkpoint, "cnn train load_checkpoint");
}

TEST(rnn_train_all_slots) {
    const NNTrainBackend* b = nn_train_vtable_find("rnn");
    ASSERT_NOT_NULL(b, "rnn train found");
    if (!b) return;
    ASSERT_STREQ("rnn", b->type_name, "rnn train type_name");
    ASSERT_NOT_NULL(b->create, "rnn train create");
    ASSERT_NOT_NULL(b->destroy, "rnn train destroy");
    ASSERT_NOT_NULL(b->step, "rnn train step");
    ASSERT_NOT_NULL(b->step_with_data, "rnn train step_with_data");
    ASSERT_NOT_NULL(b->save_checkpoint, "rnn train save_checkpoint");
    ASSERT_NOT_NULL(b->load_checkpoint, "rnn train load_checkpoint");
}

TEST(gnn_train_all_slots) {
    const NNTrainBackend* b = nn_train_vtable_find("gnn");
    ASSERT_NOT_NULL(b, "gnn train found");
    if (!b) return;
    ASSERT_STREQ("gnn", b->type_name, "gnn train type_name");
    ASSERT_NOT_NULL(b->create, "gnn train create");
    ASSERT_NOT_NULL(b->destroy, "gnn train destroy");
    ASSERT_NOT_NULL(b->step, "gnn train step");
    ASSERT_NOT_NULL(b->step_with_data, "gnn train step_with_data");
    ASSERT_NOT_NULL(b->save_checkpoint, "gnn train save_checkpoint");
    ASSERT_NOT_NULL(b->load_checkpoint, "gnn train load_checkpoint");
}

TEST(cnn_dual_pool_train_all_slots) {
    const NNTrainBackend* b = nn_train_vtable_find("cnn_dual_pool");
    ASSERT_NOT_NULL(b, "cnn_dual_pool train found");
    if (!b) return;
    ASSERT_STREQ("cnn_dual_pool", b->type_name, "cnn_dual_pool train type_name");
    ASSERT_NOT_NULL(b->create, "cnn_dual_pool train create");
    ASSERT_NOT_NULL(b->destroy, "cnn_dual_pool train destroy");
    ASSERT_NOT_NULL(b->step, "cnn_dual_pool train step");
    ASSERT_NOT_NULL(b->step_with_data, "cnn_dual_pool train step_with_data");
    ASSERT_NOT_NULL(b->save_checkpoint, "cnn_dual_pool train save_checkpoint");
    ASSERT_NOT_NULL(b->load_checkpoint, "cnn_dual_pool train load_checkpoint");
}

int main(void) {
    RUN_TEST(all_backends_registered);
    RUN_TEST(mlp_infer_all_slots);
    RUN_TEST(transformer_infer_all_slots);
    RUN_TEST(cnn_infer_all_slots);
    RUN_TEST(rnn_infer_all_slots);
    RUN_TEST(gnn_infer_all_slots);
    RUN_TEST(cnn_dual_pool_infer_all_slots);
    RUN_TEST(mlp_train_all_slots);
    RUN_TEST(transformer_train_all_slots);
    RUN_TEST(cnn_train_all_slots);
    RUN_TEST(rnn_train_all_slots);
    RUN_TEST(gnn_train_all_slots);
    RUN_TEST(cnn_dual_pool_train_all_slots);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
