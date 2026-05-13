#include "test_harness.h"
#include "profiler/prof_validate.h"
#include "profiler/prof_flatten.h"
#include "profiler/network_def.h"
#include "profiler/prof_error.h"
#include "profiler/prof_path.h"

#include <string.h>

/* --- helpers (reuse linear network builder pattern) --- */

static NN_NetworkDef* build_linear_network(void) {
    NN_NetworkDef* net = nn_network_def_create("val_net");
    NNSubnetDef* a = nn_subnet_def_create("a", "input", 2, 2);
    NNSubnetDef* b = nn_subnet_def_create("b", "mlp", 2, 2);
    NNSubnetDef* c = nn_subnet_def_create("c", "output", 2, 2);
    nn_network_def_add_subnet(net, a);
    nn_network_def_add_subnet(net, b);
    nn_network_def_add_subnet(net, c);
    /* Only add connections if there are at least 2 leaf subnets that
       will pass _flat validation (output_layer_size >= 1). */
    nn_network_def_add_connection(net,
        nn_connection_def_create("a", NULL, 0, "b", NULL, 0));
    nn_network_def_add_connection(net,
        nn_connection_def_create("b", NULL, 0, "c", NULL, 0));
    return net;
}

static void fill_output_layout_trivial(ProfOutputLayout* layout) {
    (void)memset(layout, 0, sizeof(*layout));
    layout->tokenizer.c_path    = "build/val/tokenizer.c";
    layout->tokenizer.h_path    = "build/val/tokenizer.h";
    layout->network_init.c_path = "build/val/network_init.c";
    layout->network_init.h_path = "build/val/network_init.h";
    layout->weights_load.c_path = "build/val/weights_load.c";
    layout->weights_load.h_path = "build/val/weights_load.h";
    layout->train.c_path        = "build/val/train.c";
    layout->train.h_path        = "build/val/train.h";
    layout->weights_save.c_path = "build/val/weights_save.c";
    layout->weights_save.h_path = "build/val/weights_save.h";
    layout->infer.c_path        = "build/val/infer.c";
    layout->infer.h_path        = "build/val/infer.h";
    layout->metadata_path       = "build/val/metadata.h";
}

/* --- tests --- */

TEST(validate_rejects_null_request) {
    char buf[256];
    ProfErrorBuffer err;
    prof_error_init(&err, buf, sizeof(buf));

    ProfStatus st = prof_validate_all(NULL, &err);
    ASSERT_TRUE(st != PROF_STATUS_OK,
                "validate_all rejects NULL request");
}

TEST(validate_dag_flat_detects_cycles) {
    /* Build a cycle network directly. */
    NN_NetworkDef* net = nn_network_def_create("cycle_test");
    NNSubnetDef* a = nn_subnet_def_create("x", "mlp", 2, 2);
    NNSubnetDef* b = nn_subnet_def_create("y", "mlp", 2, 2);
    nn_network_def_add_subnet(net, a);
    nn_network_def_add_subnet(net, b);
    nn_network_def_add_connection(net,
        nn_connection_def_create("x", NULL, 0, "y", NULL, 0));
    nn_network_def_add_connection(net,
        nn_connection_def_create("y", NULL, 0, "x", NULL, 0));

    FlatNetwork flat;
    (void)memset(&flat, 0, sizeof(flat));
    int rc = prof_flatten_build(net, &flat);
    ASSERT_EQ_INT(0, rc, "flatten succeeded");
    ASSERT_EQ_INT(1, flat.has_cycles, "cycle detected");

    char buf[256];
    ProfErrorBuffer err;
    prof_error_init(&err, buf, sizeof(buf));

    ProfStatus st = prof_validate_dag_flat(&flat, &err);
    ASSERT_EQ_INT(PROF_STATUS_CYCLE_DETECTED, (int)st,
                  "dag_flat reports cycle error");

    prof_flatten_free(&flat);
    nn_network_def_free(net);
}

TEST(validate_dag_flat_accepts_acyclic) {
    NN_NetworkDef* net = build_linear_network();

    FlatNetwork flat;
    (void)memset(&flat, 0, sizeof(flat));
    int rc = prof_flatten_build(net, &flat);
    ASSERT_EQ_INT(0, rc, "flatten succeeded");
    ASSERT_EQ_INT(0, flat.has_cycles, "no cycles");

    char buf[256];
    ProfErrorBuffer err;
    prof_error_init(&err, buf, sizeof(buf));

    ProfStatus st = prof_validate_dag_flat(&flat, &err);
    ASSERT_EQ_INT(PROF_STATUS_OK, (int)st,
                  "dag_flat passes for acyclic network");

    prof_flatten_free(&flat);
    nn_network_def_free(net);
}

TEST(validate_connections_flat_rejects_bad_node_index) {
    NN_NetworkDef* net = nn_network_def_create("bad_idx");
    NNSubnetDef* a = nn_subnet_def_create("src", "mlp", 2, 2);
    NNSubnetDef* b = nn_subnet_def_create("dst", "mlp", 2, 2);
    nn_network_def_add_subnet(net, a);
    nn_network_def_add_subnet(net, b);
    /* dst has output_layer_size=2, so source_node_index=5 is out of range. */
    nn_network_def_add_connection(net,
        nn_connection_def_create("src", NULL, 5, "dst", NULL, 0));

    FlatNetwork flat;
    (void)memset(&flat, 0, sizeof(flat));
    int rc = prof_flatten_build(net, &flat);
    ASSERT_EQ_INT(0, rc, "flatten succeeded");

    char buf[256];
    ProfErrorBuffer err;
    prof_error_init(&err, buf, sizeof(buf));

    ProfStatus st = prof_validate_connections_flat(net, &flat, &err);
    ASSERT_EQ_INT(PROF_STATUS_VALIDATION_FAILED, (int)st,
                  "bad node index rejected");

    prof_flatten_free(&flat);
    nn_network_def_free(net);
}

TEST(validate_connections_flat_null_guards) {
    char buf[256];
    ProfErrorBuffer err;
    prof_error_init(&err, buf, sizeof(buf));

    ProfStatus st = prof_validate_connections_flat(NULL, NULL, &err);
    ASSERT_EQ_INT(PROF_STATUS_OK, (int)st,
                  "connections_flat returns OK for NULL args");
}

int main(void) {
    RUN_TEST(validate_rejects_null_request);
    RUN_TEST(validate_dag_flat_detects_cycles);
    RUN_TEST(validate_dag_flat_accepts_acyclic);
    RUN_TEST(validate_connections_flat_rejects_bad_node_index);
    RUN_TEST(validate_connections_flat_null_guards);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
