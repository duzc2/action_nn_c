#include "test_harness.h"
#include "profiler/prof_flatten.h"
#include "profiler/network_def.h"

#include <string.h>

/* --- helpers --- */

static NN_NetworkDef* build_linear_network(void) {
    NN_NetworkDef* net = nn_network_def_create("test_net");
    NNSubnetDef* a = nn_subnet_def_create("a", "input", 2, 2);
    NNSubnetDef* b = nn_subnet_def_create("b", "mlp", 2, 2);
    NNSubnetDef* c = nn_subnet_def_create("c", "output", 2, 2);
    nn_network_def_add_subnet(net, a);
    nn_network_def_add_subnet(net, b);
    nn_network_def_add_subnet(net, c);
    nn_network_def_add_connection(net,
        nn_connection_def_create("a", NULL, 0, "b", NULL, 0));
    nn_network_def_add_connection(net,
        nn_connection_def_create("b", NULL, 0, "c", NULL, 0));
    return net;
}

/* --- tests --- */

TEST(flatten_linear_network) {
    NN_NetworkDef* net = build_linear_network();
    FlatNetwork flat;
    (void)memset(&flat, 0, sizeof(flat));

    int rc = prof_flatten_build(net, &flat);
    ASSERT_EQ_INT(0, rc, "flatten succeeds for linear network");

    ASSERT_EQ_SIZE(3, flat.leaves.count, "3 leaf subnets");
    ASSERT_EQ_INT(0, flat.has_cycles, "linear network has no cycles");

    /* Topological order covers all leaves. */
    ASSERT_NOT_NULL(flat.topological_order, "topological order is set");

    /* ID lookup via hash map (O(1)). */
    size_t idx;
    ASSERT_TRUE(string_hash_map_get(&flat.id_to_index, "a", &idx),
                "subnet 'a' found in id_to_index");
    ASSERT_TRUE(string_hash_map_get(&flat.id_to_index, "c", &idx),
                "subnet 'c' found in id_to_index");
    ASSERT_TRUE(!string_hash_map_get(&flat.id_to_index, "nonexistent", &idx),
                "nonexistent subnet not found");

    prof_flatten_free(&flat);
    nn_network_def_free(net);
}

TEST(flatten_cycle_detection) {
    NN_NetworkDef* net = nn_network_def_create("cycle_net");
    NNSubnetDef* a = nn_subnet_def_create("a", "mlp", 2, 2);
    NNSubnetDef* b = nn_subnet_def_create("b", "mlp", 2, 2);
    nn_network_def_add_subnet(net, a);
    nn_network_def_add_subnet(net, b);
    /* a → b */
    nn_network_def_add_connection(net,
        nn_connection_def_create("a", NULL, 0, "b", NULL, 0));
    /* b → a  (creates a 2-node cycle) */
    nn_network_def_add_connection(net,
        nn_connection_def_create("b", NULL, 0, "a", NULL, 0));

    FlatNetwork flat;
    (void)memset(&flat, 0, sizeof(flat));
    int rc = prof_flatten_build(net, &flat);

    /* build does NOT fail on cycles -- it sets has_cycles for the caller
       to inspect, because cycle detection is a validation concern. */
    ASSERT_TRUE(rc == 0, "flatten build succeeds structurally");
    ASSERT_EQ_INT(1, flat.has_cycles, "cycle network is flagged as cyclic");

    prof_flatten_free(&flat);
    nn_network_def_free(net);
}

TEST(flatten_single_node) {
    NN_NetworkDef* net = nn_network_def_create("single");
    nn_network_def_add_subnet(net,
        nn_subnet_def_create("only", "input", 1, 1));

    FlatNetwork flat;
    (void)memset(&flat, 0, sizeof(flat));
    int rc = prof_flatten_build(net, &flat);
    ASSERT_EQ_INT(0, rc, "flatten succeeds for single node");

    ASSERT_EQ_SIZE(1, flat.leaves.count, "1 leaf");
    ASSERT_EQ_INT(0, flat.has_cycles, "single node has no cycles");

    size_t idx;
    ASSERT_TRUE(string_hash_map_get(&flat.id_to_index, "only", &idx),
                "single node found");
    ASSERT_EQ_SIZE(0, idx, "single node has index 0");

    prof_flatten_free(&flat);
    nn_network_def_free(net);
}

TEST(flatten_null_guard) {
    int rc = prof_flatten_build(NULL, NULL);
    ASSERT_TRUE(rc < 0, "flatten_build rejects NULL args");

    FlatNetwork f;
    (void)memset(&f, 0, sizeof(f));
    rc = prof_flatten_build(NULL, &f);
    ASSERT_TRUE(rc < 0, "flatten_build rejects NULL network");
    rc = prof_flatten_build(NULL, NULL);
    ASSERT_TRUE(rc < 0, "flatten_build rejects both NULL");

    /* prof_flatten_free on NULL is safe. */
    prof_flatten_free(NULL);
    ASSERT_TRUE(1, "prof_flatten_free(NULL) does not crash");
}

TEST(hash_map_empty_get) {
    StringHashMap m;
    string_hash_map_init(&m, 16);
    ASSERT_EQ_SIZE(16, m.capacity, "capacity is at least requested");

    size_t val;
    ASSERT_TRUE(!string_hash_map_get(&m, "anything", &val),
                "empty map returns nothing");

    string_hash_map_free(&m);
    ASSERT_NULL(m.keys, "keys freed");
    ASSERT_NULL(m.values, "values freed");
}

TEST(hash_map_put_and_get) {
    StringHashMap m;
    string_hash_map_init(&m, 16);

    string_hash_map_put(&m, "key1", 42);
    string_hash_map_put(&m, "key2", 100);

    size_t val;
    ASSERT_TRUE(string_hash_map_get(&m, "key1", &val), "key1 found");
    ASSERT_EQ_SIZE(42, val, "key1 value correct");

    ASSERT_TRUE(string_hash_map_get(&m, "key2", &val), "key2 found");
    ASSERT_EQ_SIZE(100, val, "key2 value correct");

    ASSERT_TRUE(!string_hash_map_get(&m, "key3", &val), "key3 not found");

    string_hash_map_free(&m);
}

TEST(hash_map_replace_existing) {
    StringHashMap m;
    string_hash_map_init(&m, 16);

    string_hash_map_put(&m, "dup", 10);
    string_hash_map_put(&m, "dup", 20);

    size_t val;
    ASSERT_TRUE(string_hash_map_get(&m, "dup", &val), "dup found");
    ASSERT_EQ_SIZE(20, val, "dup updated to 20");

    string_hash_map_free(&m);
}

TEST(hash_map_large_insert) {
    StringHashMap m;
    string_hash_map_init(&m, 16);

    char keys[1000][16];
    int i;
    for (i = 0; i < 1000; i++) {
        snprintf(keys[i], sizeof(keys[i]), "key_%04d", i);
        string_hash_map_put(&m, keys[i], (size_t)i);
    }

    /* All entries retrievable. */
    for (i = 0; i < 1000; i++) {
        size_t val;
        ASSERT_TRUE(string_hash_map_get(&m, keys[i], &val),
                    "key found after bulk insert");
        ASSERT_EQ_SIZE((size_t)i, val, "value correct after bulk insert");
    }

    string_hash_map_free(&m);
}

int main(void) {
    RUN_TEST(flatten_linear_network);
    RUN_TEST(flatten_cycle_detection);
    RUN_TEST(flatten_single_node);
    RUN_TEST(flatten_null_guard);
    RUN_TEST(hash_map_empty_get);
    RUN_TEST(hash_map_put_and_get);
    RUN_TEST(hash_map_replace_existing);
    RUN_TEST(hash_map_large_insert);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
