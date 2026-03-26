/**
 * @file network_def.c
 * @brief Network definition implementation
 */

#include "network_def.h"

#include <stdlib.h>
#include <string.h>

/**
 * @brief Duplicate a NUL-terminated string into profiler-owned memory.
 *
 * The network definition keeps several caller-supplied identifiers and header
 * paths. Copying them when needed prevents later mutations in user code from
 * invalidating the generated model description.
 */
static char* nn_strdup_local(const char* source) {
    char* copy;
    size_t length;

    if (source == NULL) {
        return NULL;
    }

    length = strlen(source) + 1U;
    copy = (char*)malloc(length);
    if (copy == NULL) {
        return NULL;
    }

    (void)memcpy(copy, source, length);
    return copy;
}

/**
 * @brief Duplicate an opaque configuration blob into profiler-owned memory.
 *
 * Type-specific config data is intentionally treated as bytes because the core
 * profiler must not depend on concrete network implementation structs.
 */
static unsigned char* nn_memdup_local(const void* source, size_t size) {
    unsigned char* copy;

    if (source == NULL || size == 0U) {
        return NULL;
    }

    copy = (unsigned char*)malloc(size);
    if (copy == NULL) {
        return NULL;
    }

    (void)memcpy(copy, source, size);
    return copy;
}

/**
 * @brief Release inference-side type metadata currently attached to a subnet.
 *
 * The helper exists so config replacement and final destruction share exactly
 * the same cleanup path. That keeps ownership rules simple and leak resistant.
 */
static void nn_subnet_def_clear_infer_type_config(NNSubnetDef* subnet) {
    if (subnet == NULL) {
        return;
    }

    free(subnet->infer_type_config_data);
    free(subnet->infer_config_header_path);
    free(subnet->infer_config_type_name);
    subnet->infer_type_config_data = NULL;
    subnet->infer_type_config_size = 0U;
    subnet->infer_config_header_path = NULL;
    subnet->infer_config_type_name = NULL;
}

/**
 * @brief Release training-side type metadata currently attached to a subnet.
 */
static void nn_subnet_def_clear_train_type_config(NNSubnetDef* subnet) {
    if (subnet == NULL) {
        return;
    }

    free(subnet->train_type_config_data);
    free(subnet->train_config_header_path);
    free(subnet->train_config_type_name);
    subnet->train_type_config_data = NULL;
    subnet->train_type_config_size = 0U;
    subnet->train_config_header_path = NULL;
    subnet->train_config_type_name = NULL;
}

/**
 * @brief Allocate the root network definition with conservative defaults.
 *
 * The root object starts empty and leaves topology population to the caller,
 * which matches the documented workflow where user code explicitly builds the
 * full graph before invoking the profiler.
 */
NN_NetworkDef* nn_network_def_create(const char* name) {
    NN_NetworkDef* def = (NN_NetworkDef*)malloc(sizeof(NN_NetworkDef));
    if (def == NULL) {
        return NULL;
    }

    def->network_name = name;
    def->network_version = "1.0";
    def->subnets = NULL;
    def->subnet_count = 0;
    def->connections = NULL;
    def->connection_count = 0;
    def->routings = NULL;
    def->routing_count = 0;

    return def;
}

/**
 * @brief Recursively free the full network definition tree.
 *
 * Ownership flows from the root network into subnets and connection objects,
 * so freeing the root must also release every nested subnet, opaque config
 * blob, and auxiliary routing record.
 */
void nn_network_def_free(NN_NetworkDef* def) {
    size_t i;

    if (def == NULL) {
        return;
    }

    if (def->subnets != NULL) {
        for (i = 0; i < def->subnet_count; i++) {
            nn_subnet_def_free(def->subnets[i]);
        }
        free(def->subnets);
    }

    if (def->connections != NULL) {
        for (i = 0; i < def->connection_count; i++) {
            nn_connection_def_free(def->connections[i]);
        }
        free(def->connections);
    }

    if (def->routings != NULL) {
        for (i = 0; i < def->routing_count; i++) {
            free(def->routings[i]);
        }
        free(def->routings);
    }

    free(def);
}

/**
 * @brief Append a top-level subnet to the network definition.
 *
 * The container grows with realloc because the user-facing construction API is
 * append-oriented and the expected topology size is moderate.
 */
void nn_network_def_add_subnet(NN_NetworkDef* network, NNSubnetDef* subnet) {
    NNSubnetDef** new_subnets;
    size_t new_count;

    if (network == NULL || subnet == NULL) {
        return;
    }

    new_count = network->subnet_count + 1;
    new_subnets = (NNSubnetDef**)realloc(
        network->subnets,
        new_count * sizeof(NNSubnetDef*)
    );

    if (new_subnets == NULL) {
        return;
    }

    network->subnets = new_subnets;
    network->subnets[network->subnet_count] = subnet;
    network->subnet_count = new_count;
}

/**
 * @brief Append a connection record to the root network definition.
 */
void nn_network_def_add_connection(NN_NetworkDef* network, NNConnectionDef* connection) {
    NNConnectionDef** new_conns;
    size_t new_count;

    if (network == NULL || connection == NULL) {
        return;
    }

    new_count = network->connection_count + 1;
    new_conns = (NNConnectionDef**)realloc(
        network->connections,
        new_count * sizeof(NNConnectionDef*)
    );

    if (new_conns == NULL) {
        return;
    }

    network->connections = new_conns;
    network->connections[network->connection_count] = connection;
    network->connection_count = new_count;
}

/**
 * @brief Allocate one subnet definition with safe defaults for optional fields.
 *
 * Leaf subnets may later receive layer sizes and type-specific config blobs,
 * while container subnets simply keep their nested child list populated.
 */
NNSubnetDef* nn_subnet_def_create(
    const char* subnet_id,
    const char* subnet_type,
    size_t input_size,
    size_t output_size
) {
    NNSubnetDef* subnet = (NNSubnetDef*)malloc(sizeof(NNSubnetDef));
    if (subnet == NULL) {
        return NULL;
    }

    subnet->subnet_id = subnet_id;
    subnet->subnet_type = subnet_type;
    subnet->input_layer_size = input_size;
    subnet->output_layer_size = output_size;
    subnet->hidden_layer_count = 0;
    subnet->hidden_layer_sizes = NULL;
    subnet->default_activation = NN_ACTIVATION_RELU;
    subnet->node_overrides = NULL;
    subnet->node_override_count = 0;
    subnet->inputs = NULL;
    subnet->input_count = 0;
    subnet->outputs = NULL;
    subnet->output_count = 0;
    subnet->infer_type_config_data = NULL;
    subnet->infer_type_config_size = 0U;
    subnet->infer_config_header_path = NULL;
    subnet->infer_config_type_name = NULL;
    subnet->train_type_config_data = NULL;
    subnet->train_type_config_size = 0U;
    subnet->train_config_header_path = NULL;
    subnet->train_config_type_name = NULL;
    subnet->subnets = NULL;
    subnet->subnet_count = 0;

    return subnet;
}

/**
 * @brief Replace the hidden-layer description owned by a subnet.
 *
 * The setter fully replaces previous storage so callers can rebuild a subnet
 * description incrementally without manually freeing old buffers first.
 */
void nn_subnet_def_set_hidden_layers(
    NNSubnetDef* subnet,
    size_t layer_count,
    const size_t* layer_sizes
) {
    size_t i;

    if (subnet == NULL) {
        return;
    }

    if (subnet->hidden_layer_sizes != NULL) {
        free(subnet->hidden_layer_sizes);
    }

    subnet->hidden_layer_count = layer_count;

    if (layer_count > 0 && layer_sizes != NULL) {
        subnet->hidden_layer_sizes = (size_t*)malloc(
            layer_count * sizeof(size_t)
        );
        if (subnet->hidden_layer_sizes != NULL) {
            for (i = 0; i < layer_count; i++) {
                subnet->hidden_layer_sizes[i] = layer_sizes[i];
            }
        }
    } else {
        subnet->hidden_layer_sizes = NULL;
    }
}

/**
 * @brief Append a child subnet to a container subnet.
 */
void nn_subnet_def_add_subnet(NNSubnetDef* parent, NNSubnetDef* child) {
    NNSubnetDef** new_subnets;
    size_t new_count;

    if (parent == NULL || child == NULL) {
        return;
    }

    new_count = parent->subnet_count + 1U;
    new_subnets = (NNSubnetDef**)realloc(
        parent->subnets,
        new_count * sizeof(NNSubnetDef*)
    );

    if (new_subnets == NULL) {
        return;
    }

    parent->subnets = new_subnets;
    parent->subnets[parent->subnet_count] = child;
    parent->subnet_count = new_count;
}

/**
 * @brief Recursively free one subnet and every resource it owns.
 */
void nn_subnet_def_free(NNSubnetDef* subnet) {
    size_t i;

    if (subnet == NULL) {
        return;
    }

    if (subnet->hidden_layer_sizes != NULL) {
        free(subnet->hidden_layer_sizes);
    }

    if (subnet->node_overrides != NULL) {
        free(subnet->node_overrides);
    }

    if (subnet->inputs != NULL) {
        free(subnet->inputs);
    }

    if (subnet->outputs != NULL) {
        free(subnet->outputs);
    }

    nn_subnet_def_clear_infer_type_config(subnet);
    nn_subnet_def_clear_train_type_config(subnet);

    if (subnet->subnets != NULL) {
        for (i = 0; i < subnet->subnet_count; i++) {
            nn_subnet_def_free((NNSubnetDef*)subnet->subnets[i]);
        }
        free(subnet->subnets);
    }

    free(subnet);
}

/**
 * @brief Allocate one connection record with the default merge strategy.
 *
 * Merge strategy defaults to SUM so callers get a deterministic baseline even
 * before advanced routing policies are configured.
 */
NNConnectionDef* nn_connection_def_create(
    const char* src_subnet,
    const char* src_port,
    size_t src_node,
    const char* tgt_subnet,
    const char* tgt_port,
    size_t tgt_node
) {
    NNConnectionDef* conn = (NNConnectionDef*)malloc(sizeof(NNConnectionDef));
    if (conn == NULL) {
        return NULL;
    }

    conn->source_subnet_id = src_subnet;
    conn->source_port_name = src_port;
    conn->source_node_index = src_node;
    conn->target_subnet_id = tgt_subnet;
    conn->target_port_name = tgt_port;
    conn->target_node_index = tgt_node;
    conn->merge_strategy = NN_MERGE_SUM;

    return conn;
}

/**
 * @brief Release one connection record.
 */
void nn_connection_def_free(NNConnectionDef* conn) {
    if (conn == NULL) {
        return;
    }

    free(conn);
}

/**
 * @brief Copy inference-side type metadata into the subnet definition.
 *
 * The profiler stores the raw bytes plus the header/type names needed to
 * reconstruct the concrete configuration inside generated infer.c.
 */
int nn_subnet_def_set_infer_type_config(
    NNSubnetDef* subnet,
    const void* config_data,
    size_t config_size,
    const char* header_path,
    const char* type_name
) {
    unsigned char* config_copy;
    char* header_copy;
    char* type_copy;

    if (subnet == NULL || config_data == NULL || config_size == 0U ||
        header_path == NULL || header_path[0] == '\0' ||
        type_name == NULL || type_name[0] == '\0') {
        return -1;
    }

    config_copy = nn_memdup_local(config_data, config_size);
    header_copy = nn_strdup_local(header_path);
    type_copy = nn_strdup_local(type_name);
    if (config_copy == NULL || header_copy == NULL || type_copy == NULL) {
        free(config_copy);
        free(header_copy);
        free(type_copy);
        return -1;
    }

    nn_subnet_def_clear_infer_type_config(subnet);
    subnet->infer_type_config_data = config_copy;
    subnet->infer_type_config_size = config_size;
    subnet->infer_config_header_path = header_copy;
    subnet->infer_config_type_name = type_copy;
    return 0;
}

/**
 * @brief Copy training-side type metadata into the subnet definition.
 */
int nn_subnet_def_set_train_type_config(
    NNSubnetDef* subnet,
    const void* config_data,
    size_t config_size,
    const char* header_path,
    const char* type_name
) {
    unsigned char* config_copy;
    char* header_copy;
    char* type_copy;

    if (subnet == NULL || config_data == NULL || config_size == 0U ||
        header_path == NULL || header_path[0] == '\0' ||
        type_name == NULL || type_name[0] == '\0') {
        return -1;
    }

    config_copy = nn_memdup_local(config_data, config_size);
    header_copy = nn_strdup_local(header_path);
    type_copy = nn_strdup_local(type_name);
    if (config_copy == NULL || header_copy == NULL || type_copy == NULL) {
        free(config_copy);
        free(header_copy);
        free(type_copy);
        return -1;
    }

    nn_subnet_def_clear_train_type_config(subnet);
    subnet->train_type_config_data = config_copy;
    subnet->train_type_config_size = config_size;
    subnet->train_config_header_path = header_copy;
    subnet->train_config_type_name = type_copy;
    return 0;
}
