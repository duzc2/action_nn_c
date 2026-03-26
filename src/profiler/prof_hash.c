/**
 * @file prof_hash.c
 * @brief Network hash computation implementation
 */

#include "prof_hash.h"
#include "prof_flatten.h"

#include <string.h>

#define FNV_64_PRIME 0x100000001b3ULL
#define FNV_64_OFFSET_BASIS 0xcbf29ce484222325ULL

/**
 * @section prof_hash_design Hashing responsibilities
 *
 * The profiler uses two related but distinct hashes: a broad semantic network
 * hash and a narrower layout hash. Both are derived from deterministic folding
 * helpers in this file so weight compatibility checks remain stable across
 * validation, generation, save, and load stages.
 */

/**
 * @brief Hash a raw byte span with the 64-bit FNV-1a algorithm.
 */
uint64_t prof_fnv1a_hash(const void* data, size_t len) {
    const unsigned char* ptr = (const unsigned char*)data;
    uint64_t hash = FNV_64_OFFSET_BASIS;
    size_t index;

    for (index = 0U; index < len; ++index) {
        hash ^= (uint64_t)ptr[index];
        hash *= FNV_64_PRIME;
    }

    return hash;
}

/**
 * @brief Hash a possibly NULL string in a deterministic way.
 *
 * Treating NULL as a fixed constant keeps absent optional metadata from
 * collapsing into accidental omission of a hash contribution.
 */
static uint64_t hash_string(const char* str) {
    if (str == NULL) {
        return FNV_64_OFFSET_BASIS;
    }
    return prof_fnv1a_hash(str, strlen(str));
}

/**
 * @brief Hash a size_t value without depending on string formatting.
 */
static uint64_t hash_size_t(size_t value) {
    return prof_fnv1a_hash(&value, sizeof(value));
}

/**
 * @brief Fold one 64-bit value into an existing rolling hash.
 *
 * All higher-level helpers eventually funnel through this mix step so the file
 * has one consistent rule for combining structural fields.
 */
static void hash_mix_u64(uint64_t* hash, uint64_t value) {
    if (hash == NULL) {
        return;
    }
    *hash ^= value;
    *hash *= FNV_64_PRIME;
}

/**
 * @brief Mix an opaque configuration blob into a rolling hash.
 *
 * Backend-private config bytes are treated as opaque payloads; the profiler
 * only cares that any byte change affects compatibility decisions.
 */
static void hash_blob(uint64_t* hash, const void* data, size_t length) {
    if (data == NULL || length == 0U) {
        hash_mix_u64(hash, FNV_64_OFFSET_BASIS);
        return;
    }
    hash_mix_u64(hash, prof_fnv1a_hash(data, length));
}

/**
 * @brief Hash per-node activation overrides in declaration order.
 *
 * Declaration order is preserved intentionally so two logically different
 * override lists cannot alias to the same hash by reordering entries.
 */
static void hash_node_overrides(
    uint64_t* hash,
    const NNNodeActivation* overrides,
    size_t count
) {
    size_t index;

    /* Count participates first so empty and truncated arrays stay distinguishable. */
    hash_mix_u64(hash, hash_size_t(count));
    for (index = 0U; index < count; ++index) {
        hash_mix_u64(hash, hash_size_t(overrides[index].node_index));
        hash_mix_u64(hash, (uint64_t)overrides[index].activation);
    }
}

/**
 * @brief Hash a port array together with its activation metadata.
 *
 * Ports affect graph wiring semantics, so their names, directions, widths, and
 * node-level activation overrides all contribute to the semantic network hash.
 */
static void hash_ports(
    uint64_t* hash,
    const NNPortDef* ports,
    size_t count
) {
    size_t index;

    hash_mix_u64(hash, hash_size_t(count));
    for (index = 0U; index < count; ++index) {
        hash_mix_u64(hash, hash_string(ports[index].port_name));
        hash_mix_u64(hash, (uint64_t)ports[index].direction);
        hash_mix_u64(hash, hash_size_t(ports[index].node_count));
        hash_mix_u64(hash, (uint64_t)ports[index].default_activation);
        hash_node_overrides(
            hash,
            ports[index].node_overrides,
            ports[index].node_override_count
        );
    }
}

/**
 * @brief Recursively hash one subnet, including nested children and config blobs.
 *
 * Container subnets and executable leaves share this path so the full nested
 * topology contributes to the network hash even before flattening occurs.
 */
static uint64_t prof_subnet_hash_recursive(const NNSubnetDef* subnet) {
    uint64_t hash = FNV_64_OFFSET_BASIS;
    size_t hidden_index;
    size_t child_index;

    if (subnet == NULL) {
        return hash;
    }

    hash_mix_u64(&hash, hash_string(subnet->subnet_id));
    hash_mix_u64(&hash, hash_string(subnet->subnet_type));
    hash_mix_u64(&hash, hash_size_t(subnet->input_layer_size));
    hash_mix_u64(&hash, hash_size_t(subnet->output_layer_size));
    hash_mix_u64(&hash, hash_size_t(subnet->hidden_layer_count));

    for (hidden_index = 0U; hidden_index < subnet->hidden_layer_count; ++hidden_index) {
        hash_mix_u64(&hash, hash_size_t(subnet->hidden_layer_sizes[hidden_index]));
    }

    hash_mix_u64(&hash, (uint64_t)subnet->default_activation);
    hash_node_overrides(&hash, subnet->node_overrides, subnet->node_override_count);
    hash_ports(&hash, subnet->inputs, subnet->input_count);
    hash_ports(&hash, subnet->outputs, subnet->output_count);

    hash_mix_u64(&hash, hash_string(subnet->infer_config_header_path));
    hash_mix_u64(&hash, hash_string(subnet->infer_config_type_name));
    hash_blob(&hash, subnet->infer_type_config_data, subnet->infer_type_config_size);

    hash_mix_u64(&hash, hash_string(subnet->train_config_header_path));
    hash_mix_u64(&hash, hash_string(subnet->train_config_type_name));
    hash_blob(&hash, subnet->train_type_config_data, subnet->train_type_config_size);

    /* Child count is hashed before child contents so nesting shape remains visible. */
    hash_mix_u64(&hash, hash_size_t(subnet->subnet_count));
    for (child_index = 0U; child_index < subnet->subnet_count; ++child_index) {
        hash_mix_u64(&hash, prof_subnet_hash_recursive(subnet->subnets[child_index]));
    }

    return hash;
}

/**
 * @brief Public wrapper that hashes a single subnet tree.
 */
uint64_t prof_subnet_hash(const NNSubnetDef* subnet) {
    return prof_subnet_hash_recursive(subnet);
}

/**
 * @brief Compute the full network signature hash used for weight compatibility.
 *
 * Every structural field that can change runtime semantics must contribute to
 * this hash so incompatible weights are rejected instead of silently reused.
 */
uint64_t prof_network_hash(const NN_NetworkDef* network) {
    uint64_t hash = FNV_64_OFFSET_BASIS;
    size_t subnet_index;
    size_t connection_index;

    if (network == NULL) {
        return hash;
    }

    hash_mix_u64(&hash, hash_string(network->network_name));
    hash_mix_u64(&hash, hash_string(network->network_version));
    hash_mix_u64(&hash, hash_size_t(network->subnet_count));

    /* Root subnet trees are folded in declaration order to preserve author intent. */
    for (subnet_index = 0U; subnet_index < network->subnet_count; ++subnet_index) {
        hash_mix_u64(&hash, prof_subnet_hash_recursive(network->subnets[subnet_index]));
    }

    hash_mix_u64(&hash, hash_size_t(network->connection_count));
    /* Connections participate after subnet structure so wiring changes perturb the final hash. */
    for (connection_index = 0U; connection_index < network->connection_count; ++connection_index) {
        NNConnectionDef* connection = network->connections[connection_index];
        if (connection == NULL) {
            continue;
        }

        hash_mix_u64(&hash, hash_string(connection->source_subnet_id));
        hash_mix_u64(&hash, hash_string(connection->source_port_name));
        hash_mix_u64(&hash, hash_size_t(connection->source_node_index));
        hash_mix_u64(&hash, hash_string(connection->target_subnet_id));
        hash_mix_u64(&hash, hash_string(connection->target_port_name));
        hash_mix_u64(&hash, hash_size_t(connection->target_node_index));
        hash_mix_u64(&hash, (uint64_t)connection->merge_strategy);
    }

    return hash;
}

/**
 * @brief Estimate the parameter count of a leaf subnet from its layer sizes.
 *
 * Layout hash needs a backend-agnostic estimate of parameter packing. For dense
 * MLP-style leaves, parameter count is sufficient to detect most layout changes
 * that matter to save/load compatibility.
 */
static size_t prof_leaf_param_count(const NNSubnetDef* subnet) {
    size_t parameter_count;
    size_t hidden_index;

    if (subnet == NULL || subnet->output_layer_size == 0U || subnet->input_layer_size == 0U) {
        return 0U;
    }

    parameter_count = 0U;
    /* A single-layer network still has both weights and biases. */
    if (subnet->hidden_layer_count == 0U) {
        parameter_count += subnet->input_layer_size * subnet->output_layer_size;
        parameter_count += subnet->output_layer_size;
        return parameter_count;
    }

    parameter_count += subnet->input_layer_size * subnet->hidden_layer_sizes[0U];
    parameter_count += subnet->hidden_layer_sizes[0U];

    /* Hidden-to-hidden transitions contribute one weight matrix plus one bias vector each. */
    for (hidden_index = 0U; hidden_index + 1U < subnet->hidden_layer_count; ++hidden_index) {
        parameter_count += subnet->hidden_layer_sizes[hidden_index] *
            subnet->hidden_layer_sizes[hidden_index + 1U];
        parameter_count += subnet->hidden_layer_sizes[hidden_index + 1U];
    }

    parameter_count += subnet->hidden_layer_sizes[subnet->hidden_layer_count - 1U] *
        subnet->output_layer_size;
    parameter_count += subnet->output_layer_size;
    return parameter_count;
}

/**
 * @brief Compute a lighter-weight hash focused on parameter layout stability.
 *
 * Layout hash deliberately tracks the fields that affect parameter packing so
 * continued training and checkpoint restore can reject mismatched layouts even
 * when the broader semantic network hash is not being checked directly.
 */
uint64_t prof_layout_hash(const NN_NetworkDef* network) {
    ProfSubnetList leaf_subnets;
    uint64_t hash = FNV_64_OFFSET_BASIS;
    size_t leaf_index;

    if (network == NULL) {
        return hash;
    }

    /* Layout hashing works on executable leaves because save/load is leaf-oriented. */
    if (prof_flatten_collect_leaf_subnets(network, &leaf_subnets) != PROF_STATUS_OK) {
        return hash;
    }

    hash_mix_u64(&hash, hash_string(network->network_name));
    hash_mix_u64(&hash, hash_size_t(leaf_subnets.count));

    /* Only fields that influence parameter packing or typed config are folded here. */
    for (leaf_index = 0U; leaf_index < leaf_subnets.count; ++leaf_index) {
        NNSubnetDef* subnet = leaf_subnets.items[leaf_index];
        hash_mix_u64(&hash, hash_string(subnet->subnet_id));
        hash_mix_u64(&hash, hash_string(subnet->subnet_type));
        hash_mix_u64(&hash, hash_string(subnet->infer_config_header_path));
        hash_mix_u64(&hash, hash_string(subnet->infer_config_type_name));
        hash_blob(&hash, subnet->infer_type_config_data, subnet->infer_type_config_size);
        hash_mix_u64(&hash, hash_string(subnet->train_config_header_path));
        hash_mix_u64(&hash, hash_string(subnet->train_config_type_name));
        hash_blob(&hash, subnet->train_type_config_data, subnet->train_type_config_size);
        hash_mix_u64(&hash, hash_size_t(prof_leaf_param_count(subnet)));
    }

    prof_flatten_free_list(&leaf_subnets);
    return hash;
}
