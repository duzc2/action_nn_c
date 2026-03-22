/**
 * @file prof_hash.c
 * @brief Network hash computation implementation
 */

#include "prof_hash.h"

#include <string.h>

/**
 * @brief FNV-1a 64-bit hash constants
 */
#define FNV_64_PRIME 0x100000001b3ULL
#define FNV_64_OFFSET_BASIS 0xcbf29ce484222325ULL

uint64_t prof_fnv1a_hash(const void* data, size_t len) {
    const unsigned char* ptr = (const unsigned char*)data;
    uint64_t hash = FNV_64_OFFSET_BASIS;
    size_t i;

    for (i = 0; i < len; i++) {
        hash ^= (uint64_t)ptr[i];
        hash *= FNV_64_PRIME;
    }

    return hash;
}

static uint64_t hash_string(const char* str) {
    if (str == NULL) {
        return FNV_64_OFFSET_BASIS;
    }
    return prof_fnv1a_hash(str, strlen(str));
}

static uint64_t hash_size_t(size_t val) {
    return prof_fnv1a_hash(&val, sizeof(val));
}

uint64_t prof_subnet_hash(const NNSubnetDef* subnet) {
    uint64_t hash = FNV_64_OFFSET_BASIS;
    size_t i;

    if (subnet == NULL) {
        return hash;
    }

    hash ^= hash_string(subnet->subnet_id);
    hash *= FNV_64_PRIME;

    hash ^= hash_string(subnet->subnet_type);
    hash *= FNV_64_PRIME;

    hash ^= hash_size_t(subnet->input_layer_size);
    hash *= FNV_64_PRIME;

    hash ^= hash_size_t(subnet->output_layer_size);
    hash *= FNV_64_PRIME;

    hash ^= hash_size_t(subnet->hidden_layer_count);
    hash *= FNV_64_PRIME;

    if (subnet->hidden_layer_sizes != NULL) {
        for (i = 0; i < subnet->hidden_layer_count; i++) {
            hash ^= hash_size_t(subnet->hidden_layer_sizes[i]);
            hash *= FNV_64_PRIME;
        }
    }

    hash ^= (uint64_t)subnet->default_activation;
    hash *= FNV_64_PRIME;

    return hash;
}

uint64_t prof_network_hash(const NN_NetworkDef* network) {
    uint64_t hash = FNV_64_OFFSET_BASIS;
    size_t i;

    if (network == NULL) {
        return hash;
    }

    hash ^= hash_string(network->network_name);
    hash *= FNV_64_PRIME;

    hash ^= hash_string(network->network_version);
    hash *= FNV_64_PRIME;

    hash ^= hash_size_t(network->subnet_count);
    hash *= FNV_64_PRIME;

    for (i = 0; i < network->subnet_count; i++) {
        uint64_t subnet_hash = prof_subnet_hash(network->subnets[i]);
        hash ^= subnet_hash;
        hash *= FNV_64_PRIME;
    }

    hash ^= hash_size_t(network->connection_count);
    hash *= FNV_64_PRIME;

    for (i = 0; i < network->connection_count; i++) {
        NNConnectionDef* conn = network->connections[i];
        if (conn != NULL) {
            hash ^= hash_string(conn->source_subnet_id);
            hash *= FNV_64_PRIME;
            hash ^= hash_string(conn->target_subnet_id);
            hash *= FNV_64_PRIME;
            hash ^= hash_size_t(conn->source_node_index);
            hash *= FNV_64_PRIME;
            hash ^= hash_size_t(conn->target_node_index);
            hash *= FNV_64_PRIME;
        }
    }

    return hash;
}

uint64_t prof_layout_hash(const NN_NetworkDef* network) {
    uint64_t hash = FNV_64_OFFSET_BASIS;
    size_t i;

    if (network == NULL) {
        return hash;
    }

    for (i = 0; i < network->subnet_count; i++) {
        NNSubnetDef* subnet = network->subnets[i];
        size_t param_count;

        if (subnet == NULL) {
            continue;
        }

        hash ^= hash_string(subnet->subnet_id);
        hash *= FNV_64_PRIME;

        param_count = 0;

        if (subnet->input_layer_size > 0 && subnet->hidden_layer_count > 0) {
            size_t first_hidden = subnet->hidden_layer_sizes != NULL ?
                subnet->hidden_layer_sizes[0] : 0;
            param_count += subnet->input_layer_size * first_hidden;
        }

        for (i = 0; i < subnet->hidden_layer_count - 1; i++) {
            size_t curr = subnet->hidden_layer_sizes[i];
            size_t next = subnet->hidden_layer_sizes[i + 1];
            param_count += curr * next;
        }

        if (subnet->hidden_layer_count > 0) {
            size_t last_hidden = subnet->hidden_layer_sizes != NULL ?
                subnet->hidden_layer_sizes[subnet->hidden_layer_count - 1] : 0;
            param_count += last_hidden * subnet->output_layer_size;
        } else {
            param_count += subnet->input_layer_size * subnet->output_layer_size;
        }

        hash ^= hash_size_t(param_count);
        hash *= FNV_64_PRIME;
    }

    return hash;
}
