/**
 * @file prof_validate.c
 * @brief Network validation implementation
 */

#include "prof_validate.h"
#include "prof_error.h"
#include "nn_infer_registry.h"
#include "nn_train_registry.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

/**
 * @brief Check if string is empty or NULL
 */
static int is_empty(const char* str) {
    return str == NULL || str[0] == '\0';
}

/**
 * @brief Find subnet index by ID
 *
 * @return Subnet index, or -1 if not found
 */
static int find_subnet_index(const NN_NetworkDef* network, const char* subnet_id) {
    size_t i;
    if (network == NULL || subnet_id == NULL) {
        return -1;
    }

    for (i = 0; i < network->subnet_count; i++) {
        if (network->subnets[i] != NULL &&
            network->subnets[i]->subnet_id != NULL &&
            strcmp(network->subnets[i]->subnet_id, subnet_id) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/**
 * @brief Check if subnet IDs are unique
 */
static int are_subnet_ids_unique(const NN_NetworkDef* network) {
    size_t i, j;
    if (network == NULL || network->subnets == NULL) {
        return 1;
    }

    for (i = 0; i < network->subnet_count; i++) {
        if (network->subnets[i] == NULL || network->subnets[i]->subnet_id == NULL) {
            continue;
        }
        for (j = i + 1; j < network->subnet_count; j++) {
            if (network->subnets[j] != NULL &&
                network->subnets[j]->subnet_id != NULL &&
                strcmp(network->subnets[i]->subnet_id, network->subnets[j]->subnet_id) == 0) {
                return 0;
            }
        }
    }
    return 1;
}

ProfStatus prof_validate_request(
    const ProfGenerateRequest* req,
    ProfErrorBuffer* error
) {
    if (req == NULL) {
        return prof_error_set(error, PROF_STATUS_INVALID_ARGUMENT,
            "Request pointer is NULL");
    }

    if (req->network_def == NULL) {
        return prof_error_set(error, PROF_STATUS_INVALID_ARGUMENT,
            "Network definition is NULL");
    }

    return PROF_STATUS_OK;
}

ProfStatus prof_validate_network_def(
    const NN_NetworkDef* network,
    ProfErrorBuffer* error
) {
    size_t i;

    if (network == NULL) {
        return prof_error_set(error, PROF_STATUS_INVALID_ARGUMENT,
            "Network definition pointer is NULL");
    }

    if (is_empty(network->network_name)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Network name is empty");
    }

    if (network->subnet_count == 0 || network->subnets == NULL) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Network must contain at least one subnet");
    }

    if (!are_subnet_ids_unique(network)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Subnet IDs must be unique, duplicate ID found");
    }

    nn_infer_registry_bootstrap();
    nn_train_registry_bootstrap();

    for (i = 0; i < network->subnet_count; i++) {
        NNSubnetDef* subnet = network->subnets[i];
        ProfStatus st;

        st = prof_validate_subnet(subnet, error);
        if (st != PROF_STATUS_OK) {
            return st;
        }
    }

    return PROF_STATUS_OK;
}

ProfStatus prof_validate_subnet(
    const NNSubnetDef* subnet,
    ProfErrorBuffer* error
) {
    size_t i;

    if (subnet == NULL) {
        return prof_error_set(error, PROF_STATUS_INVALID_ARGUMENT,
            "Subnet pointer is NULL");
    }

    if (is_empty(subnet->subnet_id)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Subnet ID is empty");
    }

    if (is_empty(subnet->subnet_type)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Subnet '%s' has empty type", subnet->subnet_id);
    }

    if (!nn_infer_registry_is_registered(subnet->subnet_type)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Subnet '%s' has unregistered type '%s'",
            subnet->subnet_id, subnet->subnet_type);
    }

    if (subnet->input_layer_size == 0) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Subnet '%s' has invalid input layer size (0)",
            subnet->subnet_id);
    }

    if (subnet->output_layer_size == 0) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Subnet '%s' has invalid output layer size (0)",
            subnet->subnet_id);
    }

    if (subnet->hidden_layer_count > 0 && subnet->hidden_layer_sizes != NULL) {
        for (i = 0; i < subnet->hidden_layer_count; i++) {
            if (subnet->hidden_layer_sizes[i] == 0) {
                return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                    "Subnet '%s' has invalid hidden layer size at index %zu",
                    subnet->subnet_id, i);
            }
        }
    }

    if (subnet->infer_type_config_data == NULL ||
        subnet->infer_type_config_size == 0U ||
        is_empty(subnet->infer_config_header_path) ||
        is_empty(subnet->infer_config_type_name)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Subnet '%s' is missing infer type configuration metadata",
            subnet->subnet_id);
    }

    if (subnet->train_type_config_data == NULL ||
        subnet->train_type_config_size == 0U ||
        is_empty(subnet->train_config_header_path) ||
        is_empty(subnet->train_config_type_name)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Subnet '%s' is missing train type configuration metadata",
            subnet->subnet_id);
    }

    return PROF_STATUS_OK;
}

ProfStatus prof_validate_connections(
    const NN_NetworkDef* network,
    ProfErrorBuffer* error
) {
    size_t i;

    if (network == NULL) {
        return PROF_STATUS_OK;
    }

    for (i = 0; i < network->connection_count; i++) {
        NNConnectionDef* conn = network->connections[i];
        int src_idx, tgt_idx;
        NNSubnetDef* src_subnet;
        NNSubnetDef* tgt_subnet;

        if (conn == NULL) {
            continue;
        }

        if (is_empty(conn->source_subnet_id)) {
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection %zu has empty source subnet ID", i);
        }

        if (is_empty(conn->target_subnet_id)) {
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection %zu has empty target subnet ID", i);
        }

        src_idx = find_subnet_index(network, conn->source_subnet_id);
        if (src_idx < 0) {
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection references non-existent source subnet '%s'",
                conn->source_subnet_id);
        }

        tgt_idx = find_subnet_index(network, conn->target_subnet_id);
        if (tgt_idx < 0) {
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection references non-existent target subnet '%s'",
                conn->target_subnet_id);
        }

        src_subnet = network->subnets[src_idx];
        tgt_subnet = network->subnets[tgt_idx];

        if (conn->source_node_index >= src_subnet->output_layer_size) {
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection from '%s' node %zu exceeds output layer size %zu",
                conn->source_subnet_id, conn->source_node_index,
                src_subnet->output_layer_size);
        }

        if (conn->target_node_index >= tgt_subnet->input_layer_size) {
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection to '%s' node %zu exceeds input layer size %zu",
                conn->target_subnet_id, conn->target_node_index,
                tgt_subnet->input_layer_size);
        }
    }

    return PROF_STATUS_OK;
}

static ProfStatus dfs_visit(
    const NN_NetworkDef* network,
    int* visited,
    int* in_progress,
    size_t node,
    int* has_cycle
) {
    size_t i;
    visited[node] = 1;
    in_progress[node] = 1;

    for (i = 0; i < network->connection_count; i++) {
        NNConnectionDef* conn = network->connections[i];
        int tgt_idx;
        const char* src_id;

        if (conn == NULL) {
            continue;
        }

        src_id = network->subnets[node]->subnet_id;
        if (src_id == NULL || strcmp(src_id, conn->source_subnet_id) != 0) {
            continue;
        }

        tgt_idx = find_subnet_index(network, conn->target_subnet_id);
        if (tgt_idx < 0) {
            continue;
        }

        if (in_progress[tgt_idx]) {
            *has_cycle = 1;
            return PROF_STATUS_CYCLE_DETECTED;
        }

        if (!visited[tgt_idx]) {
            ProfStatus st = dfs_visit(network, visited, in_progress, tgt_idx, has_cycle);
            if (st != PROF_STATUS_OK) {
                return st;
            }
        }
    }

    in_progress[node] = 0;
    return PROF_STATUS_OK;
}

ProfStatus prof_validate_dag(
    const NN_NetworkDef* network,
    ProfErrorBuffer* error
) {
    char* visited_raw;
    char* in_progress_raw;
    size_t alloc_size;
    size_t i;
    int has_cycle = 0;
    ProfStatus st;

    if (network == NULL || network->subnet_count == 0) {
        return PROF_STATUS_OK;
    }

    alloc_size = network->subnet_count * sizeof(int);
    visited_raw = (char*)(uintptr_t)calloc(alloc_size, 1);
    in_progress_raw = (char*)(uintptr_t)calloc(alloc_size, 1);

    if (visited_raw == NULL || in_progress_raw == NULL) {
        free(visited_raw);
        free(in_progress_raw);
        return prof_error_set(error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to allocate memory for DAG validation");
    }

    for (i = 0; i < network->subnet_count && !has_cycle; i++) {
        int* visited = (int*)visited_raw;
        int* in_progress = (int*)in_progress_raw;
        if (!visited[i]) {
            st = dfs_visit(network, visited, in_progress, i, &has_cycle);
            if (st != PROF_STATUS_OK) {
                free(visited_raw);
                free(in_progress_raw);
                if (st == PROF_STATUS_CYCLE_DETECTED) {
                    return prof_error_set(error, PROF_STATUS_CYCLE_DETECTED,
                        "Cycle detected in network topology involving subnet '%s'",
                        network->subnets[i]->subnet_id);
                }
                return st;
            }
        }
    }

    free(visited_raw);
    free(in_progress_raw);

    return PROF_STATUS_OK;
}

ProfStatus prof_validate_all(
    const ProfGenerateRequest* req,
    ProfErrorBuffer* error
) {
    NN_NetworkDef* network;
    ProfStatus st;

    st = prof_validate_request(req, error);
    if (st != PROF_STATUS_OK) {
        return st;
    }

    network = (NN_NetworkDef*)req->network_def;

    st = prof_validate_network_def(network, error);
    if (st != PROF_STATUS_OK) {
        return st;
    }

    st = prof_validate_connections(network, error);
    if (st != PROF_STATUS_OK) {
        return st;
    }

    st = prof_validate_dag(network, error);
    if (st != PROF_STATUS_OK) {
        return st;
    }

    return PROF_STATUS_OK;
}
