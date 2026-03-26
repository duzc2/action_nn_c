/**
 * @file prof_validate.c
 * @brief Network validation implementation
 */

#include "prof_validate.h"
#include "prof_error.h"
#include "prof_flatten.h"
#include "nn_graph_contract.h"

#include <stdlib.h>
#include <string.h>

/**
 * @brief Check if string is empty or NULL
 */
static int is_empty(const char* str) {
    return str == NULL || str[0] == '\0';
}

/**
 * @brief Check whether subnet is a pure structural container
 */
static int is_container_subnet(const NNSubnetDef* subnet) {
    return subnet != NULL && subnet->subnet_count > 0U;
}

/**
 * @brief Check if all subnet identifiers are globally unique
 */
static int are_subnet_ids_unique(const ProfSubnetList* subnet_list) {
    size_t left_index;
    size_t right_index;

    if (subnet_list == NULL) {
        return 1;
    }

    for (left_index = 0U; left_index < subnet_list->count; ++left_index) {
        NNSubnetDef* left = subnet_list->items[left_index];
        if (left == NULL || left->subnet_id == NULL) {
            continue;
        }

        for (right_index = left_index + 1U; right_index < subnet_list->count; ++right_index) {
            NNSubnetDef* right = subnet_list->items[right_index];
            if (right != NULL &&
                right->subnet_id != NULL &&
                strcmp(left->subnet_id, right->subnet_id) == 0) {
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

ProfStatus prof_validate_subnet(
    const NNSubnetDef* subnet,
    ProfErrorBuffer* error
) {
    size_t hidden_index;
    size_t child_index;

    if (subnet == NULL) {
        return prof_error_set(error, PROF_STATUS_INVALID_ARGUMENT,
            "Subnet pointer is NULL");
    }

    if (is_empty(subnet->subnet_id)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Subnet ID is empty");
    }

    if (is_container_subnet(subnet)) {
        if (!is_empty(subnet->subnet_type)) {
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Container subnet '%s' must not declare executable type metadata",
                subnet->subnet_id);
        }
        if (subnet->input_layer_size != 0U || subnet->output_layer_size != 0U ||
            subnet->hidden_layer_count != 0U) {
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Container subnet '%s' must not declare layer sizes",
                subnet->subnet_id);
        }
        if (subnet->infer_type_config_data != NULL ||
            subnet->infer_type_config_size != 0U ||
            subnet->infer_config_header_path != NULL ||
            subnet->infer_config_type_name != NULL ||
            subnet->train_type_config_data != NULL ||
            subnet->train_type_config_size != 0U ||
            subnet->train_config_header_path != NULL ||
            subnet->train_config_type_name != NULL) {
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Container subnet '%s' must not declare type configuration blobs",
                subnet->subnet_id);
        }

        for (child_index = 0U; child_index < subnet->subnet_count; ++child_index) {
            ProfStatus child_status = prof_validate_subnet(subnet->subnets[child_index], error);
            if (child_status != PROF_STATUS_OK) {
                return child_status;
            }
        }
        return PROF_STATUS_OK;
    }

    if (is_empty(subnet->subnet_type)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Leaf subnet '%s' has empty type",
            subnet->subnet_id);
    }

    if (nn_graph_infer_contract_find(subnet->subnet_type) == NULL ||
        nn_graph_train_contract_find(subnet->subnet_type) == NULL) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Leaf subnet '%s' has unregistered type '%s'",
            subnet->subnet_id, subnet->subnet_type);
    }

    if (subnet->input_layer_size == 0U) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Leaf subnet '%s' has invalid input layer size (0)",
            subnet->subnet_id);
    }

    if (subnet->output_layer_size == 0U) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Leaf subnet '%s' has invalid output layer size (0)",
            subnet->subnet_id);
    }

    if (subnet->hidden_layer_count > 0U && subnet->hidden_layer_sizes == NULL) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Leaf subnet '%s' is missing hidden layer size array",
            subnet->subnet_id);
    }

    for (hidden_index = 0U; hidden_index < subnet->hidden_layer_count; ++hidden_index) {
        if (subnet->hidden_layer_sizes[hidden_index] == 0U) {
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Leaf subnet '%s' has invalid hidden layer size at index %zu",
                subnet->subnet_id, hidden_index);
        }
    }

    if (subnet->infer_type_config_data == NULL ||
        subnet->infer_type_config_size == 0U ||
        is_empty(subnet->infer_config_header_path) ||
        is_empty(subnet->infer_config_type_name)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Leaf subnet '%s' is missing infer type configuration metadata",
            subnet->subnet_id);
    }

    if (subnet->train_type_config_data == NULL ||
        subnet->train_type_config_size == 0U ||
        is_empty(subnet->train_config_header_path) ||
        is_empty(subnet->train_config_type_name)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Leaf subnet '%s' is missing train type configuration metadata",
            subnet->subnet_id);
    }

    return PROF_STATUS_OK;
}

ProfStatus prof_validate_network_def(
    const NN_NetworkDef* network,
    ProfErrorBuffer* error
) {
    ProfSubnetList all_subnets;
    ProfSubnetList leaf_subnets;
    ProfStatus status;
    size_t leaf_index;

    if (network == NULL) {
        return prof_error_set(error, PROF_STATUS_INVALID_ARGUMENT,
            "Network definition pointer is NULL");
    }

    if (is_empty(network->network_name)) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Network name is empty");
    }

    if (network->subnet_count == 0U || network->subnets == NULL) {
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Network must contain at least one subnet");
    }

    status = prof_flatten_collect_all_subnets(network, &all_subnets);
    if (status != PROF_STATUS_OK) {
        return prof_error_set(error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to flatten subnet tree for validation");
    }

    status = prof_flatten_collect_leaf_subnets(network, &leaf_subnets);
    if (status != PROF_STATUS_OK) {
        prof_flatten_free_list(&all_subnets);
        return prof_error_set(error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to collect executable leaf subnets");
    }

    if (leaf_subnets.count == 0U) {
        prof_flatten_free_list(&leaf_subnets);
        prof_flatten_free_list(&all_subnets);
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Network must contain at least one executable leaf subnet");
    }

    if (!are_subnet_ids_unique(&all_subnets)) {
        prof_flatten_free_list(&leaf_subnets);
        prof_flatten_free_list(&all_subnets);
        return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
            "Subnet IDs must be unique across the full nested graph");
    }

    for (leaf_index = 0U; leaf_index < network->subnet_count; ++leaf_index) {
        status = prof_validate_subnet(network->subnets[leaf_index], error);
        if (status != PROF_STATUS_OK) {
            prof_flatten_free_list(&leaf_subnets);
            prof_flatten_free_list(&all_subnets);
            return status;
        }
    }

    if (leaf_subnets.count > 1U) {
        for (leaf_index = 0U; leaf_index < leaf_subnets.count; ++leaf_index) {
            NNSubnetDef* leaf = leaf_subnets.items[leaf_index];
            if (leaf == NULL) {
                prof_flatten_free_list(&leaf_subnets);
                prof_flatten_free_list(&all_subnets);
                return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                    "Flattened leaf subnet list contains NULL entry");
            }
            if (!nn_graph_infer_contract_supports_graph_mode(leaf->subnet_type)) {
                prof_flatten_free_list(&leaf_subnets);
                prof_flatten_free_list(&all_subnets);
                return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                    "Leaf subnet '%s' type '%s' does not support graph forward execution",
                    leaf->subnet_id,
                    leaf->subnet_type);
            }
            if (!nn_graph_train_contract_supports_backprop(leaf->subnet_type)) {
                prof_flatten_free_list(&leaf_subnets);
                prof_flatten_free_list(&all_subnets);
                return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                    "Leaf subnet '%s' type '%s' does not support graph backpropagation",
                    leaf->subnet_id,
                    leaf->subnet_type);
            }
        }
    }

    prof_flatten_free_list(&leaf_subnets);
    prof_flatten_free_list(&all_subnets);
    return PROF_STATUS_OK;
}

ProfStatus prof_validate_connections(
    const NN_NetworkDef* network,
    ProfErrorBuffer* error
) {
    ProfSubnetList leaf_subnets;
    ProfStatus status;
    size_t connection_index;

    if (network == NULL) {
        return PROF_STATUS_OK;
    }

    status = prof_flatten_collect_leaf_subnets(network, &leaf_subnets);
    if (status != PROF_STATUS_OK) {
        return prof_error_set(error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to flatten leaf subnets for connection validation");
    }

    for (connection_index = 0U; connection_index < network->connection_count; ++connection_index) {
        NNConnectionDef* connection = network->connections[connection_index];
        int source_index;
        int target_index;
        NNSubnetDef* source_subnet;
        NNSubnetDef* target_subnet;

        if (connection == NULL) {
            continue;
        }

        if (is_empty(connection->source_subnet_id)) {
            prof_flatten_free_list(&leaf_subnets);
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection %zu has empty source subnet ID", connection_index);
        }
        if (is_empty(connection->target_subnet_id)) {
            prof_flatten_free_list(&leaf_subnets);
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection %zu has empty target subnet ID", connection_index);
        }

        source_index = prof_flatten_find_subnet_index(&leaf_subnets, connection->source_subnet_id);
        if (source_index < 0) {
            prof_flatten_free_list(&leaf_subnets);
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection references non-existent source leaf subnet '%s'",
                connection->source_subnet_id);
        }

        target_index = prof_flatten_find_subnet_index(&leaf_subnets, connection->target_subnet_id);
        if (target_index < 0) {
            prof_flatten_free_list(&leaf_subnets);
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection references non-existent target leaf subnet '%s'",
                connection->target_subnet_id);
        }

        source_subnet = leaf_subnets.items[(size_t)source_index];
        target_subnet = leaf_subnets.items[(size_t)target_index];
        if (source_subnet == NULL || target_subnet == NULL) {
            prof_flatten_free_list(&leaf_subnets);
            return prof_error_set(error, PROF_STATUS_INTERNAL_ERROR,
                "Connection flattening produced NULL subnet pointer");
        }

        if (connection->source_node_index >= source_subnet->output_layer_size) {
            prof_flatten_free_list(&leaf_subnets);
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection from '%s' node %zu exceeds output layer size %zu",
                connection->source_subnet_id,
                connection->source_node_index,
                source_subnet->output_layer_size);
        }

        if (connection->target_node_index >= target_subnet->input_layer_size) {
            prof_flatten_free_list(&leaf_subnets);
            return prof_error_set(error, PROF_STATUS_VALIDATION_FAILED,
                "Connection to '%s' node %zu exceeds input layer size %zu",
                connection->target_subnet_id,
                connection->target_node_index,
                target_subnet->input_layer_size);
        }
    }

    prof_flatten_free_list(&leaf_subnets);
    return PROF_STATUS_OK;
}

ProfStatus prof_validate_dag(
    const NN_NetworkDef* network,
    ProfErrorBuffer* error
) {
    ProfSubnetList leaf_subnets;
    size_t* topo_order;
    size_t* incoming_counts;
    size_t* outgoing_counts;
    ProfStatus status;

    if (network == NULL) {
        return PROF_STATUS_OK;
    }

    status = prof_flatten_collect_leaf_subnets(network, &leaf_subnets);
    if (status != PROF_STATUS_OK) {
        return prof_error_set(error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to flatten leaf subnets for DAG validation");
    }

    topo_order = NULL;
    incoming_counts = NULL;
    outgoing_counts = NULL;
    status = prof_flatten_build_leaf_topology(
        network,
        &leaf_subnets,
        &topo_order,
        &incoming_counts,
        &outgoing_counts
    );
    free(topo_order);
    free(incoming_counts);
    free(outgoing_counts);
    prof_flatten_free_list(&leaf_subnets);

    if (status == PROF_STATUS_CYCLE_DETECTED) {
        return prof_error_set(error, PROF_STATUS_CYCLE_DETECTED,
            "Cycle detected in flattened leaf subnet topology");
    }
    if (status != PROF_STATUS_OK) {
        return prof_error_set(error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to compute flattened leaf topology");
    }

    return PROF_STATUS_OK;
}

ProfStatus prof_validate_all(
    const ProfGenerateRequest* req,
    ProfErrorBuffer* error
) {
    NN_NetworkDef* network;
    ProfStatus status;

    status = prof_validate_request(req, error);
    if (status != PROF_STATUS_OK) {
        return status;
    }

    network = (NN_NetworkDef*)req->network_def;

    status = prof_validate_network_def(network, error);
    if (status != PROF_STATUS_OK) {
        return status;
    }

    status = prof_validate_connections(network, error);
    if (status != PROF_STATUS_OK) {
        return status;
    }

    status = prof_validate_dag(network, error);
    if (status != PROF_STATUS_OK) {
        return status;
    }

    return PROF_STATUS_OK;
}
