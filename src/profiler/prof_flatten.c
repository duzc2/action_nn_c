/**
 * @file prof_flatten.c
 * @brief Recursive subnet flattening helpers implementation
 */

#include "prof_flatten.h"

#include <stdlib.h>
#include <string.h>

static ProfStatus prof_flatten_append(
    ProfSubnetList* list,
    NNSubnetDef* subnet
) {
    NNSubnetDef** new_items;
    size_t new_count;

    if (list == NULL || subnet == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    new_count = list->count + 1U;
    new_items = (NNSubnetDef**)realloc(list->items, new_count * sizeof(NNSubnetDef*));
    if (new_items == NULL) {
        return PROF_STATUS_INTERNAL_ERROR;
    }

    list->items = new_items;
    list->items[list->count] = subnet;
    list->count = new_count;
    return PROF_STATUS_OK;
}

static ProfStatus prof_flatten_collect_recursive(
    NNSubnetDef* subnet,
    int leaf_only,
    ProfSubnetList* out_list
) {
    ProfStatus st;
    size_t child_index;

    if (subnet == NULL || out_list == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    if (!leaf_only || prof_flatten_is_leaf_subnet(subnet)) {
        st = prof_flatten_append(out_list, subnet);
        if (st != PROF_STATUS_OK) {
            return st;
        }
    }

    for (child_index = 0U; child_index < subnet->subnet_count; ++child_index) {
        st = prof_flatten_collect_recursive(
            subnet->subnets[child_index],
            leaf_only,
            out_list
        );
        if (st != PROF_STATUS_OK) {
            return st;
        }
    }

    return PROF_STATUS_OK;
}

static ProfStatus prof_flatten_collect(
    const NN_NetworkDef* network,
    int leaf_only,
    ProfSubnetList* out_list
) {
    ProfStatus st;
    size_t subnet_index;

    if (out_list == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    out_list->items = NULL;
    out_list->count = 0U;

    if (network == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    for (subnet_index = 0U; subnet_index < network->subnet_count; ++subnet_index) {
        st = prof_flatten_collect_recursive(
            network->subnets[subnet_index],
            leaf_only,
            out_list
        );
        if (st != PROF_STATUS_OK) {
            prof_flatten_free_list(out_list);
            return st;
        }
    }

    return PROF_STATUS_OK;
}

int prof_flatten_is_leaf_subnet(const NNSubnetDef* subnet) {
    return subnet != NULL && subnet->subnet_count == 0U;
}

ProfStatus prof_flatten_collect_all_subnets(
    const NN_NetworkDef* network,
    ProfSubnetList* out_list
) {
    return prof_flatten_collect(network, 0, out_list);
}

ProfStatus prof_flatten_collect_leaf_subnets(
    const NN_NetworkDef* network,
    ProfSubnetList* out_list
) {
    return prof_flatten_collect(network, 1, out_list);
}

void prof_flatten_free_list(ProfSubnetList* list) {
    if (list == NULL) {
        return;
    }

    free(list->items);
    list->items = NULL;
    list->count = 0U;
}

int prof_flatten_find_subnet_index(
    const ProfSubnetList* list,
    const char* subnet_id
) {
    size_t subnet_index;

    if (list == NULL || subnet_id == NULL) {
        return -1;
    }

    for (subnet_index = 0U; subnet_index < list->count; ++subnet_index) {
        NNSubnetDef* subnet = list->items[subnet_index];
        if (subnet != NULL &&
            subnet->subnet_id != NULL &&
            strcmp(subnet->subnet_id, subnet_id) == 0) {
            return (int)subnet_index;
        }
    }

    return -1;
}

ProfStatus prof_flatten_build_leaf_topology(
    const NN_NetworkDef* network,
    const ProfSubnetList* leaf_list,
    size_t** out_order,
    size_t** out_incoming_counts,
    size_t** out_outgoing_counts
) {
    size_t* incoming_counts;
    size_t* outgoing_counts;
    size_t* working_incoming;
    size_t* order;
    size_t* queue;
    size_t queue_head;
    size_t queue_tail;
    size_t produced;
    size_t connection_index;
    size_t leaf_index;

    if (network == NULL || leaf_list == NULL ||
        out_order == NULL || out_incoming_counts == NULL ||
        out_outgoing_counts == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    *out_order = NULL;
    *out_incoming_counts = NULL;
    *out_outgoing_counts = NULL;

    if (leaf_list->count == 0U) {
        return PROF_STATUS_OK;
    }

    incoming_counts = (size_t*)calloc(leaf_list->count, sizeof(size_t));
    outgoing_counts = (size_t*)calloc(leaf_list->count, sizeof(size_t));
    working_incoming = (size_t*)calloc(leaf_list->count, sizeof(size_t));
    order = (size_t*)calloc(leaf_list->count, sizeof(size_t));
    queue = (size_t*)calloc(leaf_list->count, sizeof(size_t));
    if (incoming_counts == NULL || outgoing_counts == NULL ||
        working_incoming == NULL || order == NULL || queue == NULL) {
        free(incoming_counts);
        free(outgoing_counts);
        free(working_incoming);
        free(order);
        free(queue);
        return PROF_STATUS_INTERNAL_ERROR;
    }

    for (connection_index = 0U; connection_index < network->connection_count; ++connection_index) {
        NNConnectionDef* connection = network->connections[connection_index];
        int source_index;
        int target_index;

        if (connection == NULL) {
            continue;
        }

        source_index = prof_flatten_find_subnet_index(leaf_list, connection->source_subnet_id);
        target_index = prof_flatten_find_subnet_index(leaf_list, connection->target_subnet_id);
        if (source_index < 0 || target_index < 0) {
            continue;
        }

        outgoing_counts[(size_t)source_index] += 1U;
        incoming_counts[(size_t)target_index] += 1U;
    }

    (void)memcpy(
        working_incoming,
        incoming_counts,
        leaf_list->count * sizeof(size_t)
    );

    queue_head = 0U;
    queue_tail = 0U;
    for (leaf_index = 0U; leaf_index < leaf_list->count; ++leaf_index) {
        if (working_incoming[leaf_index] == 0U) {
            queue[queue_tail] = leaf_index;
            queue_tail += 1U;
        }
    }

    produced = 0U;
    while (queue_head < queue_tail) {
        size_t current = queue[queue_head];
        queue_head += 1U;
        order[produced] = current;
        produced += 1U;

        for (connection_index = 0U; connection_index < network->connection_count; ++connection_index) {
            NNConnectionDef* connection = network->connections[connection_index];
            int source_index;
            int target_index;

            if (connection == NULL) {
                continue;
            }

            source_index = prof_flatten_find_subnet_index(leaf_list, connection->source_subnet_id);
            target_index = prof_flatten_find_subnet_index(leaf_list, connection->target_subnet_id);
            if (source_index < 0 || target_index < 0) {
                continue;
            }
            if ((size_t)source_index != current) {
                continue;
            }

            if (working_incoming[(size_t)target_index] > 0U) {
                working_incoming[(size_t)target_index] -= 1U;
                if (working_incoming[(size_t)target_index] == 0U) {
                    queue[queue_tail] = (size_t)target_index;
                    queue_tail += 1U;
                }
            }
        }
    }

    free(working_incoming);
    free(queue);

    if (produced != leaf_list->count) {
        free(incoming_counts);
        free(outgoing_counts);
        free(order);
        return PROF_STATUS_CYCLE_DETECTED;
    }

    *out_order = order;
    *out_incoming_counts = incoming_counts;
    *out_outgoing_counts = outgoing_counts;
    return PROF_STATUS_OK;
}
