/**
 * @file prof_flatten.c
 * @brief Recursive subnet flattening helpers implementation
 */

#include "prof_flatten.h"

#include <stdlib.h>
#include <string.h>

/**
 * @section prof_flatten_design Flattened graph helper responsibilities
 *
 * Validation and code generation both need temporary linear views over a nested
 * subnet tree. This file centralizes that work so every later stage observes
 * the same leaf ordering, the same connection interpretation, and the same DAG
 * rules. The returned lists own only their pointer arrays; the caller retains
 * ownership of the actual network definition objects.
 */

/**
 * @brief Append one subnet pointer to a flat working list.
 *
 * Flattened lists are temporary views over caller-owned subnet objects, so the
 * helper only manages the pointer array and never clones the subnet itself.
 */
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

/**
 * @brief Walk a nested subnet tree and append matching nodes to a flat list.
 *
 * The recursion deliberately visits parents before children so the flattened
 * order remains intuitive for diagnostics, even though executable scheduling is
 * later derived from explicit connection topology rather than traversal order.
 */
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

    /* Append either every node or only executable leaves, depending on the caller. */
    if (!leaf_only || prof_flatten_is_leaf_subnet(subnet)) {
        st = prof_flatten_append(out_list, subnet);
        if (st != PROF_STATUS_OK) {
            return st;
        }
    }

    /* Recurse into children after processing the current node to keep ownership simple. */
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

/**
 * @brief Build either the all-subnet view or the executable-leaf view.
 *
 * The public collectors differ only by the leaf_only flag, so both funnel into
 * this helper to guarantee identical initialization and cleanup behaviour.
 */
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

    /* Root-level subnets are flattened independently so partial failure can cleanly abort. */
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

/**
 * @brief Return non-zero only for executable leaf subnets.
 */
int prof_flatten_is_leaf_subnet(const NNSubnetDef* subnet) {
    return subnet != NULL && subnet->subnet_count == 0U;
}

/**
 * @brief Collect every subnet, including structural containers.
 */
ProfStatus prof_flatten_collect_all_subnets(
    const NN_NetworkDef* network,
    ProfSubnetList* out_list
) {
    return prof_flatten_collect(network, 0, out_list);
}

/**
 * @brief Collect only executable leaves that can appear in the generated graph.
 */
ProfStatus prof_flatten_collect_leaf_subnets(
    const NN_NetworkDef* network,
    ProfSubnetList* out_list
) {
    return prof_flatten_collect(network, 1, out_list);
}

/**
 * @brief Release the pointer array owned by a flattened subnet list.
 */
void prof_flatten_free_list(ProfSubnetList* list) {
    if (list == NULL) {
        return;
    }

    free(list->items);
    list->items = NULL;
    list->count = 0U;
}

/**
 * @brief Find the flat-list index associated with a subnet identifier.
 */
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

/**
 * @brief Build topological metadata for the flattened executable leaf graph.
 *
 * The helper computes incoming and outgoing counts as well as a Kahn-style
 * topological order. The result is reused by validation, metadata emission,
 * and generated execution code so every stage sees the same leaf ordering.
 * Connections whose endpoints do not resolve to executable leaves are skipped
 * here because request-level validation reports them separately with richer
 * diagnostics; topology building assumes it is operating on an already checked
 * network definition and focuses only on scheduling data.
 */
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

    /* Allocate one working array per topology view so cycle detection stays explicit. */
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

    /* First pass: compute in-degree and out-degree for every executable leaf. */
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
            /* Non-leaf or unresolved endpoints are validated elsewhere and ignored here. */
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
    /* Seed Kahn's queue with source leaves that have no unresolved predecessors. */
    for (leaf_index = 0U; leaf_index < leaf_list->count; ++leaf_index) {
        if (working_incoming[leaf_index] == 0U) {
            queue[queue_tail] = leaf_index;
            queue_tail += 1U;
        }
    }

    produced = 0U;
    /* Repeatedly emit ready leaves and relax outgoing edges until the queue drains. */
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

            /* Decrement the remaining predecessor count for each outgoing edge. */
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

    /* Any leaf left unproduced belongs to a cycle or unresolved strongly connected set. */
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
