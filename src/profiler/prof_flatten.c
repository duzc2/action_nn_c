/**
 * @file prof_flatten.c
 * @brief Recursive subnet flattening helpers implementation
 */

#include "prof_flatten.h"
#include "prof_hash.h"

#include <stdlib.h>
#include <string.h>
#include "../utils/error.h"

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
        return ACTION_C_ERR_NULL_POINTER;
    }

    for (subnet_index = 0U; subnet_index < list->count; ++subnet_index) {
        NNSubnetDef* subnet = list->items[subnet_index];
        if (subnet != NULL &&
            subnet->subnet_id != NULL &&
            strcmp(subnet->subnet_id, subnet_id) == 0) {
            return (int)subnet_index;
        }
    }

    return ACTION_C_ERR_NOT_FOUND;
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

/**
 * @section string_hash_map String-keyed hash map
 *
 * O(1) amortised lookups using FNV-1a hashing and open addressing
 * with linear probing. Used by FlatNetwork for subnet ID → index
 * translation, replacing the O(n) linear scan in the simpler helpers.
 */

#define SHM_EMPTY_MARKER ((const char*)(uintptr_t)1)

void string_hash_map_init(StringHashMap* m, size_t cap) {
    size_t i;

    if (m == NULL) {
        return;
    }
    if (cap < 8U) {
        cap = 8U;
    }
    m->keys     = (const char**)calloc(cap, sizeof(const char*));
    m->values   = (size_t*)calloc(cap, sizeof(size_t));
    m->count    = 0U;
    m->capacity = (m->keys != NULL && m->values != NULL) ? cap : 0U;

    /* Mark every slot empty. */
    for (i = 0U; i < m->capacity; ++i) {
        m->keys[i] = SHM_EMPTY_MARKER;
    }
}

void string_hash_map_free(StringHashMap* m) {
    if (m == NULL) {
        return;
    }
    free((void*)m->keys);
    free(m->values);
    m->keys     = NULL;
    m->values   = NULL;
    m->count    = 0U;
    m->capacity = 0U;
}

/**
 * @brief Grow the table to twice its current size and rehash all entries.
 */
static int string_hash_map_grow(StringHashMap* m) {
    const char** old_keys;
    size_t*      old_values;
    size_t       old_capacity;
    size_t       i;

    if (m == NULL || m->capacity == 0U) {
        return -1;
    }

    old_keys       = m->keys;
    old_values     = m->values;
    old_capacity   = m->capacity;

    /* Allocate new storage at double size. */
    m->capacity   *= 2U;
    m->keys       = (const char**)calloc(m->capacity, sizeof(const char*));
    m->values     = (size_t*)calloc(m->capacity, sizeof(size_t));
    if (m->keys == NULL || m->values == NULL) {
        free((void*)m->keys);
        free(m->values);
        m->keys     = old_keys;
        m->values   = old_values;
        m->capacity = old_capacity;
        return -1;
    }

    /* Mark every slot in the new table empty. */
    for (i = 0U; i < m->capacity; ++i) {
        m->keys[i] = SHM_EMPTY_MARKER;
    }

    m->count = 0U;

    /* Rehash all existing entries into the expanded table. */
    for (i = 0U; i < old_capacity; ++i) {
        if (old_keys[i] != NULL && old_keys[i] != SHM_EMPTY_MARKER) {
            string_hash_map_put(m, old_keys[i], old_values[i]);
        }
    }

    free((void*)old_keys);
    free(old_values);
    return 0;
}

void string_hash_map_put(StringHashMap* m, const char* key, size_t value) {
    size_t idx;

    if (m == NULL || key == NULL || m->capacity == 0U) {
        return;
    }

    /* Grow when load factor exceeds 0.7. */
    if (m->count * 10U >= m->capacity * 7U) {
        if (string_hash_map_grow(m) != 0) {
            return;
        }
    }

    /* FNV-1a hash with open-address linear probe. */
    idx = prof_fnv1a_hash(key, strlen(key)) % m->capacity;

    while (m->keys[idx] != SHM_EMPTY_MARKER) {
        if (m->keys[idx] != NULL && strcmp(m->keys[idx], key) == 0) {
            /* Update existing entry. */
            m->values[idx] = value;
            return;
        }
        idx = (idx + 1U) % m->capacity;
    }

    /* Append new entry. */
    m->keys[idx] = key;
    m->values[idx] = value;
    m->count++;
}

int string_hash_map_get(const StringHashMap* m, const char* key, size_t* out) {
    size_t idx;

    if (m == NULL || key == NULL || m->capacity == 0U) {
        return 0;
    }

    idx = prof_fnv1a_hash(key, strlen(key)) % m->capacity;

    while (m->keys[idx] != SHM_EMPTY_MARKER) {
        if (m->keys[idx] != NULL && strcmp(m->keys[idx], key) == 0) {
            if (out != NULL) {
                *out = m->values[idx];
            }
            return 1;
        }
        idx = (idx + 1U) % m->capacity;
    }

    return 0;
}

/**
 * @section flat_network FlatNetwork build / free
 *
 * prof_flatten_build is the single entry point that callers use to obtain
 * a complete immutable view of the executable leaf graph. It internally
 * collects leaves, computes topology, populates the ID-to-index hash map,
 * and sets the cycle flag.
 */

int prof_flatten_build(const NN_NetworkDef* network, FlatNetwork* out) {
    ProfStatus st;
    size_t leaf_index;

    if (network == NULL || out == NULL) {
        return -1;
    }

    /* Clear output fields so partial failures can be cleaned up predictably. */
    (void)memset(out, 0, sizeof(*out));
    out->topological_order    = NULL;
    out->incoming_counts      = NULL;
    out->outgoing_counts      = NULL;

    /* Stage 1: collect only executable leaves. */
    st = prof_flatten_collect_leaf_subnets(network, &out->leaves);
    if (st != PROF_STATUS_OK) {
        prof_flatten_free(out);
        return -1;
    }

    /* Stage 2: build the topological order (Kahn's algorithm). */
    st = prof_flatten_build_leaf_topology(
        network,
        &out->leaves,
        &out->topological_order,
        &out->incoming_counts,
        &out->outgoing_counts
    );

    if (st == PROF_STATUS_CYCLE_DETECTED) {
        out->has_cycles = 1;
        /* Continue despite cycles -- the caller inspects has_cycles. */
    } else if (st != PROF_STATUS_OK) {
        prof_flatten_free(out);
        return -1;
    }

    /* Stage 3: populate the O(1) ID → index hash map. */
    string_hash_map_init(&out->id_to_index, out->leaves.count * 2U);
    for (leaf_index = 0U; leaf_index < out->leaves.count; ++leaf_index) {
        if (out->leaves.items[leaf_index] != NULL &&
            out->leaves.items[leaf_index]->subnet_id != NULL) {
            string_hash_map_put(
                &out->id_to_index,
                out->leaves.items[leaf_index]->subnet_id,
                leaf_index
            );
        }
    }

    return 0;
}

void prof_flatten_free(FlatNetwork* f) {
    if (f == NULL) {
        return;
    }

    prof_flatten_free_list(&f->leaves);
    free(f->topological_order);
    free(f->incoming_counts);
    free(f->outgoing_counts);
    string_hash_map_free(&f->id_to_index);

    f->topological_order = NULL;
    f->incoming_counts   = NULL;
    f->outgoing_counts   = NULL;
    f->has_cycles        = 0;
}
