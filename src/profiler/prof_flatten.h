/**
 * @file prof_flatten.h
 * @brief Recursive subnet flattening helpers
 *
 * Collects nested subnet trees into flat leaf/all-subnet views.
 * These helpers are shared by validation, hashing, and code generation.
 */

#ifndef PROF_FLATTEN_H
#define PROF_FLATTEN_H

#include "network_def.h"
#include "profiler_types.h"

#include <stddef.h>

/**
 * @brief Flat subnet pointer list
 */
typedef struct {
    NNSubnetDef** items;
    size_t count;
} ProfSubnetList;

/**
 * @brief Return 1 if subnet has no child subnet, otherwise 0
 */
int prof_flatten_is_leaf_subnet(const NNSubnetDef* subnet);

/**
 * @brief Collect every subnet, including containers and leaves
 */
ProfStatus prof_flatten_collect_all_subnets(
    const NN_NetworkDef* network,
    ProfSubnetList* out_list
);

/**
 * @brief Collect only executable leaf subnets
 */
ProfStatus prof_flatten_collect_leaf_subnets(
    const NN_NetworkDef* network,
    ProfSubnetList* out_list
);

/**
 * @brief Release list storage allocated by collect helpers
 */
void prof_flatten_free_list(ProfSubnetList* list);

/**
 * @brief Find flat-list index by subnet id
 *
 * @return Non-negative index on success, -1 when not found
 */
int prof_flatten_find_subnet_index(
    const ProfSubnetList* list,
    const char* subnet_id
);

/**
 * @brief Build topological order across the flattened leaf graph
 *
 * The returned arrays are allocated with malloc() and must be freed
 * by the caller.
 */
ProfStatus prof_flatten_build_leaf_topology(
    const NN_NetworkDef* network,
    const ProfSubnetList* leaf_list,
    size_t** out_order,
    size_t** out_incoming_counts,
    size_t** out_outgoing_counts
);

#endif
