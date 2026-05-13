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
#include <stdint.h>

/**
 * @brief Flat subnet pointer list
 */
typedef struct {
    NNSubnetDef** items;
    size_t count;
} ProfSubnetList;

/**
 * @brief String-keyed hash map for subnet ID lookups
 *
 * Uses FNV-1a hashing with open addressing and geometric growth.
 * Keys are borrowed (not copied) -- the caller must keep them alive.
 */
typedef struct {
    const char** keys;
    size_t*      values;
    size_t       count;
    size_t       capacity;
} StringHashMap;

void string_hash_map_init(StringHashMap* m, size_t cap);
void string_hash_map_free(StringHashMap* m);
void string_hash_map_put(StringHashMap* m, const char* key, size_t value);
int  string_hash_map_get(const StringHashMap* m, const char* key, size_t* out);

/**
 * @brief Immutable flat network result
 *
 * Bundles the flattened executable leaf list, topological order, and an
 * O(1) ID-to-index hash map. The cycle flag is set when the leaf DAG
 * cannot be topologically sorted.
 */
typedef struct {
    ProfSubnetList leaves;
    size_t*        topological_order;
    size_t*        incoming_counts;
    size_t*        outgoing_counts;
    StringHashMap  id_to_index;
    int            has_cycles;
} FlatNetwork;

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

/**
 * @brief Build a complete FlatNetwork from a network definition
 *
 * Collects executable leaf subnets, computes topological order, detects
 * cycles, and populates the ID-to-index hash map. Returns 0 on success
 * or -1 when argument validation fails. The caller must check
 * flat->has_cycles after a successful return.
 *
 * @param network Network definition
 * @param out     Zero-initialised FlatNetwork to populate
 * @return 0 on success, -1 on invalid arguments
 */
int  prof_flatten_build(const NN_NetworkDef* network, FlatNetwork* out);

/**
 * @brief Release all memory owned by a FlatNetwork
 */
void prof_flatten_free(FlatNetwork* f);

#endif
