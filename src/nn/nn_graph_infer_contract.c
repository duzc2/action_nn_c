/**
 * @file nn_graph_infer_contract.c
 * @brief Cache inference graph contracts derived from registry entries.
 *
 * Validation queries graph capabilities repeatedly while traversing the
 * flattened topology. Caching the derived contract avoids rebuilding the same
 * structure on every lookup and keeps the profiler path deterministic.
 */

#include "nn_graph_contract.h"
#include "nn_infer_registry.h"

#include <string.h>

/**
 * @brief One cache slot that mirrors a single registry entry plus derived flags.
 */
typedef struct {
    int used;                        /**< Non-zero when this slot already contains a cached contract. */
    char type_name[64];              /**< Stable lookup key copied from the registry entry. */
    NNGraphInferContract contract;   /**< Cached contract exposed to validation/codegen. */
} NNGraphInferContractSlot;

/** Fixed-size cache because only a bounded set of CMake-enabled types exists. */
static NNGraphInferContractSlot g_infer_contract_slots[32];

/**
 * @brief Resolve an inference contract and cache it on first use.
 */
const NNGraphInferContract* nn_graph_infer_contract_find(const char* type_name) {
    const NNInferRegistryEntry* entry;
    int index;

    /* Empty type names can never map to executable graph contracts. */
    if (type_name == 0 || type_name[0] == '\0') {
        return 0;
    }

    /* Ensure compile-time enabled backends have populated the registry. */
    if (nn_infer_registry_bootstrap() != 0) {
        return 0;
    }

    /* Fast path: return the previously materialized cache entry. */
    for (index = 0; index < (int)(sizeof(g_infer_contract_slots) / sizeof(g_infer_contract_slots[0])); ++index) {
        if (g_infer_contract_slots[index].used &&
            strcmp(g_infer_contract_slots[index].type_name, type_name) == 0) {
            return &g_infer_contract_slots[index].contract;
        }
    }

    /* Slow path: read the registry entry and derive graph-specific metadata. */
    entry = nn_infer_registry_find_entry(type_name);
    if (entry == 0) {
        return 0;
    }

    /* Cache the derived contract so subsequent lookups stay allocation-free. */
    for (index = 0; index < (int)(sizeof(g_infer_contract_slots) / sizeof(g_infer_contract_slots[0])); ++index) {
        if (!g_infer_contract_slots[index].used) {
            g_infer_contract_slots[index].used = 1;
            (void)strncpy(g_infer_contract_slots[index].type_name, type_name, sizeof(g_infer_contract_slots[index].type_name) - 1);
            g_infer_contract_slots[index].type_name[sizeof(g_infer_contract_slots[index].type_name) - 1] = '\0';
            g_infer_contract_slots[index].contract.type_name = entry->type_name;
            g_infer_contract_slots[index].contract.create = entry->create;
            g_infer_contract_slots[index].contract.destroy = entry->destroy;
            g_infer_contract_slots[index].contract.auto_run = entry->auto_run;
            g_infer_contract_slots[index].contract.graph_run = entry->graph_run;
            g_infer_contract_slots[index].contract.load_weights = entry->load_weights;
            g_infer_contract_slots[index].contract.save_weights = entry->save_weights;
            g_infer_contract_slots[index].contract.supports_graph_mode =
                entry->graph_run != 0 ? 1 : 0;
            return &g_infer_contract_slots[index].contract;
        }
    }

    /* A full cache means the configured type surface exceeded the static budget. */
    return 0;
}

/**
 * @brief Report whether a type exposes the graph-run hook needed by composed graphs.
 */
int nn_graph_infer_contract_supports_graph_mode(const char* type_name) {
    const NNGraphInferContract* contract = nn_graph_infer_contract_find(type_name);
    return (contract != 0 && contract->supports_graph_mode) ? 1 : 0;
}
