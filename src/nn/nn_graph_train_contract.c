/**
 * @file nn_graph_train_contract.c
 * @brief Cache training graph contracts derived from registry entries.
 *
 * Training validation needs to know whether a leaf type can accept externally
 * supplied gradients. This module mirrors the inference-side cache and exposes
 * that derived capability beside the raw training callbacks.
 */

#include "nn_graph_contract.h"
#include "nn_train_registry.h"

#include <string.h>

/**
 * @brief Cached training contract slot keyed by semantic type name.
 */
typedef struct {
    int used;                        /**< Non-zero once the slot has been materialized. */
    char type_name[64];              /**< Stable cache key copied from the registry entry. */
    NNGraphTrainContract contract;   /**< Cached contract served to validation/codegen. */
} NNGraphTrainContractSlot;

/** Fixed-size cache matching the bounded set of compiled training backends. */
static NNGraphTrainContractSlot g_train_contract_slots[32];

/**
 * @brief Resolve a training contract and cache it on first use.
 */
const NNGraphTrainContract* nn_graph_train_contract_find(const char* type_name) {
    const NNTrainRegistryEntry* entry;
    int index;

    /* Empty names never resolve to executable training contracts. */
    if (type_name == 0 || type_name[0] == '\0') {
        return 0;
    }

    /* Bootstrap the training registry before reading entries from it. */
    if (nn_train_registry_bootstrap() != 0) {
        return 0;
    }

    /* Reuse a cached contract whenever validation asks the same question twice. */
    for (index = 0; index < (int)(sizeof(g_train_contract_slots) / sizeof(g_train_contract_slots[0])); ++index) {
        if (g_train_contract_slots[index].used &&
            strcmp(g_train_contract_slots[index].type_name, type_name) == 0) {
            return &g_train_contract_slots[index].contract;
        }
    }

    /* Materialize the contract from the registry entry on the slow path. */
    entry = nn_train_registry_find_entry(type_name);
    if (entry == 0) {
        return 0;
    }

    /* Cache the result so future queries stay simple and allocation-free. */
    for (index = 0; index < (int)(sizeof(g_train_contract_slots) / sizeof(g_train_contract_slots[0])); ++index) {
        if (!g_train_contract_slots[index].used) {
            g_train_contract_slots[index].used = 1;
            (void)strncpy(g_train_contract_slots[index].type_name, type_name, sizeof(g_train_contract_slots[index].type_name) - 1);
            g_train_contract_slots[index].type_name[sizeof(g_train_contract_slots[index].type_name) - 1] = '\0';
            g_train_contract_slots[index].contract.type_name = entry->type_name;
            g_train_contract_slots[index].contract.create = entry->create;
            g_train_contract_slots[index].contract.destroy = entry->destroy;
            g_train_contract_slots[index].contract.step_with_data = entry->step_with_data;
            g_train_contract_slots[index].contract.step_with_output_gradient = entry->step_with_output_gradient;
            g_train_contract_slots[index].contract.get_stats = entry->get_stats;
            g_train_contract_slots[index].contract.supports_graph_backprop =
                entry->step_with_output_gradient != 0 ? 1 : 0;
            return &g_train_contract_slots[index].contract;
        }
    }

    /* Returning NULL here preserves fast-fail behaviour when the cache is full. */
    return 0;
}

/**
 * @brief Report whether a type can backpropagate gradients inside a generated graph.
 */
int nn_graph_train_contract_supports_backprop(const char* type_name) {
    const NNGraphTrainContract* contract = nn_graph_train_contract_find(type_name);
    return (contract != 0 && contract->supports_graph_backprop) ? 1 : 0;
}
