#include "nn_graph_contract.h"
#include "nn_train_registry.h"

#include <string.h>

typedef struct {
    int used;
    char type_name[64];
    NNGraphTrainContract contract;
} NNGraphTrainContractSlot;

static NNGraphTrainContractSlot g_train_contract_slots[32];

const NNGraphTrainContract* nn_graph_train_contract_find(const char* type_name) {
    const NNTrainRegistryEntry* entry;
    int index;

    if (type_name == 0 || type_name[0] == '\0') {
        return 0;
    }
    if (nn_train_registry_bootstrap() != 0) {
        return 0;
    }

    for (index = 0; index < (int)(sizeof(g_train_contract_slots) / sizeof(g_train_contract_slots[0])); ++index) {
        if (g_train_contract_slots[index].used &&
            strcmp(g_train_contract_slots[index].type_name, type_name) == 0) {
            return &g_train_contract_slots[index].contract;
        }
    }

    entry = nn_train_registry_find_entry(type_name);
    if (entry == 0) {
        return 0;
    }

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

    return 0;
}

int nn_graph_train_contract_supports_backprop(const char* type_name) {
    const NNGraphTrainContract* contract = nn_graph_train_contract_find(type_name);
    return (contract != 0 && contract->supports_graph_backprop) ? 1 : 0;
}
