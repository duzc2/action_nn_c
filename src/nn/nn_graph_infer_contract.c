#include "nn_graph_contract.h"
#include "nn_infer_registry.h"

#include <string.h>

typedef struct {
    int used;
    char type_name[64];
    NNGraphInferContract contract;
} NNGraphInferContractSlot;

static NNGraphInferContractSlot g_infer_contract_slots[32];

const NNGraphInferContract* nn_graph_infer_contract_find(const char* type_name) {
    const NNInferRegistryEntry* entry;
    int index;

    if (type_name == 0 || type_name[0] == '\0') {
        return 0;
    }
    if (nn_infer_registry_bootstrap() != 0) {
        return 0;
    }

    for (index = 0; index < (int)(sizeof(g_infer_contract_slots) / sizeof(g_infer_contract_slots[0])); ++index) {
        if (g_infer_contract_slots[index].used &&
            strcmp(g_infer_contract_slots[index].type_name, type_name) == 0) {
            return &g_infer_contract_slots[index].contract;
        }
    }

    entry = nn_infer_registry_find_entry(type_name);
    if (entry == 0) {
        return 0;
    }

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

    return 0;
}

int nn_graph_infer_contract_supports_graph_mode(const char* type_name) {
    const NNGraphInferContract* contract = nn_graph_infer_contract_find(type_name);
    return (contract != 0 && contract->supports_graph_mode) ? 1 : 0;
}
