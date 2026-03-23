#include "nn_infer_registry.h"

#include <string.h>

typedef struct {
    int used;
    char type_name[64];
    const NNInferRegistryEntry* entry;
} NNInferRegistrySlot;

static NNInferRegistrySlot g_slots[32];
static int g_bootstrapped = 0;
static int g_bootstrap_failed = 0;

static int is_empty(const char* text) {
    return text == 0 || text[0] == '\0';
}

int nn_infer_registry_register(const NNInferRegistryEntry* entry) {
    int i = 0;

    if (entry == 0 || is_empty(entry->type_name) || entry->infer_step == 0) {
        return -1;
    }

    for (i = 0; i < (int)(sizeof(g_slots) / sizeof(g_slots[0])); ++i) {
        if (g_slots[i].used && strcmp(g_slots[i].type_name, entry->type_name) == 0) {
            g_slots[i].entry = entry;
            return 0;
        }
    }

    for (i = 0; i < (int)(sizeof(g_slots) / sizeof(g_slots[0])); ++i) {
        if (!g_slots[i].used) {
            g_slots[i].used = 1;
            (void)strncpy(g_slots[i].type_name, entry->type_name, sizeof(g_slots[i].type_name) - 1);
            g_slots[i].type_name[sizeof(g_slots[i].type_name) - 1] = '\0';
            g_slots[i].entry = entry;
            return 0;
        }
    }

    return -2;
}

const NNInferRegistryEntry* nn_infer_registry_find_entry(const char* type_name) {
    int i = 0;

    if (is_empty(type_name)) {
        return 0;
    }

    for (i = 0; i < (int)(sizeof(g_slots) / sizeof(g_slots[0])); ++i) {
        if (g_slots[i].used && strcmp(g_slots[i].type_name, type_name) == 0) {
            return g_slots[i].entry;
        }
    }

    return 0;
}

int nn_infer_registry_get(const char* type_name, NNInferStepFn* out_infer_step) {
    const NNInferRegistryEntry* entry;

    if (out_infer_step == 0) {
        return -1;
    }

    entry = nn_infer_registry_find_entry(type_name);
    if (entry == 0) {
        return 1;
    }

    *out_infer_step = entry->infer_step;
    return 0;
}

int nn_infer_registry_is_registered(const char* type_name) {
    return nn_infer_registry_find_entry(type_name) != 0 ? 1 : 0;
}

int nn_infer_registry_clear(void) {
    memset(g_slots, 0, sizeof(g_slots));
    g_bootstrapped = 0;
    g_bootstrap_failed = 0;
    return 0;
}

int nn_infer_registry_bootstrap(void) {
    const NNInferRegistryEntry* const* entries = 0;
    size_t count = 0;
    size_t i = 0;

    if (g_bootstrapped) {
        return g_bootstrap_failed ? -1 : 0;
    }

    if (nn_infer_registry_clear() != 0) {
        g_bootstrap_failed = 1;
        g_bootstrapped = 1;
        return -1;
    }

    entries = nn_infer_registry_builtin_entries(&count);
    for (i = 0; i < count; ++i) {
        if (entries[i] == 0 || nn_infer_registry_register(entries[i]) != 0) {
            g_bootstrap_failed = 1;
        }
    }

    g_bootstrapped = 1;
    return g_bootstrap_failed ? -1 : 0;
}
