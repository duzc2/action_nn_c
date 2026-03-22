#include "nn_infer_registry.h"

#include <string.h>

typedef struct {
    int used;
    char type_name[64];
    NNInferStepFn infer_step;
} NNInferRegistrySlot;

static NNInferRegistrySlot g_slots[32];
static int g_bootstrapped = 0;
static int g_bootstrap_failed = 0;

static int is_empty(const char* text) {
    return text == 0 || text[0] == '\0';
}

int nn_infer_registry_register(const char* type_name, NNInferStepFn infer_step) {
    int i = 0;
    if (is_empty(type_name) || infer_step == 0) {
        return -1;
    }
    for (i = 0; i < (int)(sizeof(g_slots) / sizeof(g_slots[0])); ++i) {
        if (g_slots[i].used && strcmp(g_slots[i].type_name, type_name) == 0) {
            g_slots[i].infer_step = infer_step;
            return 0;
        }
    }
    for (i = 0; i < (int)(sizeof(g_slots) / sizeof(g_slots[0])); ++i) {
        if (!g_slots[i].used) {
            g_slots[i].used = 1;
            (void)strncpy(g_slots[i].type_name, type_name, sizeof(g_slots[i].type_name) - 1);
            g_slots[i].type_name[sizeof(g_slots[i].type_name) - 1] = '\0';
            g_slots[i].infer_step = infer_step;
            return 0;
        }
    }
    return -2;
}

int nn_infer_registry_get(const char* type_name, NNInferStepFn* out_infer_step) {
    int i = 0;
    if (is_empty(type_name) || out_infer_step == 0) {
        return -1;
    }
    for (i = 0; i < (int)(sizeof(g_slots) / sizeof(g_slots[0])); ++i) {
        if (g_slots[i].used && strcmp(g_slots[i].type_name, type_name) == 0) {
            *out_infer_step = g_slots[i].infer_step;
            return 0;
        }
    }
    return 1;
}

int nn_infer_registry_is_registered(const char* type_name) {
    NNInferStepFn step = 0;
    return nn_infer_registry_get(type_name, &step) == 0 ? 1 : 0;
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
        if (entries[i] == 0 || nn_infer_registry_register(entries[i]->type_name, entries[i]->infer_step) != 0) {
            g_bootstrap_failed = 1;
        }
    }
    g_bootstrapped = 1;
    return g_bootstrap_failed ? -1 : 0;
}
