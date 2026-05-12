/**
 * @file nn_infer_registry.c
 * @brief Static registry implementation for enabled inference backends.
 *
 * Maintains two storage arrays: legacy NNInferRegistryEntry (for auto-generated
 * code) and VTable-based NNInferBackend (for new dispatch). The VTable API
 * validates pointer completeness before accepting a registration.
 */

#include "nn_infer_registry.h"

#include <string.h>
#include "../utils/error.h"
#include "../utils/log.h"

/**
 * @brief One slot in the static inference registry table.
 */
typedef struct {
    int used;
    char type_name[64];
    const NNInferRegistryEntry* entry;
} NNInferRegistrySlot;

static NNInferRegistrySlot g_slots[32];
static int g_bootstrapped = 0;
static int g_bootstrap_failed = 0;

/* --- VTable storage (new dispatch) --- */
static NNInferBackend* g_vtable_backends[NN_INFER_MAX_BACKENDS];
static size_t g_vtable_count = 0;

static int is_empty(const char* text) {
    return text == 0 || text[0] == '\0';
}

static void copy_type_name(char* destination, size_t capacity, const char* source) {
    size_t copy_length;

    if (destination == 0 || capacity == 0U) {
        return;
    }
    if (source == 0) {
        destination[0] = '\0';
        return;
    }

    copy_length = strlen(source);
    if (copy_length >= capacity) {
        copy_length = capacity - 1U;
    }
    (void)memcpy(destination, source, copy_length);
    destination[copy_length] = '\0';
}

/* --- Legacy entry API --- */

int nn_infer_registry_register(const NNInferRegistryEntry* entry) {
    int i = 0;

    if (entry == 0 || is_empty(entry->type_name) || entry->infer_step == 0) {
        return ACTION_C_ERR_NULL_POINTER;
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
            copy_type_name(g_slots[i].type_name, sizeof(g_slots[i].type_name), entry->type_name);
            g_slots[i].entry = entry;
            return 0;
        }
    }

    return ACTION_C_ERR_NO_MEMORY;
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
        return ACTION_C_ERR_NULL_POINTER;
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
    /* Also clear VTable storage. */
    g_vtable_count = 0;
    memset(g_vtable_backends, 0, sizeof(g_vtable_backends));
    return 0;
}

/* Skip bootstrap when building standalone test -- it depends on autogen symbols. */
#ifndef NN_REGISTRY_BUILD_TEST

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
        return ACTION_C_ERR_CONFIG_INVALID;
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

#endif /* !NN_REGISTRY_BUILD_TEST */

/* --- VTable API --- */

int nn_infer_vtable_register(const NNInferBackend* backend) {
    size_t i;

    if (backend == 0 || backend->type_name == 0) {
        return ACTION_C_ERR_NULL_POINTER;
    }
    /* Validate that the minimal required function pointers are present. */
    if (backend->create == 0 || backend->destroy == 0 || backend->step == 0) {
        LOG_ERROR("VTable register failed: backend '%s' missing create/destroy/step",
                  backend->type_name);
        return ACTION_C_ERR_NULL_POINTER;
    }
    if (g_vtable_count >= NN_INFER_MAX_BACKENDS) {
        LOG_ERROR("Infer VTable registry full (max %u)", NN_INFER_MAX_BACKENDS);
        return ACTION_C_ERR_NO_MEMORY;
    }

    /* Replace duplicates. */
    for (i = 0; i < g_vtable_count; i++) {
        if (strcmp(g_vtable_backends[i]->type_name, backend->type_name) == 0) {
            LOG_WARN("Infer VTable '%s' already registered, replacing",
                     backend->type_name);
            g_vtable_backends[i] = (NNInferBackend*)backend;
            return ACTION_C_OK;
        }
    }

    g_vtable_backends[g_vtable_count++] = (NNInferBackend*)backend;
    return ACTION_C_OK;
}

const NNInferBackend* nn_infer_vtable_find(const char* type_name) {
    size_t i;

    if (type_name == 0) {
        return 0;
    }
    for (i = 0; i < g_vtable_count; i++) {
        if (strcmp(g_vtable_backends[i]->type_name, type_name) == 0) {
            return g_vtable_backends[i];
        }
    }
    return 0;
}

size_t nn_infer_vtable_count(void) {
    return g_vtable_count;
}

const NNInferBackend* nn_infer_vtable_get(size_t index) {
    if (index >= g_vtable_count) {
        return 0;
    }
    return g_vtable_backends[index];
}
