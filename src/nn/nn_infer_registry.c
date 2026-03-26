/**
 * @file nn_infer_registry.c
 * @brief Static registry implementation for enabled inference backends.
 *
 * The registry deliberately uses a fixed-size table instead of dynamic
 * containers because the enabled backend set is known at build time. That
 * design keeps bootstrap deterministic and avoids hidden heap failures during
 * runtime dispatch.
 */

#include "nn_infer_registry.h"

#include <string.h>

/**
 * @brief One slot in the static inference registry table.
 */
typedef struct {
    int used;                         /**< Non-zero once the slot contains a valid entry. */
    char type_name[64];               /**< Local copy used as the lookup key. */
    const NNInferRegistryEntry* entry;/**< Pointer to the compile-time builtin entry. */
} NNInferRegistrySlot;

/** Static storage for all enabled inference backends. */
static NNInferRegistrySlot g_slots[32];
/** Latch indicating whether builtin registration has already run. */
static int g_bootstrapped = 0;
/** Sticky failure flag so later callers observe the original bootstrap error. */
static int g_bootstrap_failed = 0;

/**
 * @brief Treat NULL and empty strings as equally unusable registry keys.
 */
static int is_empty(const char* text) {
    return text == 0 || text[0] == '\0';
}

/**
 * @brief Register or replace one inference backend entry.
 */
int nn_infer_registry_register(const NNInferRegistryEntry* entry) {
    int i = 0;

    /* Reject incomplete entries because generated code depends on infer_step. */
    if (entry == 0 || is_empty(entry->type_name) || entry->infer_step == 0) {
        return -1;
    }

    /* Duplicate names replace the hook pointer so bootstrap stays idempotent. */
    for (i = 0; i < (int)(sizeof(g_slots) / sizeof(g_slots[0])); ++i) {
        if (g_slots[i].used && strcmp(g_slots[i].type_name, entry->type_name) == 0) {
            g_slots[i].entry = entry;
            return 0;
        }
    }

    /* First free slot wins because the table is small and deterministic. */
    for (i = 0; i < (int)(sizeof(g_slots) / sizeof(g_slots[0])); ++i) {
        if (!g_slots[i].used) {
            g_slots[i].used = 1;
            (void)strncpy(g_slots[i].type_name, entry->type_name, sizeof(g_slots[i].type_name) - 1);
            g_slots[i].type_name[sizeof(g_slots[i].type_name) - 1] = '\0';
            g_slots[i].entry = entry;
            return 0;
        }
    }

    /* A full table means the static registry budget was exceeded. */
    return -2;
}

/**
 * @brief Find the registry entry matching a semantic type name.
 */
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

/**
 * @brief Resolve only the single-step inference hook needed by runtime dispatch.
 */
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

/**
 * @brief Convenience predicate used by validation and tests.
 */
int nn_infer_registry_is_registered(const char* type_name) {
    return nn_infer_registry_find_entry(type_name) != 0 ? 1 : 0;
}

/**
 * @brief Reset all cached registry state.
 */
int nn_infer_registry_clear(void) {
    memset(g_slots, 0, sizeof(g_slots));
    g_bootstrapped = 0;
    g_bootstrap_failed = 0;
    return 0;
}

/**
 * @brief Populate the static table from the build-generated builtin entry list.
 */
int nn_infer_registry_bootstrap(void) {
    const NNInferRegistryEntry* const* entries = 0;
    size_t count = 0;
    size_t i = 0;

    /* Later callers observe the original bootstrap outcome without rework. */
    if (g_bootstrapped) {
        return g_bootstrap_failed ? -1 : 0;
    }

    /* Start from a clean table so bootstrap remains deterministic. */
    if (nn_infer_registry_clear() != 0) {
        g_bootstrap_failed = 1;
        g_bootstrapped = 1;
        return -1;
    }

    /* Register every CMake-enabled builtin entry emitted by the build. */
    entries = nn_infer_registry_builtin_entries(&count);
    for (i = 0; i < count; ++i) {
        if (entries[i] == 0 || nn_infer_registry_register(entries[i]) != 0) {
            g_bootstrap_failed = 1;
        }
    }

    g_bootstrapped = 1;
    return g_bootstrap_failed ? -1 : 0;
}
