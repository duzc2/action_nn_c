/**
 * @file nn_train_registry.c
 * @brief Static registry implementation for enabled training backends.
 *
 * The training registry intentionally mirrors the inference registry so both
 * execution phases share the same bootstrap, lookup, and fast-fail semantics.
 */

#include "nn_train_registry.h"

#include <string.h>

/**
 * @brief One slot in the static training registry table.
 */
typedef struct {
    int used;                         /**< Non-zero once the slot contains a valid entry. */
    char type_name[64];               /**< Local copy used as the lookup key. */
    const NNTrainRegistryEntry* entry;/**< Pointer to the compile-time builtin entry. */
} NNTrainRegistrySlot;

/** Static storage for all enabled training backends. */
static NNTrainRegistrySlot g_slots[32];
/** Latch indicating whether builtin registration has already been attempted. */
static int g_bootstrapped = 0;
/** Sticky failure flag so later callers see the original bootstrap result. */
static int g_bootstrap_failed = 0;

/**
 * @brief Treat NULL and empty strings as equally unusable registry keys.
 */
static int is_empty(const char* text) {
    return text == 0 || text[0] == '\0';
}

/**
 * @brief Register or replace one training backend entry.
 */
int nn_train_registry_register(const NNTrainRegistryEntry* entry) {
    int i = 0;

    /* Reject incomplete entries because runtime dispatch depends on train_step. */
    if (entry == 0 || is_empty(entry->type_name) || entry->train_step == 0) {
        return -1;
    }

    /* Duplicate names replace the old pointer to keep bootstrap idempotent. */
    for (i = 0; i < (int)(sizeof(g_slots) / sizeof(g_slots[0])); ++i) {
        if (g_slots[i].used && strcmp(g_slots[i].type_name, entry->type_name) == 0) {
            g_slots[i].entry = entry;
            return 0;
        }
    }

    /* First free slot wins because the enabled type set is intentionally small. */
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
const NNTrainRegistryEntry* nn_train_registry_find_entry(const char* type_name) {
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
 * @brief Resolve only the single-step training hook needed by runtime dispatch.
 */
int nn_train_registry_get(const char* type_name, NNTrainStepFn* out_train_step) {
    const NNTrainRegistryEntry* entry;

    if (out_train_step == 0) {
        return -1;
    }

    entry = nn_train_registry_find_entry(type_name);
    if (entry == 0) {
        return 1;
    }

    *out_train_step = entry->train_step;
    return 0;
}

/**
 * @brief Convenience predicate used by validation and tests.
 */
int nn_train_registry_is_registered(const char* type_name) {
    return nn_train_registry_find_entry(type_name) != 0 ? 1 : 0;
}

/**
 * @brief Reset all cached registry state.
 */
int nn_train_registry_clear(void) {
    memset(g_slots, 0, sizeof(g_slots));
    g_bootstrapped = 0;
    g_bootstrap_failed = 0;
    return 0;
}

/**
 * @brief Populate the static table from the build-generated builtin entry list.
 */
int nn_train_registry_bootstrap(void) {
    const NNTrainRegistryEntry* const* entries = 0;
    size_t count = 0;
    size_t i = 0;

    /* Later callers observe the original bootstrap result without rework. */
    if (g_bootstrapped) {
        return g_bootstrap_failed ? -1 : 0;
    }

    /* Start from a clean table so bootstrap remains deterministic. */
    if (nn_train_registry_clear() != 0) {
        g_bootstrap_failed = 1;
        g_bootstrapped = 1;
        return -1;
    }

    /* Register every CMake-enabled builtin entry emitted by the build. */
    entries = nn_train_registry_builtin_entries(&count);
    for (i = 0; i < count; ++i) {
        if (entries[i] == 0 || nn_train_registry_register(entries[i]) != 0) {
            g_bootstrap_failed = 1;
        }
    }

    g_bootstrapped = 1;
    return g_bootstrap_failed ? -1 : 0;
}
