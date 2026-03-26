/**
 * @file profiler_types.c
 * @brief Human-readable helpers for profiler public enums.
 *
 * Although the profiler primarily communicates through status codes, readable
 * strings still matter because the contract requires terminal diagnostics and a
 * caller-visible error buffer. This file keeps the mapping in one place so the
 * rest of the code can stay focused on control flow.
 */

#include "profiler_types.h"

/**
 * @brief Error code to string mapping table.
 *
 * The table order intentionally mirrors @ref ProfStatus so simple index-based
 * lookup can stay branch-light in the common case.
 */
static const char* g_status_strings[] = {
    "OK",
    "Invalid argument",
    "Validation failed",
    "Cycle detected in network topology",
    "Invalid output path",
    "File I/O failed",
    "Hash mismatch - weight file incompatible with network",
    "Layout mismatch - metadata or parameter layout inconsistent",
    "Internal error"
};

/**
 * @brief Convert a public status code into a stable English description.
 */
const char* prof_status_to_string(ProfStatus status) {
    size_t index = (size_t)status;
    size_t max_index = sizeof(g_status_strings) / sizeof(g_status_strings[0]) - 1;

    /* Unknown values collapse to the final "internal error" message. */
    if (index > max_index) {
        index = max_index;
    }

    return g_status_strings[index];
}
