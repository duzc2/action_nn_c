/**
 * @file profiler_types.c
 * @brief Profiler public types implementation
 *
 * Implements public type related functions defined in profiler_types.h.
 */

#include "profiler_types.h"

/**
 * @brief Error code to string mapping table
 *
 * Corresponds to ProfStatus enum values in order.
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

const char* prof_status_to_string(ProfStatus status) {
    size_t index = (size_t)status;
    size_t max_index = sizeof(g_status_strings) / sizeof(g_status_strings[0]) - 1;

    if (index > max_index) {
        index = max_index;
    }

    return g_status_strings[index];
}
