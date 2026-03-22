/**
 * @file prof_error.h
 * @brief Profiler error handling helper module
 *
 * Provides unified error handling mechanism:
 * - Output error description to terminal
 * - Write description to error buffer
 * - Return error code
 *
 * Constraints:
 * - Error descriptions use English only
 * - Stop immediately on first error
 * - Error buffer recommended capacity >= 256 bytes
 */

#ifndef PROF_ERROR_H
#define PROF_ERROR_H

#include "profiler_types.h"

/**
 * @brief Initialize error buffer
 *
 * @param err Error buffer pointer
 * @param buffer Character buffer
 * @param capacity Buffer capacity
 */
void prof_error_init(ProfErrorBuffer* err, char* buffer, size_t capacity);

/**
 * @brief Set error and return
 *
 * Simultaneously:
 * - Output error description to stderr
 * - Write error description to err->buffer (truncate if needed)
 * - Return specified error code
 *
 * @param err Error buffer (can be NULL)
 * @param status Error code
 * @param format Format string (English)
 * @return Always returns status
 */
ProfStatus prof_error_set(ProfErrorBuffer* err, ProfStatus status, const char* format, ...);

/**
 * @brief Check if error buffer is valid
 *
 * @param err Error buffer pointer
 * @return 1 valid, 0 invalid
 */
int prof_error_is_valid(const ProfErrorBuffer* err);

/**
 * @brief Clear error buffer
 *
 * @param err Error buffer pointer
 */
void prof_error_clear(ProfErrorBuffer* err);

#endif
