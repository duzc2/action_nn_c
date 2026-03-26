/**
 * @file prof_error.c
 * @brief Shared error formatting helpers for the profiler pipeline.
 *
 * The documentation requires three things to happen together on failure:
 * - return a stable status code,
 * - print a readable English message to the terminal,
 * - copy the same message into the caller-provided buffer.
 *
 * This file centralizes that behaviour so validation and code generation can
 * fail fast without duplicating formatting logic.
 */

#include "prof_error.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief Initialize a caller-provided error buffer wrapper.
 */
void prof_error_init(ProfErrorBuffer* err, char* buffer, size_t capacity) {
    if (err == NULL) {
        return;
    }

    err->buffer = buffer;
    err->capacity = capacity;

    /* Clearing the first byte gives every caller a predictable empty state. */
    if (buffer != NULL && capacity > 0) {
        buffer[0] = '\0';
    }
}

/**
 * @brief Check whether the wrapped error buffer can safely receive text.
 */
int prof_error_is_valid(const ProfErrorBuffer* err) {
    if (err == NULL) {
        return 0;
    }

    if (err->buffer == NULL) {
        return 0;
    }

    if (err->capacity == 0) {
        return 0;
    }

    return 1;
}

/**
 * @brief Reset the caller-visible error message without changing ownership.
 */
void prof_error_clear(ProfErrorBuffer* err) {
    if (err == NULL || err->buffer == NULL || err->capacity == 0) {
        return;
    }

    err->buffer[0] = '\0';
}

/**
 * @brief Write a formatted message into the caller-visible error buffer.
 *
 * The status value is intentionally unused here because the formatted English
 * sentence already carries the diagnostic meaning and is mirrored to stderr by
 * the outer helper below.
 */
static void prof_error_vformat(
    ProfErrorBuffer* err,
    ProfStatus status,
    const char* format,
    va_list args
) {
    (void)status;
    if (err != NULL && err->buffer != NULL && err->capacity > 0) {
        vsnprintf(err->buffer, err->capacity, format, args);
        err->buffer[err->capacity - 1] = '\0';
    }
}

/**
 * @brief Emit one formatted profiler error through every required channel.
 */
ProfStatus prof_error_set(ProfErrorBuffer* err, ProfStatus status, const char* format, ...) {
    char terminal_buffer[512];
    va_list args;

    /* First format once for terminal output so stderr always has a full line. */
    va_start(args, format);
    vsnprintf(terminal_buffer, sizeof(terminal_buffer), format, args);
    va_end(args);

    terminal_buffer[sizeof(terminal_buffer) - 1] = '\0';

    /* Terminal output is part of the documented profiler error contract. */
    fprintf(stderr, "[PROFILER ERROR %d] %s\n", (int)status, terminal_buffer);

    /* Re-run formatting because vsnprintf consumes the argument list. */
    va_start(args, format);
    prof_error_vformat(err, status, format, args);
    va_end(args);

    return status;
}
