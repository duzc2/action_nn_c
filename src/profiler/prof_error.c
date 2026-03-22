/**
 * @file prof_error.c
 * @brief Profiler error handling implementation
 */

#include "prof_error.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

void prof_error_init(ProfErrorBuffer* err, char* buffer, size_t capacity) {
    if (err == NULL) {
        return;
    }

    err->buffer = buffer;
    err->capacity = capacity;

    if (buffer != NULL && capacity > 0) {
        buffer[0] = '\0';
    }
}

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

void prof_error_clear(ProfErrorBuffer* err) {
    if (err == NULL || err->buffer == NULL || err->capacity == 0) {
        return;
    }

    err->buffer[0] = '\0';
}

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

ProfStatus prof_error_set(ProfErrorBuffer* err, ProfStatus status, const char* format, ...) {
    char terminal_buffer[512];
    va_list args;

    va_start(args, format);
    vsnprintf(terminal_buffer, sizeof(terminal_buffer), format, args);
    va_end(args);

    terminal_buffer[sizeof(terminal_buffer) - 1] = '\0';

    fprintf(stderr, "[PROFILER ERROR %d] %s\n", (int)status, terminal_buffer);

    va_start(args, format);
    prof_error_vformat(err, status, format, args);
    va_end(args);

    return status;
}
