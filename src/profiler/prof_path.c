/**
 * @file prof_path.c
 * @brief Shared profiler path utilities implementation.
 *
 * Generated files must be written exactly where the caller requested them. That
 * means profiler code cannot rely on pre-existing directories or hard-coded
 * source-tree paths; it must create missing directories defensively and stop on
 * the first filesystem error.
 */

#include "prof_path.h"

#include <errno.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define PROF_MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define PROF_MKDIR(path) mkdir(path, 0755)
#endif

/**
 * @brief Treat both slash styles as valid directory separators.
 */
static int prof_is_dir_separator(char ch) {
    return ch == '/' || ch == '\\';
}

/**
 * @brief Copy a filesystem path into a fixed buffer without deprecated CRT calls.
 */
static void prof_copy_path(char* destination, size_t capacity, const char* source) {
    size_t copy_length;

    if (destination == NULL || capacity == 0U) {
        return;
    }
    if (source == NULL) {
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

/**
 * @brief Create one directory level if it does not already exist.
 */
static ProfStatus prof_mkdir_if_needed(const char* path) {
    int rc;

    if (path == NULL || path[0] == '\0') {
        return PROF_STATUS_PATH_INVALID;
    }

    rc = PROF_MKDIR(path);
    if (rc == 0 || errno == EEXIST) {
        return PROF_STATUS_OK;
    }

    return PROF_STATUS_IO_FAILED;
}

/**
 * @brief Recursively ensure a directory path exists.
 */
ProfStatus prof_path_ensure_directory(const char* dir_path) {
    char buffer[512];
    size_t len;
    size_t start;
    size_t i;

    if (dir_path == NULL || dir_path[0] == '\0') {
        return PROF_STATUS_PATH_INVALID;
    }

    /* Work on a private copy because the algorithm inserts temporary terminators. */
    prof_copy_path(buffer, sizeof(buffer), dir_path);

    /* Strip trailing separators so the final mkdir call receives a real segment. */
    len = strlen(buffer);
    while (len > 0U && prof_is_dir_separator(buffer[len - 1U])) {
        buffer[len - 1U] = '\0';
        len--;
    }

    if (len == 0U) {
        return PROF_STATUS_PATH_INVALID;
    }

    /* Preserve drive roots and UNC-like leading separators when walking segments. */
    start = 0U;
    if (len >= 2U && buffer[1] == ':') {
        start = 2U;
        if (len >= 3U && prof_is_dir_separator(buffer[2])) {
            start = 3U;
        }
    } else if (prof_is_dir_separator(buffer[0])) {
        start = 1U;
    }

    /* Create intermediate segments from left to right to mirror mkdir -p. */
    for (i = start; i < len; i++) {
        if (prof_is_dir_separator(buffer[i])) {
            char saved = buffer[i];
            buffer[i] = '\0';
            if (buffer[0] != '\0') {
                ProfStatus st = prof_mkdir_if_needed(buffer);
                if (st != PROF_STATUS_OK) {
                    buffer[i] = saved;
                    return st;
                }
            }
            buffer[i] = saved;
        }
    }

    return prof_mkdir_if_needed(buffer);
}

/**
 * @brief Ensure the parent directory of a file path exists.
 */
ProfStatus prof_path_ensure_parent_directory(const char* file_path) {
    char buffer[512];
    char* last_sep;

    if (file_path == NULL || file_path[0] == '\0') {
        return PROF_STATUS_PATH_INVALID;
    }

    prof_copy_path(buffer, sizeof(buffer), file_path);

    /* Search for the last path separator using both Windows and POSIX styles. */
    last_sep = strrchr(buffer, '/');
    if (last_sep == NULL) {
        last_sep = strrchr(buffer, '\\');
    }

    /* A bare filename has no parent directory requirement. */
    if (last_sep == NULL) {
        return PROF_STATUS_OK;
    }

    *last_sep = '\0';
    if (buffer[0] == '\0') {
        return PROF_STATUS_OK;
    }

    return prof_path_ensure_directory(buffer);
}
