/**
 * @file prof_path.c
 * @brief Shared profiler path utilities implementation
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
 * @brief Check whether a character is a directory separator.
 */
static int prof_is_dir_separator(char ch) {
    return ch == '/' || ch == '\\';
}

/**
 * @brief Create one directory level if missing.
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

ProfStatus prof_path_ensure_directory(const char* dir_path) {
    char buffer[512];
    size_t len;
    size_t start;
    size_t i;

    if (dir_path == NULL || dir_path[0] == '\0') {
        return PROF_STATUS_PATH_INVALID;
    }

    strncpy(buffer, dir_path, sizeof(buffer) - 1U);
    buffer[sizeof(buffer) - 1U] = '\0';

    len = strlen(buffer);
    while (len > 0U && prof_is_dir_separator(buffer[len - 1U])) {
        buffer[len - 1U] = '\0';
        len--;
    }

    if (len == 0U) {
        return PROF_STATUS_PATH_INVALID;
    }

    start = 0U;
    if (len >= 2U && buffer[1] == ':') {
        start = 2U;
        if (len >= 3U && prof_is_dir_separator(buffer[2])) {
            start = 3U;
        }
    } else if (prof_is_dir_separator(buffer[0])) {
        start = 1U;
    }

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

ProfStatus prof_path_ensure_parent_directory(const char* file_path) {
    char buffer[512];
    char* last_sep;

    if (file_path == NULL || file_path[0] == '\0') {
        return PROF_STATUS_PATH_INVALID;
    }

    strncpy(buffer, file_path, sizeof(buffer) - 1U);
    buffer[sizeof(buffer) - 1U] = '\0';

    last_sep = strrchr(buffer, '/');
    if (last_sep == NULL) {
        last_sep = strrchr(buffer, '\\');
    }

    if (last_sep == NULL) {
        return PROF_STATUS_OK;
    }

    *last_sep = '\0';
    if (buffer[0] == '\0') {
        return PROF_STATUS_OK;
    }

    return prof_path_ensure_directory(buffer);
}
