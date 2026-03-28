/**
 * @file demo_runtime_paths.h
 * @brief Demo runtime path helpers
 *
 * This header provides a tiny utility used by demo executables.
 * All runtime-generated files must be emitted relative to the
 * executable location in the build tree, not into source folders.
 *
 * Usage pattern:
 * 1. Call demo_set_working_directory_to_executable() at program start.
 * 2. Use relative paths that are anchored from the executable directory,
 *    for example "../data/infer.c".
 * 3. The relative paths then resolve under build/demo/<name>/data/.
 */

#ifndef DEMO_RUNTIME_PATHS_H
#define DEMO_RUNTIME_PATHS_H

#include <stddef.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define DEMO_CHDIR _chdir
#else
#include <unistd.h>
#define DEMO_CHDIR chdir
#endif

/**
 * @brief Get the directory that contains the current executable.
 *
 * @param out_dir Output buffer for directory path
 * @param out_size Output buffer size
 * @return 0 on success, negative on failure
 */
static int demo_get_executable_dir(char* out_dir, size_t out_size) {
    size_t len;

    if (out_dir == NULL || out_size == 0U) {
        return -1;
    }

#ifdef _WIN32
    {
        DWORD copied = GetModuleFileNameA(NULL, out_dir, (DWORD)out_size);
        if (copied == 0U || copied >= (DWORD)out_size) {
            return -1;
        }
        len = (size_t)copied;
    }
#else
    if (getcwd(out_dir, out_size) == NULL) {
        return -1;
    }
    len = strlen(out_dir);
#endif

    while (len > 0U) {
        char ch = out_dir[len - 1U];
        if (ch == '\\' || ch == '/') {
            out_dir[len - 1U] = '\0';
            return 0;
        }
        len--;
    }

    return -1;
}

/**
 * @brief Change current working directory to executable directory.
 *
 * After this call, relative paths like "../data" resolve relative
 * to the executable location instead of the shell launch directory.
 *
 * @return 0 on success, negative on failure
 */
static int demo_set_working_directory_to_executable(void) {
    char executable_dir[1024];

    if (demo_get_executable_dir(executable_dir, sizeof(executable_dir)) != 0) {
        return -1;
    }

    if (DEMO_CHDIR(executable_dir) != 0) {
        return -1;
    }

    return 0;
}

#endif
