/**
 * @file prof_path.h
 * @brief Shared profiler path utilities
 *
 * Centralizes the small amount of path handling needed by profiler:
 * - create a directory tree if missing
 * - create the parent directory of an output file path
 *
 * The profiler must emit generated files into caller-provided output
 * locations without hard-coding source-tree paths.
 */

#ifndef PROF_PATH_H
#define PROF_PATH_H

#include "profiler_types.h"

/**
 * @brief Ensure a directory exists.
 *
 * Missing path segments are created recursively.
 *
 * @param dir_path Directory path
 * @return PROF_STATUS_OK on success
 */
ProfStatus prof_path_ensure_directory(const char* dir_path);

/**
 * @brief Ensure the parent directory of a file path exists.
 *
 * @param file_path File path whose parent directory should exist
 * @return PROF_STATUS_OK on success
 */
ProfStatus prof_path_ensure_parent_directory(const char* file_path);

#endif
