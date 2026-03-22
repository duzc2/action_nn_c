/**
 * @file profiler.h
 * @brief Profiler main interface header
 *
 * Defines main function interfaces for profiler.
 * User program generates network structure code by calling profiler_generate().
 *
 * Usage flow:
 * 1. User program builds NN_NetworkDef structure
 * 2. Fill ProfOutputLayout to specify output paths
 * 3. Call profiler_generate() to generate code
 * 4. Profiler validates network and generates training/inference code
 */

#ifndef PROFILER_H
#define PROFILER_H

#include "profiler_types.h"
#include "network_def.h"

#include <stddef.h>

/**
 * @brief Simplified generate request (legacy interface, kept for compatibility)
 *
 * Simplified version uses fixed paths, does not support modular output.
 */
typedef struct {
    const char* network_name;
    const char* network_type;
    const char* output_dir;
} ProfilerGenerateRequest;

/**
 * @brief Simplified IO names definition (legacy interface)
 */
typedef struct {
    const char* input_names;
    size_t input_count;
    const char* output_names;
    size_t output_count;
} ProfIONames;

/**
 * @brief Full code generation function (new interface)
 *
 * Receives network definition and output path layout,
 * generates modular network code.
 *
 * @param req Generate request, contains network definition and output paths
 * @param out_result Output result, contains network_hash and metadata_path
 * @return PROF_STATUS_OK on success, other values on failure
 *
 * Error handling:
 * - Stop immediately on first error and return
 * - Error info output to terminal and req->error.buffer simultaneously
 * - Error descriptions in English
 */
ProfStatus profiler_generate_v2(
    const ProfGenerateRequest* req,
    ProfGenerateResult* out_result
);

/**
 * @brief Simplified code generation function (legacy interface, kept for compatibility)
 *
 * @param network_name Network name
 * @param network_type Network type (e.g., "mlp", "transformer")
 * @param output_dir Output directory
 * @param io_names IO names definition (optional)
 * @return 0 on success, non-zero on failure
 */
int profiler_generate_with_io(
    const char* network_name,
    const char* network_type,
    const char* output_dir,
    const ProfIONames* io_names
);

/**
 * @brief Simplified generate entry (legacy interface, kept for compatibility)
 *
 * @param request Simplified generate request
 * @return 0 on success, non-zero on failure
 */
int profiler_generate(const ProfilerGenerateRequest* request);

#endif
