/**
 * @file profiler.h
 * @brief Profiler main interface header
 *
 * Defines the public profiler code generation interface.
 * User program generates network structure code by calling profiler_generate_v2().
 *
 * Usage flow:
 * 1. User program builds NN_NetworkDef structure
 * 2. Fill ProfOutputLayout to specify output paths
 * 3. Call profiler_generate_v2() to generate code
 * 4. Profiler validates network and generates training/inference code
 */

#ifndef PROFILER_H
#define PROFILER_H

#include "profiler_types.h"
#include "network_def.h"

#include <stddef.h>

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

#endif
