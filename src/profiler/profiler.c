/**
 * @file profiler.c
 * @brief Top-level profiler orchestration entry.
 *
 * This file wires together the three mandatory generation stages described in
 * the documentation:
 * 1. validate the caller-supplied request and network definition,
 * 2. compute deterministic hashes used for compatibility checks,
 * 3. hand the normalized request to the code generator.
 *
 * The implementation preserves the required fast-fail behaviour: as soon as one
 * stage reports an error, no later stage runs and the first diagnostic is kept.
 */

#include "profiler.h"
#include "prof_error.h"
#include "prof_validate.h"
#include "prof_hash.h"
#include "prof_codegen.h"

#include <string.h>

/**
 * @brief Run the full profiler pipeline for one caller-supplied network.
 */
ProfStatus profiler_generate_v2(
    const ProfGenerateRequest* req,
    ProfGenerateResult* out_result
) {
    ProfErrorBuffer error;
    ProfCodegenContext codegen_ctx;
    NN_NetworkDef* network;
    uint64_t network_hash;
    uint64_t layout_hash;
    ProfStatus st;

    /* The public API rejects a NULL request before touching user buffers. */
    if (req == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    /* Normalize the caller-owned error buffer so later helpers can reuse it. */
    prof_error_init(&error, req->error.buffer, req->error.capacity);

    /* The current profiler pipeline consumes the structured network directly. */
    network = (NN_NetworkDef*)req->network_def;

    /* Stage 1: validate topology, paths, and registration-dependent contracts. */
    st = prof_validate_all(req, &error);
    if (st != PROF_STATUS_OK) {
        if (error.buffer != NULL && error.capacity > 0U && error.buffer[0] == '\0') {
            prof_error_set(&error, st, "%s", prof_status_to_string(st));
        }
        return st;
    }

    /* Stage 2: compute the two hashes used by save/load compatibility checks. */
    network_hash = prof_network_hash(network);
    layout_hash = prof_layout_hash(network);

    /* Stage 3: pass the normalized request into the code generator. */
    prof_codegen_init(&codegen_ctx, network, &req->output_layout,
        network_hash, layout_hash, &error);

    st = prof_codegen_generate_all(&codegen_ctx);
    if (st != PROF_STATUS_OK) {
        if (error.buffer != NULL && error.capacity > 0U && error.buffer[0] == '\0') {
            prof_error_set(&error, st, "%s", prof_status_to_string(st));
        }
        return st;
    }

    /* Report stable outputs back to the caller once generation is complete. */
    if (out_result != NULL) {
        out_result->network_hash = network_hash;
        out_result->metadata_written_path = req->output_layout.metadata_path;
    }

    return PROF_STATUS_OK;
}
