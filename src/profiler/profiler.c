/**
 * @file profiler.c
 * @brief Network code generator core implementation
 *
 * Features:
 * - Read network specification passed by user
 * - Validate network definition (DAG, connections, ports)
 * - Compute network signature hash
 * - Generate modular network code
 *
 * Generated modules:
 * - tokenizer: Input/output encoding
 * - network_init: Network initialization
 * - weights_load: Weight loading with hash validation
 * - train: Training loop
 * - weights_save: Weight saving with hash
 * - infer: Inference
 * - metadata: Network metadata header
 */

#include "profiler.h"
#include "prof_error.h"
#include "prof_validate.h"
#include "prof_hash.h"
#include "prof_codegen.h"

#include <string.h>

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

    if (req == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    prof_error_init(&error, req->error.buffer, req->error.capacity);

    network = (NN_NetworkDef*)req->network_def;

    st = prof_validate_all(req, &error);
    if (st != PROF_STATUS_OK) {
        if (error.buffer != NULL && error.capacity > 0U && error.buffer[0] == '\0') {
            prof_error_set(&error, st, "%s", prof_status_to_string(st));
        }
        return st;
    }

    network_hash = prof_network_hash(network);
    layout_hash = prof_layout_hash(network);

    prof_codegen_init(&codegen_ctx, network, &req->output_layout,
        network_hash, layout_hash, &error);

    st = prof_codegen_generate_all(&codegen_ctx);
    if (st != PROF_STATUS_OK) {
        if (error.buffer != NULL && error.capacity > 0U && error.buffer[0] == '\0') {
            prof_error_set(&error, st, "%s", prof_status_to_string(st));
        }
        return st;
    }

    if (out_result != NULL) {
        out_result->network_hash = network_hash;
        out_result->metadata_written_path = req->output_layout.metadata_path;
    }

    return PROF_STATUS_OK;
}
