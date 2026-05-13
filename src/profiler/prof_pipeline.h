/**
 * @file prof_pipeline.h
 * @brief Immutable pipeline intermediate types
 *
 * Defines the shared data structures that pass between profiler pipeline
 * stages: hashes computed once and reused by every codegen emitter, and
 * generated file buffers held in memory before the final write pass.
 */

#ifndef PROF_PIPELINE_H
#define PROF_PIPELINE_H

#include "prof_flatten.h"

#include <stdint.h>

/**
 * @brief Deterministic hashes computed after flattening
 *
 * These three values are computed once during the hash stage and passed
 * to every code-generation emitter so each module can embed the same
 * compatibility metadata.
 */
typedef struct {
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t abi_version;
} NetworkHashes;

/**
 * @brief Generated file contents held in memory before disk write
 *
 * Each field is a malloc'd C-string containing the full text of one
 * generated module. A field that has not been emitted remains NULL.
 * The struct exists so the pipeline can separate generation from I/O
 * and so error handling around write_file stays explicit.
 */
typedef struct {
    char* metadata_c;
    char* metadata_h;
    char* tokenizer_c;
    char* tokenizer_h;
    char* network_init_c;
    char* network_init_h;
    char* infer_c;
    char* infer_h;
    char* train_c;
    char* train_h;
    char* weights_save_c;
    char* weights_save_h;
    char* weights_load_c;
    char* weights_load_h;
} GeneratedFiles;

/**
 * @brief Initialise a GeneratedFiles struct so every pointer is NULL
 */
void generated_files_init(GeneratedFiles* f);

/**
 * @brief Free every non-NULL field in a GeneratedFiles struct
 */
void generated_files_free(GeneratedFiles* f);

#endif
