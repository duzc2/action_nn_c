/**
 * @file prof_codegen.h
 * @brief Code generation module
 *
 * Generates modular network code based on network definition:
 * - tokenizer.c / tokenizer.h
 * - network_init.c / network_init.h
 * - weights_load.c / weights_load.h
 * - train.c / train.h
 * - weights_save.c / weights_save.h
 * - infer.c / infer.h
 *
 * Also generates metadata file.
 */

#ifndef PROF_CODEGEN_H
#define PROF_CODEGEN_H

#include "profiler_types.h"
#include "network_def.h"

/**
 * @brief Code generation context
 */
typedef struct {
    const NN_NetworkDef* network;
    const ProfOutputLayout* output_layout;
    uint64_t network_hash;
    uint64_t layout_hash;
    ProfErrorBuffer* error;
} ProfCodegenContext;

/**
 * @brief Initialize code generation context
 *
 * @param ctx Code generation context
 * @param network Network definition
 * @param output_layout Output path layout
 * @param network_hash Network hash
 * @param layout_hash Layout hash
 * @param error Error buffer
 */
void prof_codegen_init(
    ProfCodegenContext* ctx,
    const NN_NetworkDef* network,
    const ProfOutputLayout* output_layout,
    uint64_t network_hash,
    uint64_t layout_hash,
    ProfErrorBuffer* error
);

/**
 * @brief Generate all module files
 *
 * @param ctx Code generation context
 * @return PROF_STATUS_OK on success, error code on failure
 */
ProfStatus prof_codegen_generate_all(ProfCodegenContext* ctx);

/**
 * @brief Generate tokenizer module
 *
 * @param ctx Code generation context
 * @return PROF_STATUS_OK on success, error code on failure
 */
ProfStatus prof_codegen_tokenizer(ProfCodegenContext* ctx);

/**
 * @brief Generate network initialization module
 *
 * @param ctx Code generation context
 * @return PROF_STATUS_OK on success, error code on failure
 */
ProfStatus prof_codegen_network_init(ProfCodegenContext* ctx);

/**
 * @brief Generate weights loading module
 *
 * @param ctx Code generation context
 * @return PROF_STATUS_OK on success, error code on failure
 */
ProfStatus prof_codegen_weights_load(ProfCodegenContext* ctx);

/**
 * @brief Generate training module
 *
 * @param ctx Code generation context
 * @return PROF_STATUS_OK on success, error code on failure
 */
ProfStatus prof_codegen_train(ProfCodegenContext* ctx);

/**
 * @brief Generate weights saving module
 *
 * @param ctx Code generation context
 * @return PROF_STATUS_OK on success, error code on failure
 */
ProfStatus prof_codegen_weights_save(ProfCodegenContext* ctx);

/**
 * @brief Generate inference module
 *
 * @param ctx Code generation context
 * @return PROF_STATUS_OK on success, error code on failure
 */
ProfStatus prof_codegen_infer(ProfCodegenContext* ctx);

/**
 * @brief Generate metadata file
 *
 * @param ctx Code generation context
 * @param metadata_path Path for metadata file
 * @return PROF_STATUS_OK on success, error code on failure
 */
ProfStatus prof_codegen_metadata(
    ProfCodegenContext* ctx,
    const char* metadata_path
);

#endif
