/**
 * @file gnn_infer_ops.h
 * @brief Public inference-side API for the tiny GNN backend.
 */

#ifndef GNN_INFER_OPS_H
#define GNN_INFER_OPS_H

#include "gnn_config.h"
#include "../../dropout/dropout.h"
#include "../../../utils/arena.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @brief Inference context for the dynamically configured GNN leaf.
 */
typedef struct {
    GnnConfig* config;              /**< Owned reconstructed type-specific configuration blob. */
    size_t config_size;             /**< Byte size of the owned configuration blob. */
    uint64_t expected_network_hash; /**< Hash guard for cross-network weight reuse. */
    uint64_t expected_layout_hash;  /**< Hash guard for parameter layout compatibility. */
    uint32_t rng_state;             /**< Deterministic RNG state for parameter initialization. */
    float* input_weight;            /**< [hidden][feature] projection from node features to hidden state. */
    float* input_bias;              /**< [hidden] bias for the initial node encoder. */
    float* self_weight;             /**< [hidden][hidden] self-state transition matrix. */
    float* message_weight;          /**< [hidden][hidden] neighbor-message transition matrix. */
    float* message_bias;            /**< [hidden] bias for message-passing updates. */
    float* readout_primary;         /**< [output][hidden] readout projection from the primary anchor state. */
    float* readout_secondary;       /**< [output][hidden] readout projection from the secondary anchor state. */
    float* readout_neighbor;        /**< [output][hidden] readout projection from a slot-selected neighbor state. */
    float* output_bias;             /**< [output] bias for the exported graph readout vector. */
    float* input_buffer;            /**< Owned copy of the latest flattened graph input. */
    float* output_buffer;           /**< Owned copy of the latest exported graph readout vector. */
    Arena* arena;                   /**< Scratch arena for hot-path temporary buffers in forward pass. */
    /* ── P0 shared-module state ── */
    int              use_rms_norm;   /**< Copied from config. */
    float            norm_epsilon;   /**< Copied from config. */
    int              use_dropout;    /**< Copied from config. */
    int              use_skip;       /**< Copied from config. */
    SkipConnection   skip_msg;       /**< Per-msg-pass skip for hidden state. */
    float*           norm_gamma_h;   /**< RMSNorm gamma [hidden_size] (init to 1.0). */
    DropoutLayer     dropout_msg;    /**< Per-msg-pass dropout layer. */
    /* P0 backward-pass cache (set by train_create, NULL during pure inference) */
    float*           p0_pre_norm;    /**< hidden before RMSNorm [stages * nodes * hidden_size]. */
    float*           p0_skip_x_cache;/**< prev_h before skip [stages * nodes * hidden_size]. */
    float*           p0_skip_fx_cache;/**< normed before skip [stages * nodes * hidden_size]. */
} GnnInferContext;

GnnInferContext* nn_gnn_infer_create(void);
GnnInferContext* nn_gnn_infer_create_with_config(const GnnConfig* config, uint32_t seed);
GnnInferContext* nn_gnn_infer_create_with_config_blob(
    const void* config_data,
    size_t config_size,
    uint32_t seed
);
void nn_gnn_infer_destroy(void* context);
void nn_gnn_infer_set_input(void* context, const float* input, size_t size);
void nn_gnn_infer_get_output(void* context, float* output, size_t size);
int nn_gnn_infer_step(void* context);
int nn_gnn_infer_auto_run(void* context, const float* input, float* output);
int nn_gnn_forward_pass(
    GnnInferContext* restrict context,
    const float* restrict input,
    float* restrict output,
    float* restrict hidden_cache
);
int nn_gnn_load_weights(void* context, FILE* fp);
int nn_gnn_save_weights(void* context, FILE* fp);
uint64_t nn_gnn_get_network_hash(const void* context);

#endif
