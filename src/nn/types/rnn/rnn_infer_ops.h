/**
 * @file rnn_infer_ops.h
 * @brief Public inference-side API for the tiny RNN backend.
 */

#ifndef RNN_INFER_OPS_H
#define RNN_INFER_OPS_H

#include "rnn_config.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @brief Inference context for the tiny RNN leaf.
 */
typedef struct {
    RnnConfig config;               /**< Reconstructed type-specific configuration blob. */
    uint64_t expected_network_hash; /**< Hash guard for cross-network weight reuse. */
    uint64_t expected_layout_hash;  /**< Hash guard for parameter layout compatibility. */
    uint32_t rng_state;             /**< Deterministic RNG state for parameter init. */
    float* input_to_hidden;         /**< [hidden][input] recurrent input weights. */
    float* hidden_to_hidden;        /**< [hidden][hidden] recurrent transition weights. */
    float* hidden_bias;             /**< One scalar bias per hidden unit. */
    float* hidden_to_output;        /**< [output][hidden] readout weights. */
    float* output_bias;             /**< One scalar bias per output node. */
    float* input_buffer;            /**< Owned copy of the latest flattened input. */
    float* output_buffer;           /**< Owned copy of the latest output vector. */
    float* hidden_state_a;          /**< Reusable scratch buffer for one hidden-state vector. */
    float* hidden_state_b;          /**< Reusable scratch buffer for one hidden-state vector. */
} RnnInferContext;

RnnInferContext* nn_rnn_infer_create(void);
RnnInferContext* nn_rnn_infer_create_with_config(const RnnConfig* config, uint32_t seed);
void nn_rnn_infer_destroy(void* context);
void nn_rnn_infer_set_input(void* context, const float* input, size_t size);
void nn_rnn_infer_get_output(void* context, float* output, size_t size);
int nn_rnn_infer_step(void* context);
int nn_rnn_infer_auto_run(void* context, const float* input, float* output);
int nn_rnn_forward_pass(
    RnnInferContext* context,
    const float* input,
    float* output,
    float* hidden_cache,
    float* output_linear_cache
);
int nn_rnn_load_weights(void* context, FILE* fp);
int nn_rnn_save_weights(void* context, FILE* fp);
uint64_t nn_rnn_get_network_hash(const void* context);

#endif
