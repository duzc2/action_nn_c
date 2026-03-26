/**
 * @file transformer_infer_ops.h
 * @brief Public inference-side API for the tiny transformer backend.
 *
 * The implementation is intentionally compact and oriented toward demo-sized
 * conversational tasks. This header exposes only the pieces required by the
 * registry bridge and generated code: context shape, parameter I/O, tokenization,
 * and single-step/graph execution helpers.
 */

#ifndef TRANSFORMER_INFER_OPS_H
#define TRANSFORMER_INFER_OPS_H

#include "transformer_config.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define TRANSFORMER_VOCAB_SIZE 44U
#define TRANSFORMER_MAX_SEQ_LENGTH 32U
#define TRANSFORMER_MAX_MODEL_DIM 32U
#define TRANSFORMER_MAX_RESPONSE_CLASSES 32U
#define TRANSFORMER_MAX_TEXT_LENGTH 256U

/**
 * @brief Inference context for the tiny transformer demo backend.
 *
 * Most buffers are fixed-size because the demo targets a small, bounded model
 * and the surrounding code generation pipeline already knows the maximum sizes.
 */
typedef struct {
    const char* question;  /**< Current input text passed in by caller or graph runtime. */
    char* answer;          /**< Writable answer buffer owned by the caller. */
    size_t answer_capacity;/**< Capacity of @ref answer in bytes. */
    uint64_t expected_network_hash; /**< Weight-file compatibility guard. */
    uint64_t expected_layout_hash;  /**< Layout compatibility guard. */
    size_t graph_input_size;        /**< Graph-mode input width expected by generated code. */
    size_t graph_output_size;       /**< Graph-mode output width expected by generated code. */
    size_t vocab_size;              /**< Effective vocabulary size after initialization. */
    size_t max_seq_length;          /**< Maximum supported token sequence length. */
    size_t model_dim;               /**< Active embedding/attention width. */
    size_t class_count;             /**< Number of learned answer classes. */
    uint32_t rng_state;             /**< Deterministic RNG state for lightweight init logic. */
    float token_embedding[TRANSFORMER_VOCAB_SIZE][TRANSFORMER_MAX_MODEL_DIM];
    float position_embedding[TRANSFORMER_MAX_SEQ_LENGTH][TRANSFORMER_MAX_MODEL_DIM];
    float query_weight[TRANSFORMER_MAX_MODEL_DIM][TRANSFORMER_MAX_MODEL_DIM];
    float key_weight[TRANSFORMER_MAX_MODEL_DIM][TRANSFORMER_MAX_MODEL_DIM];
    float value_weight[TRANSFORMER_MAX_MODEL_DIM][TRANSFORMER_MAX_MODEL_DIM];
    float output_weight[TRANSFORMER_MAX_MODEL_DIM][TRANSFORMER_MAX_MODEL_DIM];
    float classifier_weight[TRANSFORMER_MAX_RESPONSE_CLASSES][TRANSFORMER_MAX_MODEL_DIM];
    float classifier_bias[TRANSFORMER_MAX_RESPONSE_CLASSES];
    char class_texts[TRANSFORMER_MAX_RESPONSE_CLASSES][TRANSFORMER_MAX_TEXT_LENGTH];
    char fallback_answer[TRANSFORMER_MAX_TEXT_LENGTH];
} TransformerInferContext;

int nn_transformer_init_parameters(
    TransformerInferContext* context,
    size_t model_dim,
    uint32_t seed
);
size_t nn_transformer_tokenize_text(
    const char* text,
    uint8_t* out_tokens,
    size_t token_capacity
);
int nn_transformer_find_class(
    const TransformerInferContext* context,
    const char* answer
);
int nn_transformer_find_or_add_class(
    TransformerInferContext* context,
    const char* answer
);
int nn_transformer_predict_class(
    const TransformerInferContext* context,
    const char* question,
    float* out_probabilities,
    size_t probability_capacity,
    float* out_loss_hint
);
int nn_transformer_graph_run(void* context, const void* input, void* output);
int nn_transformer_infer_step(void* context);

int nn_transformer_load_weights(void* context, FILE* fp);
int nn_transformer_save_weights(void* context, FILE* fp);

#endif
