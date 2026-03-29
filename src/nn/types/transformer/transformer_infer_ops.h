/**
 * @file transformer_infer_ops.h
 * @brief Public inference-side API for the dynamically sized transformer backend.
 */

#ifndef TRANSFORMER_INFER_OPS_H
#define TRANSFORMER_INFER_OPS_H

#include "transformer_config.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @brief Inference context for the dynamically sized transformer backend.
 */
typedef struct {
    const char* question;              /**< Current input text passed in by caller or graph runtime. */
    char* answer;                      /**< Writable answer buffer owned by the caller. */
    size_t answer_capacity;            /**< Capacity of @ref answer in bytes. */
    uint64_t expected_network_hash;    /**< Weight-file compatibility guard. */
    uint64_t expected_layout_hash;     /**< Layout compatibility guard. */
    size_t graph_input_size;           /**< Graph-mode input width expected by generated code. */
    size_t graph_output_size;          /**< Graph-mode output width expected by generated code. */
    size_t vocab_size;                 /**< Effective vocabulary size after initialization. */
    size_t max_seq_length;             /**< Maximum supported token sequence length. */
    size_t model_dim;                  /**< Active embedding/attention width. */
    size_t max_response_classes;       /**< Capacity of the learned answer-class table. */
    size_t max_text_length;            /**< Capacity of each stored answer string. */
    size_t class_count;                /**< Number of learned answer classes. */
    uint32_t rng_state;                /**< Deterministic RNG state for lightweight init logic. */
    float* token_embedding;            /**< [vocab][model_dim] token embeddings. */
    float* position_embedding;         /**< [max_seq_length][model_dim] position embeddings. */
    float* query_weight;               /**< [model_dim][model_dim] attention query weights. */
    float* key_weight;                 /**< [model_dim][model_dim] attention key weights. */
    float* value_weight;               /**< [model_dim][model_dim] attention value weights. */
    float* output_weight;              /**< [model_dim][model_dim] attention output weights. */
    float* classifier_weight;          /**< [max_response_classes][model_dim] classifier weights. */
    float* classifier_bias;            /**< [max_response_classes] classifier bias. */
    char* class_texts;                 /**< [max_response_classes][max_text_length] flattened texts. */
    char* fallback_answer;             /**< Stable fallback answer buffer. */
    float* graph_projection_weight;    /**< [graph_input_size][graph_output_size] graph-mode weights. */
    float* graph_projection_bias;      /**< [graph_output_size] graph-mode bias. */
} TransformerInferContext;

int nn_transformer_init_parameters(
    TransformerInferContext* context,
    const TransformerModelConfig* config,
    size_t graph_input_size,
    size_t graph_output_size
);
void nn_transformer_infer_destroy(void* context);
size_t nn_transformer_tokenize_text(
    const char* text,
    size_t* out_tokens,
    size_t token_capacity,
    size_t vocab_size
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
