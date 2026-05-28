/**
 * @file transformer_infer_ops.h
 * @brief Public inference-side API for the dynamically sized transformer backend.
 */

#ifndef TRANSFORMER_INFER_OPS_H
#define TRANSFORMER_INFER_OPS_H

#include "transformer_config.h"
#include "../../../utils/arena.h"
#include "../../residual/skip_connection.h"
#include "../../dropout/dropout.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Forward declaration: the full cache type lives in the .c file. */
typedef struct TransformerForwardCache TransformerForwardCache;

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
    /* ── P0 shared-module state ── */
    int              use_rms_norm;     /**< Copied from config.use_rms_norm. */
    float            norm_epsilon;     /**< Copied from config.norm_epsilon. */
    int              use_dropout;      /**< Copied from config.use_dropout. */
    int              use_skip;         /**< Copied from config.use_skip. */
    SkipConnection   skip_attn;        /**< LAuReL-RW skip for attention residual. */
    float*           norm_gamma_attn;  /**< RMSNorm gamma [model_dim] (initialised to 1.0). */
    DropoutLayer     dropout_attn;     /**< Post-attention dropout layer. */
    /* P0 backward-pass cache (populated by train_create, read in train ops) */
    float*           p0_attn_pre_norm;    /**< attn_out before RMSNorm [state_count] */
    float*           p0_skip_x_cache;     /**< input_states before skip [state_count] */
    float*           p0_skip_fx_cache;    /**< normed attn_out before skip [state_count] */
    Arena* arena;                      /**< Scratch arena for forward cache + training temp buffers. */
    TransformerForwardCache* forward_cache; /**< Pre-allocated scratch cache; allocated from arena once. */
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
    TransformerInferContext* context,
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
