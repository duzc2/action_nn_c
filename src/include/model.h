#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>

#include "config_user.h"

typedef struct EmbeddingLayer {
    const float* table;
    size_t vocab_size;
    size_t embed_dim;
} EmbeddingLayer;

typedef struct AttentionLayer {
    size_t embed_dim;
    size_t num_heads;
} AttentionLayer;

typedef struct MoELayer {
    size_t num_experts;
    size_t k_top;
} MoELayer;

typedef struct TransformerBlock {
    AttentionLayer attention;
    MoELayer moe;
} TransformerBlock;

void model_embedding_forward(const EmbeddingLayer* layer,
                             const int* token_ids,
                             size_t token_count,
                             float* out_vectors);

void model_attention_forward(const AttentionLayer* layer,
                             const float* input_vectors,
                             size_t token_count,
                             float* out_vectors);

void model_moe_forward(const MoELayer* layer,
                       const float* input_vectors,
                       size_t token_count,
                       float* out_vectors);

void model_transformer_block_forward(const TransformerBlock* block,
                                     const float* input_vectors,
                                     size_t token_count,
                                     float* out_vectors);

#endif
