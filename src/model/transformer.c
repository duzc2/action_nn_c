#include <string.h>

#include "../include/model.h"

void model_embedding_forward(const EmbeddingLayer* layer,
                             const int* token_ids,
                             size_t token_count,
                             float* out_vectors) {
    size_t i = 0U;
    if (layer == NULL || token_ids == NULL || out_vectors == NULL || layer->table == NULL || layer->embed_dim == 0U) {
        return;
    }
    for (i = 0U; i < token_count; ++i) {
        size_t id = (token_ids[i] >= 0) ? (size_t)token_ids[i] : 0U;
        if (id >= layer->vocab_size) {
            id = 0U;
        }
        memcpy(&out_vectors[i * layer->embed_dim], &layer->table[id * layer->embed_dim], sizeof(float) * layer->embed_dim);
    }
}

void model_attention_forward(const AttentionLayer* layer,
                             const float* input_vectors,
                             size_t token_count,
                             float* out_vectors) {
    size_t count = 0U;
    if (layer == NULL || input_vectors == NULL || out_vectors == NULL || layer->embed_dim == 0U) {
        return;
    }
    count = token_count * layer->embed_dim;
    memcpy(out_vectors, input_vectors, sizeof(float) * count);
}

void model_moe_forward(const MoELayer* layer,
                       const float* input_vectors,
                       size_t token_count,
                       float* out_vectors) {
    size_t count = 0U;
    (void)layer;
    if (input_vectors == NULL || out_vectors == NULL) {
        return;
    }
    count = token_count * EMBED_DIM;
    memcpy(out_vectors, input_vectors, sizeof(float) * count);
}

void model_transformer_block_forward(const TransformerBlock* block,
                                     const float* input_vectors,
                                     size_t token_count,
                                     float* out_vectors) {
    if (block == NULL || input_vectors == NULL || out_vectors == NULL) {
        return;
    }
    model_attention_forward(&block->attention, input_vectors, token_count, out_vectors);
    model_moe_forward(&block->moe, out_vectors, token_count, out_vectors);
}
