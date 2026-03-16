#include <string.h>

#include "../include/model.h"

/**
 * @brief Embedding 前向：按 token id 拷贝词向量。
 *
 * 关键保护点：
 * - 对负 id 和越界 id 回退到 0 号词，避免越界访问权重表。
 */
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

/**
 * @brief Attention 前向占位实现：当前直接透传输入。
 *
 * 背景：
 * - 当前项目目标是先打通端到端流程，后续可在此替换为真实注意力计算。
 */
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

/**
 * @brief MoE 前向占位实现：当前直接透传输入。
 */
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

/**
 * @brief Transformer Block 前向：先 attention 后 MoE。
 */
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
