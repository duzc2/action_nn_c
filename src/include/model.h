#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>

#include "config_user.h"

/**
 * @brief 词嵌入层描述。
 *
 * 背景：
 * - 当前工程以轻量推理为目标，词表与嵌入表通常由外部训练/导出流程提供。
 */
typedef struct EmbeddingLayer {
    const float* table;
    size_t vocab_size;
    size_t embed_dim;
} EmbeddingLayer;

/**
 * @brief 注意力层描述（当前版本为占位接口定义）。
 */
typedef struct AttentionLayer {
    size_t embed_dim;
    size_t num_heads;
} AttentionLayer;

/**
 * @brief 混合专家层描述（当前版本为占位接口定义）。
 */
typedef struct MoELayer {
    size_t num_experts;
    size_t k_top;
} MoELayer;

/**
 * @brief Transformer Block 结构定义。
 */
typedef struct TransformerBlock {
    AttentionLayer attention;
    MoELayer moe;
} TransformerBlock;

/**
 * @brief 执行 embedding 前向计算。
 */
void model_embedding_forward(const EmbeddingLayer* layer,
                             const int* token_ids,
                             size_t token_count,
                             float* out_vectors);

/**
 * @brief 执行 attention 前向计算。
 */
void model_attention_forward(const AttentionLayer* layer,
                             const float* input_vectors,
                             size_t token_count,
                             float* out_vectors);

/**
 * @brief 执行 MoE 前向计算。
 */
void model_moe_forward(const MoELayer* layer,
                       const float* input_vectors,
                       size_t token_count,
                       float* out_vectors);

/**
 * @brief 执行单个 Transformer Block 前向计算。
 */
void model_transformer_block_forward(const TransformerBlock* block,
                                     const float* input_vectors,
                                     size_t token_count,
                                     float* out_vectors);

#endif
