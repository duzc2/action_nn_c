/**
 * @file transformer_config.h
 * @brief Config-only structs shared between user code and generated code.
 *
 * These structs intentionally stay small and POD-only so callers can pass them
 * through the profiler as opaque blobs without pulling in the full backend API.
 */

#ifndef TRANSFORMER_CONFIG_H
#define TRANSFORMER_CONFIG_H

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Minimal structural configuration needed to create a tiny transformer.
 */
typedef struct {
    size_t vocab_size;            /**< Token vocabulary size chosen by the user. */
    size_t model_dim;             /**< Internal hidden width used by embeddings and attention. */
    size_t max_seq_length;        /**< Maximum token sequence length chosen by the user. */
    size_t max_response_classes;  /**< Maximum learned answer-class count chosen by the user. */
    size_t max_text_length;       /**< Maximum stored answer text length, including the terminator. */
    uint32_t seed;                /**< Deterministic seed for reproducible parameter init. */
} TransformerModelConfig;

/**
 * @brief Minimal training configuration consumed by transformer_train_ops.c.
 */
typedef struct {
    float learning_rate;  /**< Scalar learning rate used by the tiny optimizer. */
} TransformerTrainConfig;

#endif
