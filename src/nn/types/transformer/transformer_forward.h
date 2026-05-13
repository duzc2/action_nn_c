/**
 * @file transformer_forward.h
 * @brief Unified transformer forward pass — single implementation shared by
 *        inference and training backends.
 */

#ifndef TRANSFORMER_FORWARD_H
#define TRANSFORMER_FORWARD_H

#include "transformer_infer_ops.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Forward flags ───────────────────────────────────────────────────── */

typedef enum {
    TF_FWD_INFER     = 0,       /**< Inference-only — no gradient state needed. */
    TF_FWD_TRAIN     = 1 << 0,  /**< Training mode — preserves intermediates. */
} TFForwardFlags;

/* ─── Unified forward cache ──────────────────────────────────────────── */

/**
 * @brief Scratch cache holding all intermediate tensors for one forward pass.
 *
 * Allocates 12 buffers (either arena-backed in infer, or calloc-backed in
 * training).  The struct layout is public so both sides can share the same
 * field names without casts.
 */
#ifndef TRANSFORMER_FORWARD_CACHE_DEFINED
#define TRANSFORMER_FORWARD_CACHE_DEFINED
struct TransformerForwardCache {
    size_t seq_length;
    size_t* tokens;
    float* input_states;
    float* query;
    float* key;
    float* value;
    float* attention;
    float* attended;
    float* projected;
    float* hidden;
    float* pooled;
    float* logits;
    float* probabilities;
};
#endif

/* ─── Shared helpers ─────────────────────────────────────────────────── */

size_t transformer_index(size_t row, size_t column, size_t column_count);
void   transformer_softmax(float* values, size_t count);
float  transformer_vector_norm(const float* values, size_t count);

/* ─── Unified forward pass ───────────────────────────────────────────── */

int transformer_run_forward(
    const TransformerInferContext* context,
    const char* question,
    struct TransformerForwardCache* cache
);

#ifdef __cplusplus
}
#endif

#endif /* TRANSFORMER_FORWARD_H */
