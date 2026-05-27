/**
 * @file skip_connection.h
 * @brief Lightweight trainable residual (skip) connections – LAuReL-RW.
 *
 * Provides three skip modes that can be attached to any layer whose output
 * and input share the same shape:
 *
 *   SKIP_NONE      : y = F(x)                     (plain feed-forward)
 *   SKIP_IDENTITY  : y = F(x) + x                 (standard ResNet skip)
 *   SKIP_LAUREL_RW : y = α·F(x) + β·x             (learnable skip)
 *     where  α = sigmoid(α_raw),  β = 1 – α
 *
 * LAuReL-RW adds only 2 learnable scalars per layer (< 0.003 % overhead)
 * while delivering near-one-extra-layer gains.
 *
 * References:
 *   - Menghani et al., "LAuReL: Learned Augmented Residual Layer",
 *     ICML 2025 (arXiv:2411.07501) – Google
 *   - He et al., "Deep Residual Learning for Image Recognition", 2015
 *   - SCORE, arXiv 2603.10544
 */
#ifndef SKIP_CONNECTION_H
#define SKIP_CONNECTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Supported skip (residual connection) modes. */
typedef enum {
    SKIP_NONE      = 0,   /**< No skip: y = F(x) */
    SKIP_IDENTITY  = 1,   /**< Standard residual: y = F(x) + x */
    SKIP_LAUREL_RW = 2,   /**< Learnable weighted residual: y = α·F(x) + β·x */
} SkipMode;

/**
 * @brief Skip connection state (per-layer).
 *
 * For LAuReL-RW the struct holds two cached floats (α, β) written by
 * `skip_forward` and consumed by `skip_backward`.  The gradient
 * accumulator `grad_alpha_raw` is written (additive) by `skip_backward`
 * and should be zeroed before each backward pass by the caller.
 */
typedef struct {
    SkipMode mode;
    float alpha_raw;        /**< Raw learnable parameter (init 0 → α≈0.5) */
    float alpha_cache;      /**< sigmoid(alpha_raw), set during forward     */
    float beta_cache;       /**< 1 – alpha_cache,   set during forward     */
    float grad_alpha_raw;   /**< Accumulated gradient, written by backward  */
} SkipConnection;

/**
 * @brief Initialise a skip connection with default values.
 *
 * @param skip  pointer to skip state
 * @param mode  desired skip mode
 *
 * Sets alpha_raw = 0.0 (so α ≈ β ≈ 0.5 for LAuReL-RW) and resets
 * accumulated gradients to zero.
 */
void skip_init(SkipConnection *skip, SkipMode mode);

/**
 * @brief Forward pass – apply the residual connection.
 *
 * @param out         [n]  output (may alias `module_out` or `input`)
 * @param module_out  [n]  layer output F(x)
 * @param input       [n]  original input x to the layer
 * @param n                number of elements (must be > 0)
 * @param skip             skip connection state (updates caches)
 */
void skip_forward(float *restrict out,
                  const float *restrict module_out,
                  const float *restrict input,
                  size_t n,
                  SkipConnection *skip);

/**
 * @brief Backward pass – compute gradients through the skip connection.
 *
 * Writes d_module_out and d_input, and *adds* the gradient w.r.t. alpha_raw
 * into `skip->grad_alpha_raw`.
 *
 * @param d_module_out  [n]  output gradient for F(x)  (written)
 * @param d_input       [n]  output gradient for input (written)
 * @param module_out    [n]  forward F(x) value
 * @param input         [n]  forward input x value
 * @param dout          [n]  upstream gradient from loss
 * @param n                  number of elements
 * @param skip               skip connection state (reads caches, accumulates grad)
 */
void skip_backward(float *restrict d_module_out,
                   float *restrict d_input,
                   const float *restrict module_out,
                   const float *restrict input,
                   const float *restrict dout,
                   size_t n,
                   SkipConnection *skip);

#ifdef __cplusplus
}
#endif

#endif /* SKIP_CONNECTION_H */
