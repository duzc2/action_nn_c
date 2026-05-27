/**
 * @file rms_norm.h
 * @brief RMS Normalization – shared normalisation layer for all network types.
 *
 * RMSNorm discards the mean-subtraction step of LayerNorm, keeping only the
 * root-mean-square scaling.  The operation is:
 *
 *   y_i = x_i / sqrt(E[x^2] + epsilon) * gamma_i
 *
 * where E[x^2] is the mean of the squared activations across the feature
 * dimension and gamma is an optional learnable scale vector.
 *
 * References:
 *   - Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019
 *   - Peri-LN, ICML 2025 (shows mean-subtraction is redundant, RMSNorm
 *     combined with Pre-LN architecture gives better training stability)
 */
#ifndef RMS_NORM_H
#define RMS_NORM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief RMSNorm forward pass.
 *
 * @param output  [n]      output buffer, may alias `input`
 * @param input   [n]      input activations
 * @param gamma   [n]      learnable scale (NULL = all-ones, i.e. gamma_i ≡ 1)
 * @param n                feature dimension
 * @param epsilon          numerical stability constant (typical 1e-5)
 * @return ACTION_C_OK (0) on success, negative error code on failure
 */
int rms_norm_forward(float *restrict output,
                     const float *restrict input,
                     const float *restrict gamma,
                     size_t n,
                     float epsilon);

/**
 * @brief RMSNorm backward pass.
 *
 * Computes gradient w.r.t. the input and accumulates gradient w.r.t. gamma.
 * The caller *must* zero `d_gamma` before the first backward call of a batch
 * and sum across batch elements if mini-batch is used.
 *
 * @param d_input   [n]      output gradient for input (written)
 * @param d_gamma   [n]      accumulated gradient for gamma (read-modify-write)
 * @param d_output  [n]      upstream gradient from loss
 * @param input     [n]      forward-pass input (same values that were fed
 *                            to rms_norm_forward)
 * @param gamma     [n]      forward gamma (NULL = all-ones)
 * @param n                  feature dimension
 * @param epsilon            same epsilon used in forward pass
 * @return ACTION_C_OK (0) on success, negative error code on failure
 */
int rms_norm_backward(float *restrict d_input,
                      float *restrict d_gamma,
                      const float *restrict d_output,
                      const float *restrict input,
                      const float *restrict gamma,
                      size_t n,
                      float epsilon);

#ifdef __cplusplus
}
#endif

#endif /* RMS_NORM_H */
