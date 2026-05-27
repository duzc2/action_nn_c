/**
 * @file dropout.h
 * @brief Dropout and Stochastic Depth (DropPath) regularisation layers.
 *
 *   Dropout   – independently masks individual neurons during training;
 *               during inference it is a no-op.
 *   DropPath  – randomly drops an entire residual path (Stochastic Depth,
 *               Huang et al. 2016); keeps the identity shortcut in training,
 *               uses full path in inference.
 *
 * Both modes use **inverted dropout**: active elements are scaled by
 * 1/keep_prob during training so that the expected output magnitude is
 * preserved.  No scaling is applied in inference mode.
 *
 * RNG: xorshift32 – fast, deterministic, zero-libc, suitable for MCU builds.
 *
 * References:
 *   - Dropout, Srivastava et al. 2014
 *   - Stochastic Depth, Huang et al. 2016
 *   - Lipschitz-Guided adaptive Dropout, Nayal et al., arXiv 2509.10298
 */
#ifndef DROPOUT_H
#define DROPOUT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Dropout / DropPath layer state.
 *
 * The same struct is used for both per-element Dropout and DropPath.
 * An internal bit-array (`mask`) caches the element-wise decisions made
 * during `dropout_forward` so that `dropout_backward` can replay them.
 * For DropPath, a single flag `drop_path_flag` records whether the whole
 * residual path was kept (=1) or dropped (=0).
 */
typedef struct {
    float    keep_prob;         /**< Probability of keeping an element/path */
    uint32_t rng_state;         /**< xorshift32 state (seedable) */
    int      training;          /**< 1 = training (stochastic), 0 = inference (no-op) */

    /* Cached forward state – written by forward, read by backward */
    uint8_t *mask;              /**< Per-element 0/1 mask for Dropout (owned) */
    size_t   mask_capacity;     /**< Allocated size of `mask` in bytes */
    int      drop_path_flag;    /**< 1 = path kept, 0 = path dropped (DropPath) */
} DropoutLayer;

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

/**
 * @brief Initialise a DropoutLayer.
 * @param dp          layer state (must be non-NULL)
 * @param keep_prob   retention probability [0..1]; 1.0 = keep all
 * @param seed        RNG seed (deterministic across runs)
 */
void dropout_init(DropoutLayer *dp, float keep_prob, uint32_t seed);

/**
 * @brief Release internal resources (mask buffer).
 */
void dropout_free(DropoutLayer *dp);

/**
 * @brief Switch between training and inference mode.
 *
 * In inference mode (`training = 0`) both dropout_forward and
 * droppath_forward are identity passthroughs.
 */
static inline void dropout_set_training(DropoutLayer *dp, int training)
{
    if (dp != NULL) dp->training = training;
}

/* ------------------------------------------------------------------ */
/*  Per-element Dropout                                                */
/* ------------------------------------------------------------------ */

/**
 * @brief Forward pass – apply dropout mask.
 *
 * @param y   [n]  output (may alias `x`)
 * @param x   [n]  input activations
 * @param n        number of elements
 * @param dp       dropout state (updates internal mask)
 *
 * Training:   y[i] = mask[i] ? (x[i] / keep_prob) : 0
 * Inference:  y[i] = x[i]
 */
void dropout_forward(float *restrict y,
                     const float *restrict x,
                     size_t n,
                     DropoutLayer *dp);

/**
 * @brief Backward pass – propagate gradient through the dropout mask.
 *
 * @param dx  [n]  output gradient for input (written)
 * @param dy  [n]  upstream gradient from loss
 * @param n        number of elements
 * @param dp       dropout state (reads internal mask from forward)
 *
 * Training:   dx[i] = mask[i] ? (dy[i] / keep_prob) : 0
 * Inference:  dx[i] = dy[i]
 */
void dropout_backward(float *restrict dx,
                      const float *restrict dy,
                      size_t n,
                      const DropoutLayer *dp);

/* ------------------------------------------------------------------ */
/*  Stochastic Depth / DropPath                                        */
/* ------------------------------------------------------------------ */

/**
 * @brief DropPath forward pass – possibly skip an entire layer.
 *
 * @param y   [n]  output (may alias `fx` or `x`)
 * @param fx  [n]  layer output F(x)
 * @param x   [n]  original input (identity shortcut)
 * @param n        number of elements
 * @param dp       dropout state (updates drop_path_flag)
 *
 * Training, path kept:   y[i] = fx[i] / keep_prob + x[i]
 * Training, path dropped: y[i] = x[i]
 * Inference:             y[i] = fx[i] + x[i]
 */
void droppath_forward(float *restrict y,
                      const float *restrict fx,
                      const float *restrict x,
                      size_t n,
                      DropoutLayer *dp);

/**
 * @brief DropPath backward pass.
 *
 * @param d_fx  [n]  gradient for F(x)  (written)
 * @param dx    [n]  gradient for input (written)
 * @param dy    [n]  upstream gradient
 * @param n          number of elements
 * @param dp         dropout state (reads drop_path_flag)
 */
void droppath_backward(float *restrict d_fx,
                       float *restrict dx,
                       const float *restrict dy,
                       size_t n,
                       const DropoutLayer *dp);

#ifdef __cplusplus
}
#endif

#endif /* DROPOUT_H */
