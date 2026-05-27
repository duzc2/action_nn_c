/**
 * @file dropout.c
 * @brief Dropout and Stochastic Depth implementations.
 */

#include "dropout/dropout.h"
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/*  xorshift32  –  fast, zero-libc, deterministic PRNG                 */
/* ------------------------------------------------------------------ */
static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/**
 * @brief Return a float in [0, 1).
 *
 * Uses the upper 23 bits of a xorshift32 output (leaving the lower 9 bits
 * which have weaker randomness).  This gives ~1e-7 resolution – enough
 * for compare-against-keep_prob.
 */
static float randf(uint32_t *state)
{
    return (float)(xorshift32(state) >> 9) * (1.0f / 8388608.0f);  /* 2^-23 */
}

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

void dropout_init(DropoutLayer *dp, float keep_prob, uint32_t seed)
{
    if (dp == NULL) return;
    if (keep_prob < 0.0f) keep_prob = 0.0f;
    if (keep_prob > 1.0f) keep_prob = 1.0f;
    dp->keep_prob       = keep_prob;
    dp->rng_state       = (seed != 0) ? seed : 1u;  /* never let state == 0 */
    dp->training        = 1;
    dp->mask            = NULL;
    dp->mask_capacity   = 0;
    dp->drop_path_flag  = 1;
}

void dropout_free(DropoutLayer *dp)
{
    if (dp == NULL) return;
    free(dp->mask);
    dp->mask            = NULL;
    dp->mask_capacity   = 0;
}

/* ------------------------------------------------------------------ */
/*  Per-element Dropout                                                */
/* ------------------------------------------------------------------ */

void dropout_forward(float *restrict y,
                     const float *restrict x,
                     size_t n,
                     DropoutLayer *dp)
{
    if (y == NULL || x == NULL || dp == NULL || n == 0) return;

    if (!dp->training  ||  dp->keep_prob >= 1.0f) {
        /* Inference mode or keep_prob==1 → passthrough */
        for (size_t i = 0; i < n; ++i) y[i] = x[i];
        return;
    }

    if (dp->keep_prob <= 0.0f) {
        /* keep_prob==0 → zero everything */
        for (size_t i = 0; i < n; ++i) y[i] = 0.0f;
        return;
    }

    /* Ensure mask buffer is large enough */
    if (dp->mask == NULL || dp->mask_capacity < n) {
        free(dp->mask);
        dp->mask = (uint8_t *)malloc(n * sizeof(uint8_t));
        dp->mask_capacity = (dp->mask != NULL) ? n : 0;
    }

    float scale = 1.0f / dp->keep_prob;

    if (dp->mask != NULL) {
        for (size_t i = 0; i < n; ++i) {
            int keep = (randf(&dp->rng_state) < dp->keep_prob);
            dp->mask[i] = (uint8_t)keep;
            y[i] = keep ? x[i] * scale : 0.0f;
        }
    } else {
        /* allocation failed – degrade gracefully (passthrough) */
        for (size_t i = 0; i < n; ++i) y[i] = x[i];
    }
}

void dropout_backward(float *restrict dx,
                      const float *restrict dy,
                      size_t n,
                      const DropoutLayer *dp)
{
    if (dx == NULL || dy == NULL || dp == NULL || n == 0) return;

    if (!dp->training  ||  dp->keep_prob >= 1.0f) {
        for (size_t i = 0; i < n; ++i) dx[i] = dy[i];
        return;
    }

    if (dp->keep_prob <= 0.0f) {
        for (size_t i = 0; i < n; ++i) dx[i] = 0.0f;
        return;
    }

    float scale = 1.0f / dp->keep_prob;

    if (dp->mask != NULL && dp->mask_capacity >= n) {
        for (size_t i = 0; i < n; ++i) {
            dx[i] = dp->mask[i] ? dy[i] * scale : 0.0f;
        }
    } else {
        /* no mask cached (shouldn't happen normally) – passthrough */
        for (size_t i = 0; i < n; ++i) dx[i] = dy[i];
    }
}

/* ------------------------------------------------------------------ */
/*  Stochastic Depth / DropPath                                        */
/* ------------------------------------------------------------------ */

void droppath_forward(float *restrict y,
                      const float *restrict fx,
                      const float *restrict x,
                      size_t n,
                      DropoutLayer *dp)
{
    if (y == NULL || fx == NULL || x == NULL || dp == NULL || n == 0) return;

    if (!dp->training  ||  dp->keep_prob >= 1.0f) {
        /* Inference or guaranteed keep: y = fx + x */
        dp->drop_path_flag = 1;
        for (size_t i = 0; i < n; ++i) y[i] = fx[i] + x[i];
        return;
    }

    if (dp->keep_prob <= 0.0f) {
        /* Always drop: y = x */
        dp->drop_path_flag = 0;
        for (size_t i = 0; i < n; ++i) y[i] = x[i];
        return;
    }

    /* Stochastic decision for the whole path */
    int keep = (randf(&dp->rng_state) < dp->keep_prob);
    dp->drop_path_flag = keep;

    float scale = 1.0f / dp->keep_prob;
    if (keep) {
        for (size_t i = 0; i < n; ++i) y[i] = fx[i] * scale + x[i];
    } else {
        for (size_t i = 0; i < n; ++i) y[i] = x[i];
    }
}

void droppath_backward(float *restrict d_fx,
                       float *restrict dx,
                       const float *restrict dy,
                       size_t n,
                       const DropoutLayer *dp)
{
    if (d_fx == NULL || dx == NULL || dy == NULL || dp == NULL || n == 0) return;

    if (!dp->training  ||  dp->keep_prob >= 1.0f) {
        /* Inference / guaranteed keep: full gradient to both paths */
        for (size_t i = 0; i < n; ++i) {
            d_fx[i] = dy[i];
            dx[i]   = dy[i];
        }
        return;
    }

    if (dp->keep_prob <= 0.0f  ||  !dp->drop_path_flag) {
        /* Path was dropped → no gradient flows to F(x) */
        for (size_t i = 0; i < n; ++i) {
            d_fx[i] = 0.0f;
            dx[i]   = dy[i];
        }
        return;
    }

    /* Path was kept → scale gradient by 1/keep_prob */
    float scale = 1.0f / dp->keep_prob;
    for (size_t i = 0; i < n; ++i) {
        d_fx[i] = dy[i] * scale;
        dx[i]   = dy[i];
    }
}
