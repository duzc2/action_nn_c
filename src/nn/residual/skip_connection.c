/**
 * @file skip_connection.c
 * @brief Residual (skip) connection – forward and backward passes.
 */

#include "residual/skip_connection.h"
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Initialisation                                                     */
/* ------------------------------------------------------------------ */

void skip_init(SkipConnection *skip, SkipMode mode)
{
    if (skip == NULL) return;
    skip->mode           = mode;
    skip->alpha_raw      = 0.0f;   /* → sigmoid(0) = 0.5, β = 0.5 */
    skip->alpha_cache    = 0.5f;
    skip->beta_cache     = 0.5f;
    skip->grad_alpha_raw = 0.0f;
}

/* ------------------------------------------------------------------ */
/*  Forward pass                                                       */
/* ------------------------------------------------------------------ */

void skip_forward(float *restrict out,
                  const float *restrict module_out,
                  const float *restrict input,
                  size_t n,
                  SkipConnection *skip)
{
    if (out == NULL || module_out == NULL || input == NULL || skip == NULL) return;
    if (n == 0) return;

    switch (skip->mode) {
    case SKIP_NONE:
        /* y = F(x) — plain copy (caller may have already done this) */
        for (size_t i = 0; i < n; ++i) {
            out[i] = module_out[i];
        }
        break;

    case SKIP_IDENTITY:
        /* y = F(x) + x */
        for (size_t i = 0; i < n; ++i) {
            out[i] = module_out[i] + input[i];
        }
        break;

    case SKIP_LAUREL_RW: {
        /* α = sigmoid(α_raw),  β = 1 – α
         * y = α·F(x) + β·x */
        float alpha = 1.0f / (1.0f + expf(-skip->alpha_raw));
        float beta  = 1.0f - alpha;

        /* Cache for backward pass */
        skip->alpha_cache = alpha;
        skip->beta_cache  = beta;

        for (size_t i = 0; i < n; ++i) {
            out[i] = alpha * module_out[i] + beta * input[i];
        }
        break;
    }

    default:
        break;
    }
}

/* ------------------------------------------------------------------ */
/*  Backward pass                                                      */
/* ------------------------------------------------------------------ */

void skip_backward(float *restrict d_module_out,
                   float *restrict d_input,
                   const float *restrict module_out,
                   const float *restrict input,
                   const float *restrict dout,
                   size_t n,
                   SkipConnection *skip)
{
    if (d_module_out == NULL || d_input == NULL
        || module_out == NULL || input == NULL
        || dout == NULL || skip == NULL) return;
    if (n == 0) return;

    switch (skip->mode) {
    case SKIP_NONE:
        /* dF = dout,  dx = 0 */
        for (size_t i = 0; i < n; ++i) {
            d_module_out[i] = dout[i];
            d_input[i]      = 0.0f;
        }
        break;

    case SKIP_IDENTITY:
        /* dF = dout,  dx = dout */
        for (size_t i = 0; i < n; ++i) {
            d_module_out[i] = dout[i];
            d_input[i]      = dout[i];
        }
        break;

    case SKIP_LAUREL_RW: {
        float alpha = skip->alpha_cache;   /* from forward */
        float beta  = skip->beta_cache;

        /* Per-element gradients */
        for (size_t i = 0; i < n; ++i) {
            d_module_out[i] = dout[i] * alpha;
            d_input[i]      = dout[i] * beta;
        }

        /* Gradient w.r.t. alpha_raw:
         *   dL/dα = Σ dout_i · (F_i – x_i)
         *   dα/dα_raw = α · (1 – α)  = α · β
         *   grad_alpha_raw += Σ_i dout_i · (F_i – x_i) · α · β
         */
        float d_alpha = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            d_alpha += dout[i] * (module_out[i] - input[i]);
        }
        skip->grad_alpha_raw += d_alpha * alpha * beta;
        break;
    }

    default:
        break;
    }
}
