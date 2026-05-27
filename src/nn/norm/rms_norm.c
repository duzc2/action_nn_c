/**
 * @file rms_norm.c
 * @brief RMS Normalization implementation – forward and backward passes.
 */

#include "norm/rms_norm.h"
#include <math.h>

#ifndef ACTION_C_OK
#define ACTION_C_OK            0
#define ACTION_C_ERR_NULL     -1
#define ACTION_C_ERR_INVALID  -2
#endif

/* ------------------------------------------------------------------ */
/*  Forward pass                                                       */
/* ------------------------------------------------------------------ */

int rms_norm_forward(float *restrict output,
                     const float *restrict input,
                     const float *restrict gamma,
                     size_t n,
                     float epsilon)
{
    if (output == NULL || input == NULL || n == 0) return ACTION_C_ERR_NULL;

    /* Step 1: sum of squares */
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float xi = input[i];
        sum_sq += xi * xi;
    }

    /* Step 2: r = 1 / sqrt(E[x^2] + eps) */
    float r = 1.0f / sqrtf(sum_sq / (float)n + epsilon);

    /* Step 3: y_i = x_i * r * gamma_i  (or y_i = x_i * r if gamma is NULL) */
    if (gamma != NULL) {
        for (size_t i = 0; i < n; ++i) {
            output[i] = input[i] * r * gamma[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            output[i] = input[i] * r;
        }
    }

    return ACTION_C_OK;
}

/* ------------------------------------------------------------------ */
/*  Backward pass                                                      */
/* ------------------------------------------------------------------ */

int rms_norm_backward(float *restrict d_input,
                      float *restrict d_gamma,
                      const float *restrict d_output,
                      const float *restrict input,
                      const float *restrict gamma,
                      size_t n,
                      float epsilon)
{
    if (d_input == NULL || d_output == NULL || input == NULL || n == 0) {
        return ACTION_C_ERR_NULL;
    }

    /* Recompute r = 1 / sqrt(E[x^2] + eps) from the forward input */
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float xi = input[i];
        sum_sq += xi * xi;
    }
    float r = 1.0f / sqrtf(sum_sq / (float)n + epsilon);

    /* d_gamma_i += dout_i * u_i  where u_i = x_i * r */
    if (d_gamma != NULL) {
        if (gamma != NULL) {
            /* gamma != NULL, d_gamma != NULL */
            float sum_dxo = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                float ui = input[i] * r;
                d_gamma[i] += d_output[i] * ui;
                sum_dxo += d_output[i] * input[i] * gamma[i];
            }

            /* d_input_i = r * (dout_i * gamma_i - x_i * r^2 * sum_dxo / n) */
            float r2_n = (r * r) / (float)n;
            for (size_t i = 0; i < n; ++i) {
                d_input[i] = r * (d_output[i] * gamma[i] - input[i] * r2_n * sum_dxo);
            }
        } else {
            /* gamma == NULL (≡ 1), d_gamma != NULL */
            float sum_dxo = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                float ui = input[i] * r;
                d_gamma[i] += d_output[i] * ui;
                sum_dxo += d_output[i] * input[i];
            }

            float r2_n = (r * r) / (float)n;
            for (size_t i = 0; i < n; ++i) {
                d_input[i] = r * (d_output[i] - input[i] * r2_n * sum_dxo);
            }
        }
    } else {
        /* d_gamma == NULL (caller doesn't want gamma gradients) */
        if (gamma != NULL) {
            float sum_dxo = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                sum_dxo += d_output[i] * input[i] * gamma[i];
            }
            float r2_n = (r * r) / (float)n;
            for (size_t i = 0; i < n; ++i) {
                d_input[i] = r * (d_output[i] * gamma[i] - input[i] * r2_n * sum_dxo);
            }
        } else {
            float sum_dxo = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                sum_dxo += d_output[i] * input[i];
            }
            float r2_n = (r * r) / (float)n;
            for (size_t i = 0; i < n; ++i) {
                d_input[i] = r * (d_output[i] - input[i] * r2_n * sum_dxo);
            }
        }
    }

    return ACTION_C_OK;
}
