#include <math.h>
#include "../include/ops.h"

/**
 * @brief 验证输入输出张量形状一致性，减少算子重复样板代码。
 *
 * @param in   输入张量
 * @param out  输出张量
 * @return int TensorStatus
 */
static int validate_in_out_same_shape(const Tensor* in, const Tensor* out) {
    if (in == NULL || out == NULL || in->data == NULL || out->data == NULL) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }
    if (!tensor_same_shape(in, out)) {
        return TENSOR_STATUS_SHAPE_MISMATCH;
    }
    return TENSOR_STATUS_OK;
}

/**
 * @brief 2D 矩阵乘法：C = A x B。
 *
 * @param a 输入矩阵 A
 * @param b 输入矩阵 B
 * @param out 输出矩阵 C
 * @return int TensorStatus
 */
int op_matmul_2d(const Tensor* a, const Tensor* b, Tensor* out) {
    size_t m = 0U;
    size_t k = 0U;
    size_t n = 0U;
    size_t i = 0U;
    size_t j = 0U;
    size_t p = 0U;

    if (a == NULL || b == NULL || out == NULL ||
        a->data == NULL || b->data == NULL || out->data == NULL) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }
    if (a->ndim != 2U || b->ndim != 2U || out->ndim != 2U) {
        return TENSOR_STATUS_SHAPE_MISMATCH;
    }

    m = a->shape[0];
    k = a->shape[1];
    if (b->shape[0] != k) {
        return TENSOR_STATUS_SHAPE_MISMATCH;
    }
    n = b->shape[1];
    if (out->shape[0] != m || out->shape[1] != n) {
        return TENSOR_STATUS_SHAPE_MISMATCH;
    }

    for (i = 0U; i < m; ++i) {
        for (j = 0U; j < n; ++j) {
            float acc = 0.0f;
            /* 核心算法说明：
             * - 按标准三重循环实现，优先保证 C99 可移植性。
             * - 这里不做平台相关 SIMD，后续可在同签名下替换优化实现。 */
            for (p = 0U; p < k; ++p) {
                acc += a->data[i * k + p] * b->data[p * n + j];
            }
            out->data[i * n + j] = acc;
        }
    }
    return TENSOR_STATUS_OK;
}

/**
 * @brief 按最后一维做 Softmax，支持 1D/2D 张量。
 *
 * @param in   输入张量
 * @param out  输出张量（与输入同形状）
 * @return int TensorStatus
 */
int op_softmax_last_dim(const Tensor* in, Tensor* out) {
    size_t outer = 0U;
    size_t inner = 0U;
    size_t row = 0U;
    int rc = validate_in_out_same_shape(in, out);

    if (rc != TENSOR_STATUS_OK) {
        return rc;
    }
    if (in->ndim == 0U || in->ndim > 2U) {
        return TENSOR_STATUS_SHAPE_MISMATCH;
    }

    inner = in->shape[in->ndim - 1U];
    outer = in->numel / inner;

    for (row = 0U; row < outer; ++row) {
        size_t col = 0U;
        float max_val = in->data[row * inner];
        float sum = 0.0f;

        /* 关键保护点：
         * - 先减去行最大值，再做 exp，避免指数溢出。 */
        for (col = 1U; col < inner; ++col) {
            float v = in->data[row * inner + col];
            if (v > max_val) {
                max_val = v;
            }
        }

        for (col = 0U; col < inner; ++col) {
            float e = expf(in->data[row * inner + col] - max_val);
            out->data[row * inner + col] = e;
            sum += e;
        }

        if (sum == 0.0f) {
            return TENSOR_STATUS_INVALID_ARGUMENT;
        }
        for (col = 0U; col < inner; ++col) {
            out->data[row * inner + col] /= sum;
        }
    }
    return TENSOR_STATUS_OK;
}

/**
 * @brief 按最后一维做 RMSNorm：y = (x / rms(x)) * weight。
 *
 * @param in      输入张量
 * @param weight  归一化缩放参数（1D）
 * @param eps     防止除零的小常数
 * @param out     输出张量（与输入同形状）
 * @return int TensorStatus
 */
int op_rmsnorm_last_dim(const Tensor* in, const Tensor* weight, float eps, Tensor* out) {
    size_t outer = 0U;
    size_t inner = 0U;
    size_t row = 0U;
    int rc = validate_in_out_same_shape(in, out);

    if (rc != TENSOR_STATUS_OK) {
        return rc;
    }
    if (weight == NULL || weight->data == NULL || weight->ndim != 1U) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    inner = in->shape[in->ndim - 1U];
    outer = in->numel / inner;
    if (weight->shape[0] != inner) {
        return TENSOR_STATUS_SHAPE_MISMATCH;
    }

    for (row = 0U; row < outer; ++row) {
        size_t col = 0U;
        float mean_sq = 0.0f;
        float inv_rms = 0.0f;

        for (col = 0U; col < inner; ++col) {
            float v = in->data[row * inner + col];
            mean_sq += v * v;
        }
        mean_sq /= (float)inner;
        inv_rms = 1.0f / sqrtf(mean_sq + eps);

        for (col = 0U; col < inner; ++col) {
            out->data[row * inner + col] =
                in->data[row * inner + col] * inv_rms * weight->data[col];
        }
    }
    return TENSOR_STATUS_OK;
}

/**
 * @brief GELU 激活（tanh 近似版）。
 *
 * @param in   输入张量
 * @param out  输出张量（与输入同形状）
 * @return int TensorStatus
 */
int op_gelu(const Tensor* in, Tensor* out) {
    size_t i = 0U;
    int rc = validate_in_out_same_shape(in, out);
    if (rc != TENSOR_STATUS_OK) {
        return rc;
    }

    for (i = 0U; i < in->numel; ++i) {
        float x = in->data[i];
        float x3 = x * x * x;
        const float coeff = 0.7978845608028654f; /* sqrt(2/pi) */
        const float cubic = 0.044715f;
        out->data[i] = 0.5f * x * (1.0f + tanhf(coeff * (x + cubic * x3)));
    }
    return TENSOR_STATUS_OK;
}

/**
 * @brief 输出执行头激活映射，按通道应用 Sigmoid/Tanh。
 *
 * @param in            输入向量（1D）
 * @param activations   每个通道激活函数类型数组
 * @param out           输出向量（1D）
 * @return int TensorStatus
 */
int op_actuator(const Tensor* in, const int* activations, Tensor* out) {
    size_t i = 0U;
    int rc = validate_in_out_same_shape(in, out);

    if (rc != TENSOR_STATUS_OK) {
        return rc;
    }
    if (activations == NULL || in->ndim != 1U) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0U; i < in->numel; ++i) {
        float x = in->data[i];
        if (activations[i] == ACTUATOR_SIGMOID) {
            out->data[i] = 1.0f / (1.0f + expf(-x));
        } else {
            out->data[i] = tanhf(x);
        }
    }
    return TENSOR_STATUS_OK;
}
