#ifndef OPS_H
#define OPS_H

#include "tensor.h"

/**
 * @brief 输出通道激活类型，需与 config_user.h 中约定保持一致。
 */
typedef enum ActuatorActivation {
    ACTUATOR_SIGMOID = 0,
    ACTUATOR_TANH = 1
} ActuatorActivation;

/**
 * @brief 2D 矩阵乘法：C = A x B。
 *
 * 约束：
 * - A: [M, K]
 * - B: [K, N]
 * - C: [M, N]
 *
 * @param a 输入矩阵 A
 * @param b 输入矩阵 B
 * @param out 输出矩阵 C
 * @return int TensorStatus
 */
int op_matmul_2d(const Tensor* a, const Tensor* b, Tensor* out);

/**
 * @brief 按最后一维做 Softmax，支持 1D/2D 张量。
 *
 * @param in   输入张量
 * @param out  输出张量（与输入同形状）
 * @return int TensorStatus
 */
int op_softmax_last_dim(const Tensor* in, Tensor* out);

/**
 * @brief 按最后一维做 RMSNorm：y = (x / rms(x)) * weight。
 *
 * 约束：
 * - weight 必须是一维，长度等于输入最后一维。
 *
 * @param in      输入张量
 * @param weight  归一化缩放参数（1D）
 * @param eps     防止除零的小常数
 * @param out     输出张量（与输入同形状）
 * @return int TensorStatus
 */
int op_rmsnorm_last_dim(const Tensor* in, const Tensor* weight, float eps, Tensor* out);

/**
 * @brief GELU 激活（近似公式），逐元素计算。
 *
 * @param in   输入张量
 * @param out  输出张量（与输入同形状）
 * @return int TensorStatus
 */
int op_gelu(const Tensor* in, Tensor* out);

/**
 * @brief 输出执行头激活映射，按通道应用 Sigmoid/Tanh。
 *
 * @param in            输入向量（1D）
 * @param activations   每个通道激活函数类型数组
 * @param out           输出向量（1D）
 * @return int TensorStatus
 */
int op_actuator(const Tensor* in, const int* activations, Tensor* out);

#endif /* OPS_H */
