/**
 * @file infer_runtime.h
 * @brief 推理运行时接口
 *
 * 推理库是独立运行的，不包含任何训练代码。
 * 用户代码通过 nn_infer_runtime_step() 执行推理。
 */

#ifndef INFER_RUNTIME_H
#define INFER_RUNTIME_H

/**
 * @brief 推理请求结构体
 *
 * 包含：
 * - network_type: 网络类型名称（如 "mlp", "transformer"）
 * - context: 网络上下文，由 xxx_create() 创建
 *
 * 注意：用户不需要关心具体的网络实现，
 * 只需要通过注册表查找对应的推理函数即可。
 */
typedef struct {
    const char* network_type;  /**< 网络类型名称 */
    void* context;             /**< 网络上下文 */
} NNInferRequest;

/**
 * @brief 执行一步推理
 *
 * 通过网络注册表查找对应类型的推理函数，
 * 并调用该函数执行推理。
 *
 * @param request 推理请求
 * @return 0 成功，非0 失败
 */
int nn_infer_runtime_step(const NNInferRequest* request);

#endif
