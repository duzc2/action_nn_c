#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

/**
 * @brief 张量最大维度，当前按通用 4 维约束设计，覆盖 [B, S, H, D] 常见布局。
 */
#define TENSOR_MAX_DIMS 4

/**
 * @brief 统一错误码定义，便于上层做确定性错误处理。
 */
typedef enum TensorStatus {
    TENSOR_STATUS_OK = 0,
    TENSOR_STATUS_INVALID_ARGUMENT = -1,
    TENSOR_STATUS_POOL_EXHAUSTED = -2,
    TENSOR_STATUS_SHAPE_MISMATCH = -3
} TensorStatus;

/**
 * @brief Tensor 结构体仅保存视图信息，不拥有内存生命周期。
 *
 * 设计背景：
 * - 为满足嵌入式和 C99 场景，避免在核心算子中做动态分配。
 * - 数据内存来自外部池或调用方提供缓冲区。
 */
typedef struct Tensor {
    float* data;
    size_t ndim;
    size_t shape[TENSOR_MAX_DIMS];
    size_t stride[TENSOR_MAX_DIMS];
    size_t numel;
} Tensor;

/**
 * @brief 线性内存池，服务于 Tensor 数据缓冲。
 */
typedef struct TensorPool {
    float* buffer;
    size_t capacity;
    size_t used;
} TensorPool;

/**
 * @brief 根据 shape 计算元素总数。
 *
 * @param ndim      维度数量
 * @param shape     每维大小
 * @param out_numel 输出元素数
 * @return int      TensorStatus
 */
int tensor_calc_numel(size_t ndim, const size_t* shape, size_t* out_numel);

/**
 * @brief 用已有数据缓冲初始化张量视图。
 *
 * @param t         张量对象
 * @param data      数据指针
 * @param ndim      维度数量
 * @param shape     形状数组
 * @return int      TensorStatus
 */
int tensor_init_view(Tensor* t, float* data, size_t ndim, const size_t* shape);

/**
 * @brief 用常数填充张量。
 *
 * @param t      张量对象
 * @param value  填充值
 */
void tensor_fill(Tensor* t, float value);

/**
 * @brief 初始化线性内存池。
 *
 * @param pool     内存池对象
 * @param buffer   外部缓冲区（float*）
 * @param capacity 缓冲容量（float 元素数）
 * @return int     TensorStatus
 */
int tensor_pool_init(TensorPool* pool, float* buffer, size_t capacity);

/**
 * @brief 重置内存池分配游标，不清空数据内容。
 *
 * @param pool 内存池对象
 */
void tensor_pool_reset(TensorPool* pool);

/**
 * @brief 从内存池中分配一个张量并初始化其 shape/stride。
 *
 * @param pool   内存池
 * @param t      输出张量
 * @param ndim   维度数量
 * @param shape  形状数组
 * @return int   TensorStatus
 */
int tensor_pool_alloc(TensorPool* pool, Tensor* t, size_t ndim, const size_t* shape);

/**
 * @brief 判断两个张量 shape 是否一致。
 *
 * @param a 张量 A
 * @param b 张量 B
 * @return int 1 表示一致，0 表示不一致
 */
int tensor_same_shape(const Tensor* a, const Tensor* b);

#endif /* TENSOR_H */
