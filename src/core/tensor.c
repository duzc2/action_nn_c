#include "../include/tensor.h"

/**
 * @brief 计算连续内存布局下的 stride。
 *
 * @param ndim      维度数量
 * @param shape     形状数组
 * @param stride    输出 stride 数组
 */
static void tensor_compute_stride(size_t ndim, const size_t* shape, size_t* stride) {
    size_t i = 0U;
    size_t running = 1U;

    for (i = ndim; i > 0U; --i) {
        stride[i - 1U] = running;
        running *= shape[i - 1U];
    }
}

/**
 * @brief 根据 shape 计算元素总数。
 *
 * @param ndim      维度数量
 * @param shape     每维大小
 * @param out_numel 输出元素数
 * @return int      TensorStatus
 */
int tensor_calc_numel(size_t ndim, const size_t* shape, size_t* out_numel) {
    size_t i = 0U;
    size_t numel = 1U;

    if (shape == NULL || out_numel == NULL || ndim == 0U || ndim > TENSOR_MAX_DIMS) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0U; i < ndim; ++i) {
        if (shape[i] == 0U) {
            return TENSOR_STATUS_INVALID_ARGUMENT;
        }
        numel *= shape[i];
    }

    *out_numel = numel;
    return TENSOR_STATUS_OK;
}

/**
 * @brief 用已有数据缓冲初始化张量视图。
 *
 * @param t         张量对象
 * @param data      数据指针
 * @param ndim      维度数量
 * @param shape     形状数组
 * @return int      TensorStatus
 */
int tensor_init_view(Tensor* t, float* data, size_t ndim, const size_t* shape) {
    size_t numel = 0U;
    size_t i = 0U;
    int rc = 0;

    if (t == NULL || data == NULL) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    rc = tensor_calc_numel(ndim, shape, &numel);
    if (rc != TENSOR_STATUS_OK) {
        return rc;
    }

    t->data = data;
    t->ndim = ndim;
    t->numel = numel;
    for (i = 0U; i < TENSOR_MAX_DIMS; ++i) {
        t->shape[i] = (i < ndim) ? shape[i] : 1U;
        t->stride[i] = 0U;
    }
    tensor_compute_stride(ndim, shape, t->stride);
    return TENSOR_STATUS_OK;
}

/**
 * @brief 用常数填充张量。
 *
 * @param t      张量对象
 * @param value  填充值
 */
void tensor_fill(Tensor* t, float value) {
    size_t i = 0U;
    if (t == NULL || t->data == NULL) {
        return;
    }
    for (i = 0U; i < t->numel; ++i) {
        t->data[i] = value;
    }
}

/**
 * @brief 初始化线性内存池。
 *
 * @param pool     内存池对象
 * @param buffer   外部缓冲区（float*）
 * @param capacity 缓冲容量（float 元素数）
 * @return int     TensorStatus
 */
int tensor_pool_init(TensorPool* pool, float* buffer, size_t capacity) {
    if (pool == NULL || buffer == NULL || capacity == 0U) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }
    pool->buffer = buffer;
    pool->capacity = capacity;
    pool->used = 0U;
    return TENSOR_STATUS_OK;
}

/**
 * @brief 重置内存池分配游标，不清空数据内容。
 *
 * @param pool 内存池对象
 */
void tensor_pool_reset(TensorPool* pool) {
    if (pool == NULL) {
        return;
    }
    pool->used = 0U;
}

/**
 * @brief 从内存池中分配一个张量并初始化其 shape/stride。
 *
 * @param pool   内存池
 * @param t      输出张量
 * @param ndim   维度数量
 * @param shape  形状数组
 * @return int   TensorStatus
 */
int tensor_pool_alloc(TensorPool* pool, Tensor* t, size_t ndim, const size_t* shape) {
    size_t need = 0U;
    int rc = 0;

    if (pool == NULL || t == NULL) {
        return TENSOR_STATUS_INVALID_ARGUMENT;
    }

    rc = tensor_calc_numel(ndim, shape, &need);
    if (rc != TENSOR_STATUS_OK) {
        return rc;
    }

    if (pool->used + need > pool->capacity) {
        return TENSOR_STATUS_POOL_EXHAUSTED;
    }

    rc = tensor_init_view(t, pool->buffer + pool->used, ndim, shape);
    if (rc != TENSOR_STATUS_OK) {
        return rc;
    }
    pool->used += need;
    return TENSOR_STATUS_OK;
}

/**
 * @brief 判断两个张量 shape 是否一致。
 *
 * @param a 张量 A
 * @param b 张量 B
 * @return int 1 表示一致，0 表示不一致
 */
int tensor_same_shape(const Tensor* a, const Tensor* b) {
    size_t i = 0U;
    if (a == NULL || b == NULL || a->ndim != b->ndim) {
        return 0;
    }
    for (i = 0U; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return 0;
        }
    }
    return 1;
}
