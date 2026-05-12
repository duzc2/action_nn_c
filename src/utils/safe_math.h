#ifndef ACTION_C_SAFE_MATH_H
#define ACTION_C_SAFE_MATH_H

#include <stddef.h>
#include <stdint.h>

/* 安全乘法：检查 a * b 是否超过 limit。
   若溢出返回 0，否则返回 a * b。 */
static inline size_t safe_size_mul(size_t a, size_t b, size_t limit) {
    if (a == 0 || b == 0) return 0;
    if (b > limit / a) return 0;
    return a * b;
}

/* 安全 size_t 乘法，上限为 SIZE_MAX */
static inline size_t safe_size_mul_max(size_t a, size_t b) {
    return safe_size_mul(a, b, SIZE_MAX);
}

/* 安全 size_t 加法的元素数版本: count * sizeof(T) 不溢出 */
#define SAFE_ALLOC_SIZE(count, type) \
    safe_size_mul((count), sizeof(type), SIZE_MAX)

#endif
