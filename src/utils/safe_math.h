#ifndef ACTION_C_SAFE_MATH_H
#define ACTION_C_SAFE_MATH_H

#include <stddef.h>
#include <stdint.h>

/* Safe multiplication: checks whether a * b exceeds limit.
   Returns 0 on overflow, otherwise returns a * b. */
static inline size_t safe_size_mul(size_t a, size_t b, size_t limit) {
    if (a == 0 || b == 0) return 0;
    if (b > limit / a) return 0;
    return a * b;
}

/* Safe size_t multiply with SIZE_MAX as limit. */
static inline size_t safe_size_mul_max(size_t a, size_t b) {
    return safe_size_mul(a, b, SIZE_MAX);
}

/* Safe allocation size: count * sizeof(T) without overflow. */
#define SAFE_ALLOC_SIZE(count, type) \
    safe_size_mul((count), sizeof(type), SIZE_MAX)

#endif
