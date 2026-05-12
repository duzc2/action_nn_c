#ifndef ACTION_C_ARENA_H
#define ACTION_C_ARENA_H

#include <stddef.h>

typedef struct {
    unsigned char* memory;
    size_t         capacity;
    size_t         used;
} Arena;

/* 创建 arena。cap 为初始容量（字节） */
Arena* arena_create(size_t cap);

/* 销毁 arena */
void arena_destroy(Arena* a);

/* 记录当前水位线，返回标记 */
size_t arena_snapshot(const Arena* a);

/* 回卷到之前的水位线 */
void arena_restore(Arena* a, size_t mark);

/* 从 arena 分配 n 个类型 T 的元素，返回未初始化内存 */
#define ARENA_ALLOC(a, T, n) \
    ((T*)_arena_alloc((a), (n) * sizeof(T)))

/* 从 arena 分配 n 个类型 T 的元素，并清零 */
#define ARENA_CALLOC(a, T, n) \
    ((T*)_arena_calloc((a), (n) * sizeof(T)))

/* 内部函数，不直接调用 */
void* _arena_alloc(Arena* a, size_t size);
void* _arena_calloc(Arena* a, size_t size);

#endif
