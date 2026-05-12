#ifndef ACTION_C_ARENA_H
#define ACTION_C_ARENA_H

#include <stddef.h>

typedef struct Arena {
    unsigned char* memory;
    size_t         capacity;
    size_t         used;
} Arena;

/* Create an arena with initial capacity in bytes. */
Arena* arena_create(size_t cap);

/* Destroy an arena and free its memory. */
void arena_destroy(Arena* a);

/* Record current watermark, return a mark for later restore. */
size_t arena_snapshot(const Arena* a);

/* Restore arena used pointer to a previous mark. */
void arena_restore(Arena* a, size_t mark);

/* Allocate n elements of type T from arena, returns uninitialized memory. */
#define ARENA_ALLOC(a, T, n) \
    ((T*)_arena_alloc((a), (n) * sizeof(T)))

/* Allocate n elements of type T from arena and zero-initialize. */
#define ARENA_CALLOC(a, T, n) \
    ((T*)_arena_calloc((a), (n) * sizeof(T)))

/* Internal: do not call directly. */
void* _arena_alloc(Arena* a, size_t size);
void* _arena_calloc(Arena* a, size_t size);

#endif
