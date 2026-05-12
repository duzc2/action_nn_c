#include "arena.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>

Arena* arena_create(size_t cap) {
    Arena* a = (Arena*)malloc(sizeof(Arena));
    if (!a) return NULL;
    a->memory = (unsigned char*)malloc(cap);
    if (!a->memory) {
        free(a);
        return NULL;
    }
    a->capacity = cap;
    a->used = 0;
    return a;
}

void arena_destroy(Arena* a) {
    if (!a) return;
    free(a->memory);
    free(a);
}

size_t arena_snapshot(const Arena* a) {
    return a->used;
}

void arena_restore(Arena* a, size_t mark) {
    a->used = mark;
}

static void arena_grow(Arena* a, size_t needed) {
    size_t new_cap = a->capacity;
    while (new_cap < needed) {
        new_cap = (new_cap == 0) ? 4096 : new_cap * 2;
    }
    unsigned char* new_mem = (unsigned char*)realloc(a->memory, new_cap);
    if (!new_mem) {
        LOG_ERROR("Arena grow failed: need %zu, capacity %zu",
                  needed, a->capacity);
        return;
    }
    a->memory   = new_mem;
    a->capacity = new_cap;
}

void* _arena_alloc(Arena* a, size_t size) {
    if (a->used + size > a->capacity) {
        arena_grow(a, a->used + size);
    }
    if (a->used + size > a->capacity) {
        return NULL;
    }
    void* ptr = a->memory + a->used;
    a->used += size;
    return ptr;
}

void* _arena_calloc(Arena* a, size_t size) {
    void* ptr = _arena_alloc(a, size);
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
}
