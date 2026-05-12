/**
 * @file nn_backend.h
 * @brief VTable interfaces for inference and training backends.
 *
 * Every NN type (MLP, CNN, RNN, GNN, Transformer) implements these
 * function-pointer tables so the registry and runtime can dispatch
 * polymorphically without switch-case or bridge layers.
 */

#ifndef ACTION_C_NN_BACKEND_H
#define ACTION_C_NN_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* If Track A is incomplete, forward-declare Arena. */
#ifndef ACTION_C_ARENA_H
typedef struct Arena Arena;
#endif

/* ─── Inference backend interface ─── */
typedef struct {
    const char* type_name;

    void*    (*create)         (const void* config_blob, size_t config_size,
                                Arena* arena);
    void     (*destroy)        (void* context);
    int      (*step)           (void* context);
    int      (*get_output)     (const void* context, float* out, size_t out_size);
    int      (*save_weights)   (const void* context, FILE* fp);
    int      (*load_weights)   (void* context, FILE* fp);

    uint64_t (*get_network_hash)(const void* context);
    uint64_t (*get_layout_hash) (const void* context);
    uint32_t (*get_abi_version) (void);
} NNInferBackend;

/* ─── Training backend interface ─── */
typedef struct {
    const char* type_name;

    void* (*create)          (const void* config_blob, size_t config_size,
                              const void* infer_config_blob, size_t infer_config_size,
                              Arena* arena);
    void  (*destroy)         (void* context);
    int   (*step)            (void* context, const float* input, const float* target);
    int   (*step_with_data)  (void* context, const float* input,
                              const float* target, float* out_grad);
    int   (*save_checkpoint) (void* context, FILE* fp);
    int   (*load_checkpoint) (void* context, FILE* fp);
} NNTrainBackend;

#endif /* ACTION_C_NN_BACKEND_H */
