/**
 * @file wasm_exports.c
 * @brief WebAssembly export wrappers for action_nn_c
 * 
 * This file provides the C functions that are exported to JavaScript
 * via Emscripten. It wraps the core action_nn_c functionality with
 * Wasm-compatible interfaces.
 */

#include "wasm_config.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/exports.h>
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* Core action_nn_c headers */
#include "infer_runtime.h"
#include "profiler.h"

#ifdef ACTION_C_WASM_ENABLE_TRAINING
#include "train_runtime.h"
#endif

/* ============================================================================
 * Version and Info Exports
 * ============================================================================ */

WASM_EXPORT
const char* action_c_wasm_get_version_string(void) {
    return "1.0.0";
}

WASM_EXPORT
int action_c_wasm_get_version_major(void) {
    return ACTION_C_WASM_VERSION_MAJOR;
}

WASM_EXPORT
int action_c_wasm_get_version_minor(void) {
    return ACTION_C_WASM_VERSION_MINOR;
}

WASM_EXPORT
int action_c_wasm_get_version_patch(void) {
    return ACTION_C_WASM_VERSION_PATCH;
}

/* ============================================================================
 * Memory Management Exports
 * These allow JavaScript to allocate/deallocate memory for data transfer.
 * ============================================================================ */

WASM_EXPORT
void* action_c_wasm_malloc(size_t size) {
    return malloc(size);
}

WASM_EXPORT
void action_c_wasm_free(void* ptr) {
    free(ptr);
}

WASM_EXPORT
size_t action_c_wasm_get_memory_size(void) {
#ifdef __EMSCRIPTEN__
    extern size_t __builtin_wasm_memory_size(int);
    return __builtin_wasm_memory_size(0) * 65536;
#else
    return 0;
#endif
}

/* ============================================================================
 * Inference API Exports
 * Note: The actual inference API uses nn_infer_runtime_step() which takes
 * an NNInferRequest struct. Users should construct this in JS/Wasm memory.
 * ============================================================================ */

/**
 * Run one inference step.
 * 
 * @param request_ptr Pointer to NNInferRequest struct in Wasm memory
 * @return 0 on success, negative error code on failure
 */
WASM_EXPORT
int action_c_wasm_infer_step(const void* request_ptr) {
    if (!request_ptr) {
        return -1;
    }
    
    const NNInferRequest* request = (const NNInferRequest*)request_ptr;
    return nn_infer_runtime_step(request);
}

/**
 * Destroy an inference context.
 * This is a convenience wrapper that frees the context pointer.
 * 
 * @param ctx_ptr Opaque context pointer
 */
WASM_EXPORT
void action_c_wasm_infer_destroy(void* ctx_ptr) {
    if (!ctx_ptr) {
        return;
    }
    
    /* Context is owned by the specific network implementation */
    /* This just frees the memory - implementations should provide their own destroy */
    free(ctx_ptr);
}

/* ============================================================================
 * Training API Exports (if enabled)
 * ============================================================================ */

#ifdef ACTION_C_WASM_ENABLE_TRAINING

/**
 * Run one training step.
 * 
 * @param request_ptr Pointer to NNTrainRequest struct in Wasm memory
 * @return 0 on success, negative error code on failure
 */
WASM_EXPORT
int action_c_wasm_train_step_request(const void* request_ptr) {
    if (!request_ptr) {
        return -1;
    }
    
    const NNTrainRequest* request = (const NNTrainRequest*)request_ptr;
    return nn_train_runtime_step(request);
}

/**
 * Destroy a training context.
 * 
 * @param ctx_ptr Opaque context pointer
 */
WASM_EXPORT
void action_c_wasm_train_destroy(void* ctx_ptr) {
    if (!ctx_ptr) {
        return;
    }
    
    free(ctx_ptr);
}

#endif /* ACTION_C_WASM_ENABLE_TRAINING */

/* ============================================================================
 * Network Type Query Exports
 * ============================================================================ */

/**
 * Get the number of supported network types in this build.
 * 
 * @return Number of supported network types
 */
WASM_EXPORT
int action_c_wasm_get_network_type_count(void) {
    int count = 0;
    
#if ACTION_C_WASM_ENABLE_MLP
    count++;
#endif
#if ACTION_C_WASM_ENABLE_TRANSFORMER
    count++;
#endif
#if ACTION_C_WASM_ENABLE_CNN
    count++;
#endif
#if ACTION_C_WASM_ENABLE_CNN_DUAL_POOL
    count++;
#endif
#if ACTION_C_WASM_ENABLE_RNN
    count++;
#endif
#if ACTION_C_WASM_ENABLE_GNN
    count++;
#endif
    
    return count;
}

/**
 * Get the name of a supported network type by index.
 * 
 * @param index Index of network type (0 to count-1)
 * @return Network type name string, or NULL if index out of range
 */
WASM_EXPORT
const char* action_c_wasm_get_network_type_name(int index) {
    int current = 0;
    
#if ACTION_C_WASM_ENABLE_MLP
    if (index == current++) return "mlp";
#endif
#if ACTION_C_WASM_ENABLE_TRANSFORMER
    if (index == current++) return "transformer";
#endif
#if ACTION_C_WASM_ENABLE_CNN
    if (index == current++) return "cnn";
#endif
#if ACTION_C_WASM_ENABLE_CNN_DUAL_POOL
    if (index == current++) return "cnn_dual_pool";
#endif
#if ACTION_C_WASM_ENABLE_RNN
    if (index == current++) return "rnn";
#endif
#if ACTION_C_WASM_ENABLE_GNN
    if (index == current++) return "gnn";
#endif
    
    return NULL;
}

/**
 * Check if a specific network type is enabled.
 * 
 * @param name Network type name (e.g., "mlp", "transformer")
 * @return 1 if enabled, 0 if disabled
 */
WASM_EXPORT
int action_c_wasm_is_network_type_enabled(const char* name) {
    return action_c_wasm_is_network_enabled(name);
}

/* ============================================================================
 * Profiler Exports (for code generation workflow)
 * ============================================================================ */

WASM_EXPORT
void* action_c_wasm_profiler_create(void) {
    /* TODO: Implement profiler creation */
    return NULL;
}

WASM_EXPORT
void action_c_wasm_profiler_destroy(void* profiler_ptr) {
    /* TODO: Implement profiler destruction */
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Get the size of the Wasm heap in bytes.
 * Useful for JavaScript to understand memory constraints.
 */
WASM_EXPORT
uint32_t action_c_wasm_get_heap_size(void) {
#ifdef __EMSCRIPTEN__
    extern size_t _get_heap_size(void);
    return (uint32_t)_get_heap_size();
#else
    return 0;
#endif
}

/**
 * Print debug information to console (Emscripten's console.log).
 */
#ifdef ACTION_C_WASM_ENABLE_DEBUG_LOG
#include <stdio.h>

WASM_EXPORT
void action_c_wasm_debug_log(const char* message) {
    if (message) {
        printf("[ACTION_C_WASM] %s\n", message);
    }
}
#endif
