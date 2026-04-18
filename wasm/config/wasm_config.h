/**
 * @file wasm_config.h
 * @brief WebAssembly-specific configuration for action_nn_c
 * 
 * This header provides Wasm-specific configuration options that can be
 * customized by users during deployment. It does not hardcode any values
 * but provides sensible defaults that can be overridden via CMake variables.
 */

#ifndef ACTION_C_WASM_CONFIG_H
#define ACTION_C_WASM_CONFIG_H

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/exports.h>
#endif

/* ============================================================================
 * Feature Toggle Configuration
 * These are controlled via CMake variables at build time.
 * Users should set these in their CMake command, not edit this file directly.
 * ============================================================================ */

#ifndef ACTION_C_WASM_ENABLE_MLP
#define ACTION_C_WASM_ENABLE_MLP 1
#endif

#ifndef ACTION_C_WASM_ENABLE_TRANSFORMER
#define ACTION_C_WASM_ENABLE_TRANSFORMER 1
#endif

#ifndef ACTION_C_WASM_ENABLE_CNN
#define ACTION_C_WASM_ENABLE_CNN 0
#endif

#ifndef ACTION_C_WASM_ENABLE_CNN_DUAL_POOL
#define ACTION_C_WASM_ENABLE_CNN_DUAL_POOL 0
#endif

#ifndef ACTION_C_WASM_ENABLE_RNN
#define ACTION_C_WASM_ENABLE_RNN 0
#endif

#ifndef ACTION_C_WASM_ENABLE_GNN
#define ACTION_C_WASM_ENABLE_GNN 0
#endif

#ifndef ACTION_C_WASM_ENABLE_TRAINING
#define ACTION_C_WASM_ENABLE_TRAINING 0
#endif

/* ============================================================================
 * Memory Configuration
 * Adjust based on your target models and deployment environment.
 * ============================================================================ */

#ifndef ACTION_C_WASM_MEMORY_MB
#define ACTION_C_WASM_MEMORY_MB 64
#endif

#ifndef ACTION_C_WASM_MAX_MEMORY_MB
#define ACTION_C_WASM_MAX_MEMORY_MB 512
#endif

/* Stack size in bytes (default 8MB) */
#ifndef ACTION_C_WASM_STACK_SIZE
#define ACTION_C_WASM_STACK_SIZE (8 * 1024 * 1024)
#endif

/* ============================================================================
 * Optimization Configuration
 * ============================================================================ */

#ifndef ACTION_C_WASM_OPT_LEVEL
#define ACTION_C_WASM_OPT_LEVEL 2
#endif

/* Enable SIMD for modern browsers (requires browser support) */
#ifndef ACTION_C_WASM_ENABLE_SIMD
#define ACTION_C_WASM_ENABLE_SIMD 1
#endif

/* Enable bulk memory operations */
#ifndef ACTION_C_WASM_ENABLE_BULK_MEMORY
#define ACTION_C_WASM_ENABLE_BULK_MEMORY 1
#endif

/* ============================================================================
 * Export Macros
 * Use these to mark functions for export to JavaScript.
 * ============================================================================ */

#ifdef __EMSCRIPTEN__

#define WASM_EXPORT EMSCRIPTEN_KEEPALIVE __attribute__((used))

#define WASM_EXPORT_NAME(name) \
    EMSCRIPTEN_KEEPALIVE __attribute__((used))

#else

#define WASM_EXPORT
#define WASM_EXPORT_NAME(name)

#endif

/* ============================================================================
 * Runtime Configuration
 * ============================================================================ */

/* Enable assertions in Wasm build (disable for production) */
#ifndef ACTION_C_WASM_ENABLE_ASSERTIONS
#ifdef NDEBUG
#define ACTION_C_WASM_ENABLE_ASSERTIONS 0
#else
#define ACTION_C_WASM_ENABLE_ASSERTIONS 1
#endif
#endif

/* Enable debug logging */
#ifndef ACTION_C_WASM_ENABLE_DEBUG_LOG
#define ACTION_C_WASM_ENABLE_DEBUG_LOG 0
#endif

/* ============================================================================
 * Version Information
 * ============================================================================ */

#define ACTION_C_WASM_VERSION_MAJOR 1
#define ACTION_C_WASM_VERSION_MINOR 0
#define ACTION_C_WASM_VERSION_PATCH 0

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the Wasm module version string.
 * @return Version string in format "major.minor.patch"
 */
static inline const char* action_c_wasm_get_version(void) {
    return "1.0.0";
}

/**
 * Check if a specific network type is enabled in this build.
 * @param network_type Network type name (e.g., "mlp", "transformer")
 * @return 1 if enabled, 0 if disabled
 */
static inline int action_c_wasm_is_network_enabled(const char* network_type) {
    if (!network_type) return 0;
    
#if ACTION_C_WASM_ENABLE_MLP
    if (__builtin_strcmp(network_type, "mlp") == 0) return 1;
#endif
#if ACTION_C_WASM_ENABLE_TRANSFORMER
    if (__builtin_strcmp(network_type, "transformer") == 0) return 1;
#endif
#if ACTION_C_WASM_ENABLE_CNN
    if (__builtin_strcmp(network_type, "cnn") == 0) return 1;
#endif
#if ACTION_C_WASM_ENABLE_CNN_DUAL_POOL
    if (__builtin_strcmp(network_type, "cnn_dual_pool") == 0) return 1;
#endif
#if ACTION_C_WASM_ENABLE_RNN
    if (__builtin_strcmp(network_type, "rnn") == 0) return 1;
#endif
#if ACTION_C_WASM_ENABLE_GNN
    if (__builtin_strcmp(network_type, "gnn") == 0) return 1;
#endif
    
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* ACTION_C_WASM_CONFIG_H */
