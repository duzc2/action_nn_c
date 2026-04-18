cmake_minimum_required(VERSION 3.20)
project(action_c_wasm C)

# ============================================================================
# WebAssembly Build Configuration for action_nn_c
# ============================================================================

# Require Emscripten toolchain
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    message(WARNING 
        "This CMakeLists.txt is designed for Emscripten.\n"
        "Please use the Emscripten toolchain file:\n"
        "  cmake -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake ...\n"
        "Or source the Emscripten environment and re-run cmake.")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ============================================================================
# Build Options (Configurable via CMake variables)
# ============================================================================

option(ACTION_C_WASM_ENABLE_MLP "Enable MLP network type" ON)
option(ACTION_C_WASM_ENABLE_TRANSFORMER "Enable Transformer network type" ON)
option(ACTION_C_WASM_ENABLE_CNN "Enable CNN network type" OFF)
option(ACTION_C_WASM_ENABLE_CNN_DUAL_POOL "Enable dual-pool CNN network type" OFF)
option(ACTION_C_WASM_ENABLE_RNN "Enable RNN network type" OFF)
option(ACTION_C_WASM_ENABLE_GNN "Enable GNN network type" OFF)
option(ACTION_C_WASM_ENABLE_TRAINING "Include training support" OFF)

set(ACTION_C_WASM_MEMORY_MB "64" CACHE STRING "Initial Wasm memory size in MB")
set(ACTION_C_WASM_MAX_MEMORY_MB "512" CACHE STRING "Maximum Wasm memory size in MB")
set(ACTION_C_WASM_OPT_LEVEL "2" CACHE STRING "Optimization level (0-3)")
set(ACTION_C_WASM_STACK_SIZE "8388608" CACHE STRING "Stack size in bytes (default 8MB)")

option(ACTION_C_WASM_ENABLE_SIMD "Enable SIMD support" ON)
option(ACTION_C_WASM_ENABLE_BULK_MEMORY "Enable bulk memory operations" ON)
option(ACTION_C_WASM_ENABLE_DEBUG_LOG "Enable debug logging" OFF)

# Validate options
if(NOT ACTION_C_WASM_ENABLE_MLP AND
   NOT ACTION_C_WASM_ENABLE_TRANSFORMER AND
   NOT ACTION_C_WASM_ENABLE_CNN AND
   NOT ACTION_C_WASM_ENABLE_CNN_DUAL_POOL AND
   NOT ACTION_C_WASM_ENABLE_RNN AND
   NOT ACTION_C_WASM_ENABLE_GNN)
    message(FATAL_ERROR "At least one network type must be enabled for Wasm build")
endif()

# ============================================================================
# Emscripten-specific Flags
# ============================================================================

# Set optimization level
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O${ACTION_C_WASM_OPT_LEVEL}")

# Memory configuration
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --initial-memory=${ACTION_C_WASM_MEMORY_MB}*65536")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --max-memory=${ACTION_C_WASM_MAX_MEMORY_MB}*65536")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --stack-size=${ACTION_C_WASM_STACK_SIZE}")

# Enable modern Wasm features
if(ACTION_C_WASM_ENABLE_SIMD)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msimd128")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -msimd128")
endif()

if(ACTION_C_WASM_ENABLE_BULK_MEMORY)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mbulk-memory")
endif()

# Exported symbols
set(WASM_EXPORTED_SYMBOLS
    "action_c_wasm_get_version_string"
    "action_c_wasm_get_version_major"
    "action_c_wasm_get_version_minor"
    "action_c_wasm_get_version_patch"
    "action_c_wasm_malloc"
    "action_c_wasm_free"
    "action_c_wasm_get_memory_size"
    "action_c_wasm_infer_create"
    "action_c_wasm_infer_run"
    "action_c_wasm_infer_destroy"
    "action_c_wasm_get_network_type_count"
    "action_c_wasm_get_network_type_name"
    "action_c_wasm_is_network_type_enabled"
)

if(ACTION_C_WASM_ENABLE_TRAINING)
    list(APPEND WASM_EXPORTED_SYMBOLS
        "action_c_wasm_train_create"
        "action_c_wasm_train_step"
        "action_c_wasm_train_get_loss"
        "action_c_wasm_train_destroy"
    )
endif()

# Create export file
set(EXPORT_FILE "${CMAKE_CURRENT_BINARY_DIR}/wasm_exports.txt")
file(WRITE ${EXPORT_FILE} "")
foreach(SYMBOL ${WASM_EXPORTED_SYMBOLS})
    file(APPEND EXPORT_FILE "_${SYMBOL}\n")
endforeach()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --export-file=${EXPORT_FILE}")

# Output format: both .js and .wasm
set(CMAKE_EXECUTABLE_SUFFIX ".js")

# Environment settings
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ENVIRONMENT=web")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s MODULARIZE=1")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s EXPORT_ES6=0")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s EXPORT_NAME='ActionNnC'")

# File system support (if needed for loading models)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s FORCE_FILESYSTEM=0")

# Allow memory growth
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ALLOW_MEMORY_GROWTH=1")

# ============================================================================
# Include Directories
# ============================================================================

set(ACTION_C_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(ACTION_C_SRC_DIR "${ACTION_C_ROOT_DIR}/src")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    ${ACTION_C_SRC_DIR}
    ${ACTION_C_SRC_DIR}/nn
    ${ACTION_C_SRC_DIR}/infer
    ${ACTION_C_SRC_DIR}/train
    ${ACTION_C_SRC_DIR}/profiler
)

# ============================================================================
# Build Core Libraries
# ============================================================================

# Re-use the existing CMake structure from src/
# We need to pass our options to the subdirectory

set(ACTION_C_ENABLE_NN_MLP ${ACTION_C_WASM_ENABLE_MLP})
set(ACTION_C_ENABLE_NN_TRANSFORMER ${ACTION_C_WASM_ENABLE_TRANSFORMER})
set(ACTION_C_ENABLE_NN_CNN ${ACTION_C_WASM_ENABLE_CNN})
set(ACTION_C_ENABLE_NN_CNN_DUAL_POOL ${ACTION_C_WASM_ENABLE_CNN_DUAL_POOL})
set(ACTION_C_ENABLE_NN_RNN ${ACTION_C_WASM_ENABLE_RNN})
set(ACTION_C_ENABLE_NN_GNN ${ACTION_C_WASM_ENABLE_GNN})

add_subdirectory(${ACTION_C_SRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/src_build)

# ============================================================================
# Build Wasm Export Layer
# ============================================================================

add_library(wasm_exports OBJECT
    src/wasm_exports.c
)

target_compile_definitions(wasm_exports PRIVATE
    ACTION_C_WASM_VERSION="1.0.0"
    ACTION_C_WASM_VERSION_MAJOR=1
    ACTION_C_WASM_VERSION_MINOR=0
    ACTION_C_WASM_VERSION_PATCH=0
)

if(ACTION_C_WASM_ENABLE_TRAINING)
    target_compile_definitions(wasm_exports PRIVATE
        ACTION_C_WASM_ENABLE_TRAINING=1
    )
endif()

if(ACTION_C_WASM_ENABLE_DEBUG_LOG)
    target_compile_definitions(wasm_exports PRIVATE
        ACTION_C_WASM_ENABLE_DEBUG_LOG=1
    )
endif()

target_link_libraries(wasm_exports PRIVATE
    infer_core
    profiler_core
)

if(ACTION_C_WASM_ENABLE_TRAINING)
    target_link_libraries(wasm_exports PRIVATE
        train_core
    )
endif()

# ============================================================================
# Build Final Wasm Module
# ============================================================================

add_executable(action_nn_c
    $<TARGET_OBJECTS:wasm_exports>
)

target_link_libraries(action_nn_c PUBLIC
    infer_core
    nn_infer_core
    profiler_core
)

if(ACTION_C_WASM_ENABLE_TRAINING)
    target_link_libraries(action_nn_c PUBLIC
        train_core
        nn_train_core
    )
endif()

# Apply strict warnings (adapted for Emscripten)
if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(action_nn_c PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Release>:-Werror>
    )
endif()

# ============================================================================
# Installation
# ============================================================================

set(WASM_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dist")

# Custom command to copy outputs to dist/
add_custom_command(TARGET action_nn_c POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${WASM_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/action_nn_c.js
        ${WASM_OUTPUT_DIR}/action_nn_c.js
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/action_nn_c.wasm
        ${WASM_OUTPUT_DIR}/action_nn_c.wasm
    COMMENT "Copying Wasm artifacts to dist/"
)

# ============================================================================
# Print Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== action_nn_c WebAssembly Build Configuration ===")
message(STATUS "MLP Network:              ${ACTION_C_WASM_ENABLE_MLP}")
message(STATUS "Transformer Network:      ${ACTION_C_WASM_ENABLE_TRANSFORMER}")
message(STATUS "CNN Network:              ${ACTION_C_WASM_ENABLE_CNN}")
message(STATUS "Dual-Pool CNN:            ${ACTION_C_WASM_ENABLE_CNN_DUAL_POOL}")
message(STATUS "RNN Network:              ${ACTION_C_WASM_ENABLE_RNN}")
message(STATUS "GNN Network:              ${ACTION_C_WASM_ENABLE_GNN}")
message(STATUS "Training Support:         ${ACTION_C_WASM_ENABLE_TRAINING}")
message(STATUS "Initial Memory:           ${ACTION_C_WASM_MEMORY_MB} MB")
message(STATUS "Max Memory:               ${ACTION_C_WASM_MAX_MEMORY_MB} MB")
message(STATUS "Stack Size:               ${ACTION_C_WASM_STACK_SIZE} bytes")
message(STATUS "Optimization Level:       ${ACTION_C_WASM_OPT_LEVEL}")
message(STATUS "SIMD Support:             ${ACTION_C_WASM_ENABLE_SIMD}")
message(STATUS "Bulk Memory:              ${ACTION_C_WASM_ENABLE_BULK_MEMORY}")
message(STATUS "Debug Logging:            ${ACTION_C_WASM_ENABLE_DEBUG_LOG}")
message(STATUS "Output Directory:         ${WASM_OUTPUT_DIR}")
message(STATUS "==========================================================")
message(STATUS "")
