cmake_minimum_required(VERSION 3.16)
project(action_c C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if (CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

if (MSVC)
    add_compile_options(/utf-8 /W4 /WX)
endif()

option(ENABLE_NN_TRANSFORMER "Enable transformer network type" ON)
option(ENABLE_NN_SEVENSEG "Enable sevenseg network type" ON)

if (ENABLE_NN_TRANSFORMER)
    set(ENABLE_NN_TRANSFORMER_INT 1)
else()
    set(ENABLE_NN_TRANSFORMER_INT 0)
endif()

if (ENABLE_NN_SEVENSEG)
    set(ENABLE_NN_SEVENSEG_INT 1)
else()
    set(ENABLE_NN_SEVENSEG_INT 0)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include/nn_build_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/nn_build_config.h
    @ONLY
)

add_library(nn_registry STATIC
    src/core/nn_registry.c
    src/core/nn_registry_enabled.c
)

if (ENABLE_NN_TRANSFORMER)
    target_sources(nn_registry PRIVATE src/nn/transformer/transformer_serde.c)
endif()

if (ENABLE_NN_SEVENSEG)
    target_sources(nn_registry PRIVATE src/nn/sevenseg/sevenseg_serde.c)
endif()

target_include_directories(nn_registry PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

add_executable(profiler src/tools/profiler.c)
target_include_directories(profiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(profiler PRIVATE nn_registry)
