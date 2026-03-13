cmake_minimum_required(VERSION 3.16)
project(action_c C)

# 设计约束：
# - 强制 C99，确保与需求一致。
# - 不引入外部依赖，保持可移植和最小构建面。
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# clang/gcc 统一告警配置，帮助在早期发现潜在问题。
if (CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# MSVC 指定 /utf-8，确保中文注释以 UTF-8 编码解析，避免 C4819 警告。
if (MSVC)
    add_compile_options(/utf-8)
endif()

add_library(dnn_core STATIC
    src/core/tensor.c
    src/core/ops.c
    src/core/tokenizer.c
    src/core/weights_io.c
    src/core/protocol.c
    src/core/platform_driver.c
    src/model/transformer.c
    src/train/csv_loader.c
)
target_include_directories(dnn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

add_executable(profiler
    src/tools/profiler.c
)
target_include_directories(profiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

add_executable(min_train_loop
    src/tools/min_train_loop.c
)
target_include_directories(min_train_loop PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)
target_link_libraries(min_train_loop PRIVATE dnn_core)

if (UNIX OR MINGW)
    target_link_libraries(dnn_core PUBLIC m)
    target_link_libraries(profiler PRIVATE m)
endif()
