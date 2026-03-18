cmake_minimum_required(VERSION 3.16)
project(action_c C)

# 设计约束：
# - 强制 C99，确保与需求一致。
# - 不引入外部依赖，保持可移植和最小构建面。
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
enable_testing()

# clang/gcc 统一告警配置，帮助在早期发现潜在问题。
if (CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# MSVC 指定 /utf-8，确保中文注释以 UTF-8 编码解析，避免 C4819 警告。
if (MSVC)
    add_compile_options(/utf-8)
endif()

add_library(dnn_infer STATIC
    src/core/network_spec.c
    src/core/tensor.c
    src/core/ops.c
    src/tokenizer/tokenizer.c
    src/core/weights_io.c
    src/core/protocol.c
    src/infer/workflow_runtime.c
    src/tokenizer/tokenizer_runtime.c
    src/model/transformer.c
)
target_include_directories(dnn_infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/infer/include
)

add_library(dnn_train STATIC
    src/train/workflow_train.c
    src/train/csv_loader.c
)
target_include_directories(dnn_train PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/train/include
)
target_link_libraries(dnn_train PUBLIC dnn_infer)

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
target_link_libraries(min_train_loop PRIVATE dnn_train)

add_subdirectory(demo/goal)
add_subdirectory(demo/step)
add_subdirectory(demo/sevenseg)
add_subdirectory(demo/transformer_simple)

if (UNIX OR MINGW)
    target_link_libraries(dnn_infer PUBLIC m)
    target_link_libraries(profiler PRIVATE m)
    target_link_libraries(dnn_train PUBLIC m)
endif()

# 全量 C99 测试套件：
# - 覆盖单元/集成/正确性/错误/临界/压力/常规测试。
# - 通过自定义测试执行器输出进度条与实时日志。
add_executable(full_test_suite
    test/src/test_framework.c
    test/src/test_cases_unit_correctness.c
    test/src/test_cases_error_boundary.c
    test/src/test_cases_stress_integration.c
    test/src/test_cases_model_special.c
    test/src/test_suite.c
)
target_include_directories(full_test_suite PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test/include
)
target_link_libraries(full_test_suite PRIVATE dnn_train)

if (UNIX OR MINGW)
    target_link_libraries(full_test_suite PRIVATE m)
endif()

add_test(NAME full_test_suite COMMAND full_test_suite)
