cmake_minimum_required(VERSION 3.20)
set(CMAKE_C_EXTENSIONS OFF)
project(snake_infer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../demo_common.cmake)
action_c_enable_network_types(TYPES mlp)
action_c_add_core_subdirectory()

get_filename_component(SNAKE_DEMO_DIR "${CMAKE_BINARY_DIR}/.." ABSOLUTE)
set(ACTION_C_GENERATED_DIR "${SNAKE_DEMO_DIR}/data")

action_c_get_demo_generated_dir(GENERATED_DIR)
set(GENERATED_INFER_C "${GENERATED_DIR}/infer.c")
set(GENERATED_WEIGHTS_LOAD_C "${GENERATED_DIR}/weights_load.c")
set_source_files_properties(
    "${GENERATED_INFER_C}"
    "${GENERATED_WEIGHTS_LOAD_C}"
    PROPERTIES GENERATED TRUE
)

add_executable(snake_infer
    ../infer_main.c
    "${GENERATED_INFER_C}"
    "${GENERATED_WEIGHTS_LOAD_C}"
)
action_c_apply_strict_warnings(snake_infer)
if(MSVC OR (CMAKE_C_COMPILER_ID MATCHES "Clang" AND WIN32))
    target_compile_definitions(snake_infer PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

target_include_directories(snake_infer PRIVATE
    "${ACTION_C_ROOT}/src"
    "${GENERATED_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/.."
)

target_link_libraries(snake_infer PRIVATE
    infer_core
)
