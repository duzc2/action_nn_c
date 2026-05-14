cmake_minimum_required(VERSION 3.20)
set(CMAKE_C_EXTENSIONS OFF)
project(snake_generate C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../demo_common.cmake)
action_c_enable_network_types(TYPES mlp)
action_c_add_core_subdirectory()

add_executable(snake_generate ../generate_main.c)
action_c_apply_strict_warnings(snake_generate)

target_link_libraries(snake_generate PRIVATE
    profiler_core
)
