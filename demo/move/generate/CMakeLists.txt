cmake_minimum_required(VERSION 3.20)
project(move_generate C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../demo_common.cmake)
action_c_enable_network_types(TYPES mlp)
action_c_add_core_subdirectory()

add_executable(move_generate ../generate_main.c)
action_c_apply_strict_warnings(move_generate)

target_link_libraries(move_generate PRIVATE
    profiler_core
)
