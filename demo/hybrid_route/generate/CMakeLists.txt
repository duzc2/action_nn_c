cmake_minimum_required(VERSION 3.20)
project(hybrid_route_generate C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../demo_common.cmake)
action_c_enable_network_types(TYPES mlp transformer)
action_c_add_core_subdirectory()

add_executable(hybrid_route_generate ../generate_main.c)

target_link_libraries(hybrid_route_generate PRIVATE
    profiler_core
)
