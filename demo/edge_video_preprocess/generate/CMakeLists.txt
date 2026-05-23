cmake_minimum_required(VERSION 3.20)
set(CMAKE_C_EXTENSIONS OFF)
project(edge_video_preprocess_generate C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../demo_common.cmake)
action_c_enable_network_types(TYPES cnn cnn_dual_pool mlp)
action_c_add_core_subdirectory()

add_executable(edge_video_preprocess_generate
    ../generate_main.c
)
action_c_apply_strict_warnings(edge_video_preprocess_generate)

target_include_directories(edge_video_preprocess_generate PRIVATE
    "${ACTION_C_ROOT}/src"
)

target_link_libraries(edge_video_preprocess_generate PRIVATE
    profiler_core
)
