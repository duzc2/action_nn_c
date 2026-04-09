cmake_minimum_required(VERSION 3.20)
project(mnist_cnn_generate C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../demo_common.cmake)
action_c_enable_network_types(TYPES cnn_dual_pool mlp)
action_c_add_core_subdirectory()

add_executable(mnist_cnn_generate
    ../generate_main.c
    ../mnist_cnn_dataset.c
)
action_c_apply_strict_warnings(mnist_cnn_generate)

target_link_libraries(mnist_cnn_generate PRIVATE
    profiler_core
)
