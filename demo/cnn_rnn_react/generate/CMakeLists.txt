cmake_minimum_required(VERSION 3.20)
project(cnn_rnn_react_generate C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../demo_common.cmake)
action_c_enable_network_types(TYPES cnn rnn)
action_c_add_core_subdirectory()

add_executable(cnn_rnn_react_generate ../generate_main.c)

set_target_properties(cnn_rnn_react_generate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Debug"
)

target_link_libraries(cnn_rnn_react_generate PRIVATE
    profiler_core
)
