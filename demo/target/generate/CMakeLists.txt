cmake_minimum_required(VERSION 3.20)
project(target_generate C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

get_filename_component(ACTION_C_ROOT "${CMAKE_SOURCE_DIR}/../../.." ABSOLUTE)
set(ACTION_C_BUILD "${ACTION_C_ROOT}/build/src")

add_executable(target_generate ../generate_main.c)

target_include_directories(target_generate PRIVATE
    "${ACTION_C_ROOT}/src/profiler"
    "${ACTION_C_ROOT}/src/nn"
    "${ACTION_C_ROOT}/src/infer"
)

target_link_libraries(target_generate PRIVATE
    "${ACTION_C_BUILD}/profiler/Debug/profiler_core.lib"
    "${ACTION_C_BUILD}/nn/Debug/nn_infer_core.lib"
    "${ACTION_C_BUILD}/nn/Debug/nn_train_core.lib"
)
