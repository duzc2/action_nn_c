cmake_minimum_required(VERSION 3.20)
project(transformer_train C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(ACTION_C_ROOT "${CMAKE_SOURCE_DIR}/../../..")
set(ACTION_C_BUILD "${ACTION_C_ROOT}/build/src")
set(GENERATED_DIR "${ACTION_C_ROOT}/demo/transformer/data")

add_executable(transformer_train
    ../train_main.c
    "${GENERATED_DIR}/transformer.c"
)

target_include_directories(transformer_train PRIVATE
    "${ACTION_C_ROOT}/src/nn"
    "${ACTION_C_ROOT}/src/nn/types/transformer"
    "${ACTION_C_ROOT}/src/infer"
    "${GENERATED_DIR}"
)

target_link_libraries(transformer_train PRIVATE
    "${ACTION_C_BUILD}/nn/Debug/nn_infer_core.lib"
)
