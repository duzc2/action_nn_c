cmake_minimum_required(VERSION 3.20)
project(sevenseg_train C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(ACTION_C_ROOT "${CMAKE_SOURCE_DIR}/../../..")
set(ACTION_C_BUILD "${ACTION_C_ROOT}/build/src")
set(GENERATED_DIR "${ACTION_C_ROOT}/demo/sevenseg/data")

add_executable(sevenseg_train
    ../train_main.c
    "${GENERATED_DIR}/sevenseg.c"
)

target_include_directories(sevenseg_train PRIVATE
    "${ACTION_C_ROOT}/src/nn"
    "${ACTION_C_ROOT}/src/nn/types/mlp"
    "${ACTION_C_ROOT}/src/infer"
    "${GENERATED_DIR}"
)

target_link_libraries(sevenseg_train PRIVATE
    "${ACTION_C_BUILD}/nn/Debug/nn_infer_core.lib"
)
