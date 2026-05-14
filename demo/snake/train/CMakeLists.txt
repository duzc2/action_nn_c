cmake_minimum_required(VERSION 3.20)
set(CMAKE_C_EXTENSIONS OFF)
project(snake_train C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../demo_common.cmake)
action_c_enable_network_types(TYPES mlp)
action_c_add_core_subdirectory()

get_filename_component(SNAKE_DEMO_DIR "${CMAKE_BINARY_DIR}/.." ABSOLUTE)
set(ACTION_C_GENERATED_DIR "${SNAKE_DEMO_DIR}/data")

action_c_get_demo_generated_dir(GENERATED_DIR)
set(GENERATED_INFER_C "${GENERATED_DIR}/infer.c")
set(GENERATED_TRAIN_C "${GENERATED_DIR}/train.c")
set(GENERATED_WEIGHTS_SAVE_C "${GENERATED_DIR}/weights_save.c")
set_source_files_properties(
    "${GENERATED_INFER_C}"
    "${GENERATED_TRAIN_C}"
    "${GENERATED_WEIGHTS_SAVE_C}"
    PROPERTIES GENERATED TRUE
)

add_executable(snake_train
    ../train_main.c
    "${GENERATED_INFER_C}"
    "${GENERATED_TRAIN_C}"
    "${GENERATED_WEIGHTS_SAVE_C}"
)
action_c_apply_strict_warnings(snake_train)

target_include_directories(snake_train PRIVATE
    "${ACTION_C_ROOT}/src"
    "${GENERATED_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/.."
)

target_link_libraries(snake_train PRIVATE
    nn_train_core
)
