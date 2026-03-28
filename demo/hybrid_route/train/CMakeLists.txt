cmake_minimum_required(VERSION 3.20)
project(hybrid_route_train C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../demo_common.cmake)
action_c_enable_network_types(TYPES mlp transformer)
action_c_add_core_subdirectory()

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

add_executable(hybrid_route_train
    ../train_main.c
    "${GENERATED_INFER_C}"
    "${GENERATED_TRAIN_C}"
    "${GENERATED_WEIGHTS_SAVE_C}"
)
action_c_apply_strict_warnings(hybrid_route_train)

target_include_directories(hybrid_route_train PRIVATE
    "${ACTION_C_ROOT}/src"
    "${GENERATED_DIR}"
)

target_link_libraries(hybrid_route_train PRIVATE
    nn_train_core
)
