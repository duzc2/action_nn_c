cmake_minimum_required(VERSION 3.20)
project(road_graph_nav_infer C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../../demo_common.cmake)
action_c_enable_network_types(TYPES gnn mlp)
action_c_add_core_subdirectory()

action_c_get_demo_generated_dir(GENERATED_DIR)
set(GENERATED_INFER_C "${GENERATED_DIR}/infer.c")
set(GENERATED_WEIGHTS_LOAD_C "${GENERATED_DIR}/weights_load.c")
set_source_files_properties(
    "${GENERATED_INFER_C}"
    "${GENERATED_WEIGHTS_LOAD_C}"
    PROPERTIES GENERATED TRUE
)

add_executable(road_graph_nav_infer
    ../infer_main.c
    "${GENERATED_INFER_C}"
    "${GENERATED_WEIGHTS_LOAD_C}"
)
action_c_apply_strict_warnings(road_graph_nav_infer)

target_include_directories(road_graph_nav_infer PRIVATE
    "${ACTION_C_ROOT}/src"
    "${GENERATED_DIR}"
)

target_link_libraries(road_graph_nav_infer PRIVATE
    infer_core
)
