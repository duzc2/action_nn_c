add_library(profiler_core STATIC
    profiler.c
    profiler_types.c
    prof_error.c
    prof_path.c
    network_def.c
    prof_flatten.c
    prof_hash.c
    prof_validate.c
    prof_codegen.c
)

target_include_directories(profiler_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/nn
    ${CMAKE_SOURCE_DIR}/src/infer
)

target_link_libraries(profiler_core PUBLIC
    nn_infer_core
    nn_train_core
)

add_custom_target(profiler DEPENDS profiler_core)
