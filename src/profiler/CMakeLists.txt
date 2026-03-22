add_library(profiler_core STATIC
    profiler.c
)

target_include_directories(profiler_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/nn
    ${CMAKE_SOURCE_DIR}/src/infer
)

target_link_libraries(profiler_core PUBLIC
    nn_infer_core
)
