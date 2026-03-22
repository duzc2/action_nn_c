add_library(infer_core STATIC
    infer_runtime.c
)

target_include_directories(infer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/nn
)

target_link_libraries(infer_core PUBLIC
    nn_infer_core
)
