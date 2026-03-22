add_library(train_core STATIC
    train_runtime.c
)

target_include_directories(train_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/nn
)

target_link_libraries(train_core PUBLIC
    nn_train_core
    infer_core
)
