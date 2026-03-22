add_executable(transformer_generate
    generate_main.c
)
target_link_libraries(transformer_generate PRIVATE profiler_core)
target_include_directories(transformer_generate PRIVATE
    ${CMAKE_SOURCE_DIR}/src/profiler
)

add_executable(transformer_train
    train_main.c
)

add_executable(transformer_infer
    infer_main.c
)
target_link_libraries(transformer_infer PRIVATE infer_core)
target_include_directories(transformer_infer PRIVATE
    ${CMAKE_SOURCE_DIR}/src/infer
    ${CMAKE_SOURCE_DIR}/src/nn/types/transformer
)
