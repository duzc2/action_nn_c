# 第一次编译：只编译 profiler 和生成器
add_executable(transformer_generate
    generate_main.c
)
target_link_libraries(transformer_generate PRIVATE profiler_core)
target_include_directories(transformer_generate PRIVATE
    ${CMAKE_SOURCE_DIR}/src/profiler
)

# 第二次编译：训练
set(TRANSFORMER_GENERATED_DIR "${CMAKE_BINARY_DIR}/demo/transformer/data")

if(NOT EXISTS ${TRANSFORMER_GENERATED_DIR}/transformer.c)
    file(MAKE_DIRECTORY ${TRANSFORMER_GENERATED_DIR})
    file(WRITE ${TRANSFORMER_GENERATED_DIR}/transformer.c "/* placeholder - run transformer_generate first */\n")
    file(WRITE ${TRANSFORMER_GENERATED_DIR}/transformer.h "/* placeholder - run transformer_generate first */\n")
    message(STATUS "Generated files not found, run transformer_generate first")
endif()

add_executable(transformer_train
    train_main.c
    ${TRANSFORMER_GENERATED_DIR}/transformer.c
)
target_link_libraries(transformer_train PRIVATE nn_infer_core)
target_include_directories(transformer_train PRIVATE
    ${CMAKE_SOURCE_DIR}/src/nn
    ${CMAKE_SOURCE_DIR}/src/nn/types/transformer
    ${TRANSFORMER_GENERATED_DIR}
    ${CMAKE_SOURCE_DIR}/src/infer
)
add_dependencies(transformer_train transformer_generate)

# 第三次编译：推理
set(TRANSFORMER_WEIGHTS_SRC "")

if(EXISTS ${TRANSFORMER_GENERATED_DIR}/weights.c)
    set(TRANSFORMER_WEIGHTS_SRC ${TRANSFORMER_GENERATED_DIR}/weights.c)
endif()

add_executable(transformer_infer
    infer_main.c
    ${TRANSFORMER_GENERATED_DIR}/transformer.c
    ${TRANSFORMER_WEIGHTS_SRC}
)
target_link_libraries(transformer_infer PRIVATE infer_core)
target_include_directories(transformer_infer PRIVATE
    ${CMAKE_SOURCE_DIR}/src/infer
    ${CMAKE_SOURCE_DIR}/src/nn/types/transformer
    ${TRANSFORMER_GENERATED_DIR}
)
add_dependencies(transformer_infer transformer_train)

# 清理目标
add_custom_target(transformer_clean
    COMMAND ${CMAKE_COMMAND} -E remove
        ${CMAKE_BINARY_DIR}/demo/transformer/Debug/transformer_generate.exe
        ${CMAKE_BINARY_DIR}/demo/transformer/Debug/transformer_train.exe
        ${CMAKE_BINARY_DIR}/demo/transformer/Debug/transformer_infer.exe
    COMMENT "Cleaning transformer demo build artifacts"
)
