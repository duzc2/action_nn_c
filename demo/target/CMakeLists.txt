# 第一次编译：只编译 profiler 和生成器
add_executable(target_generate
    generate_main.c
)
target_link_libraries(target_generate PRIVATE profiler_core)
target_include_directories(target_generate PRIVATE
    ${CMAKE_SOURCE_DIR}/src/profiler
)

# 第二次编译：训练
set(TARGET_GENERATED_DIR "${CMAKE_BINARY_DIR}/demo/target/data")

if(NOT EXISTS ${TARGET_GENERATED_DIR}/target.c)
    file(MAKE_DIRECTORY ${TARGET_GENERATED_DIR})
    file(WRITE ${TARGET_GENERATED_DIR}/target.c "/* placeholder - run target_generate first */\n")
    file(WRITE ${TARGET_GENERATED_DIR}/target.h "/* placeholder - run target_generate first */\n")
    message(STATUS "Generated files not found, run target_generate first")
endif()

add_executable(target_train
    train_main.c
    ${TARGET_GENERATED_DIR}/target.c
)
target_link_libraries(target_train PRIVATE nn_infer_core)
target_include_directories(target_train PRIVATE
    ${CMAKE_SOURCE_DIR}/src/nn
    ${CMAKE_SOURCE_DIR}/src/nn/types/mlp
    ${TARGET_GENERATED_DIR}
    ${CMAKE_SOURCE_DIR}/src/infer
)
add_dependencies(target_train target_generate)

# 第三次编译：推理
set(TARGET_WEIGHTS_SRC "")

if(EXISTS ${TARGET_GENERATED_DIR}/weights.c)
    set(TARGET_WEIGHTS_SRC ${TARGET_GENERATED_DIR}/weights.c)
endif()

add_executable(target_infer
    infer_main.c
    ${TARGET_GENERATED_DIR}/target.c
    ${TARGET_WEIGHTS_SRC}
)
target_link_libraries(target_infer PRIVATE infer_core)
target_include_directories(target_infer PRIVATE
    ${CMAKE_SOURCE_DIR}/src/infer
    ${CMAKE_SOURCE_DIR}/src/nn/types/transformer
    ${CMAKE_SOURCE_DIR}/src/nn/types/mlp
    ${TARGET_GENERATED_DIR}
)
add_dependencies(target_infer target_train)

# 清理目标
add_custom_target(target_clean
    COMMAND ${CMAKE_COMMAND} -E remove
        ${CMAKE_BINARY_DIR}/demo/target/Debug/target_generate.exe
        ${CMAKE_BINARY_DIR}/demo/target/Debug/target_train.exe
        ${CMAKE_BINARY_DIR}/demo/target/Debug/target_infer.exe
    COMMENT "Cleaning target demo build artifacts"
)
