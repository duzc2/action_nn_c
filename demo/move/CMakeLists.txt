# 第一次编译：只编译 profiler 和生成器
add_executable(move_generate
    generate_main.c
)
target_link_libraries(move_generate PRIVATE profiler_core)
target_include_directories(move_generate PRIVATE
    ${CMAKE_SOURCE_DIR}/src/profiler
)

# 第二次编译：训练
# 生成的 move.c 放在 build 目录
set(MOVE_GENERATED_DIR "${CMAKE_BINARY_DIR}/demo/move/data")

if(NOT EXISTS ${MOVE_GENERATED_DIR}/move.c)
    file(MAKE_DIRECTORY ${MOVE_GENERATED_DIR})
    file(WRITE ${MOVE_GENERATED_DIR}/move.c "/* placeholder - run move_generate first */\n")
    file(WRITE ${MOVE_GENERATED_DIR}/move.h "/* placeholder - run move_generate first */\n")
    message(STATUS "Generated files not found, run move_generate first")
endif()

add_executable(move_train
    train_main.c
    ${MOVE_GENERATED_DIR}/move.c
)
target_link_libraries(move_train PRIVATE nn_infer_core)
target_include_directories(move_train PRIVATE
    ${CMAKE_SOURCE_DIR}/src/nn
    ${CMAKE_SOURCE_DIR}/src/nn/types/mlp
    ${MOVE_GENERATED_DIR}
    ${CMAKE_SOURCE_DIR}/src/infer
)
add_dependencies(move_train move_generate)

# 第三次编译：推理
# 权重文件在训练后生成，放在 build 目录
set(MOVE_WEIGHTS_DIR "${CMAKE_BINARY_DIR}/demo/move/data")
set(MOVE_WEIGHTS_SRC "")

if(EXISTS ${MOVE_WEIGHTS_DIR}/weights.c)
    set(MOVE_WEIGHTS_SRC ${MOVE_WEIGHTS_DIR}/weights.c)
endif()

add_executable(move_infer
    infer_main.c
    ${MOVE_GENERATED_DIR}/move.c
    ${MOVE_WEIGHTS_SRC}
)
target_link_libraries(move_infer PRIVATE nn_infer_core infer_core)
target_include_directories(move_infer PRIVATE
    ${CMAKE_SOURCE_DIR}/src/infer
    ${CMAKE_SOURCE_DIR}/src/nn
    ${CMAKE_SOURCE_DIR}/src/nn/types/mlp
    ${MOVE_GENERATED_DIR}
)
add_dependencies(move_infer move_train)

# 清理目标：删除编译产物
add_custom_target(move_clean
    COMMAND ${CMAKE_COMMAND} -E remove
        ${CMAKE_BINARY_DIR}/demo/move/Debug/move_generate.exe
        ${CMAKE_BINARY_DIR}/demo/move/Debug/move_train.exe
        ${CMAKE_BINARY_DIR}/demo/move/Debug/move_infer.exe
    COMMENT "Cleaning move demo build artifacts"
)
