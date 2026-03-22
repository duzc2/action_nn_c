# 第一次编译：只编译 profiler 和生成器
add_executable(sevenseg_generate
    generate_main.c
)
target_link_libraries(sevenseg_generate PRIVATE profiler_core)
target_include_directories(sevenseg_generate PRIVATE
    ${CMAKE_SOURCE_DIR}/src/profiler
)

# 第二次编译：训练
set(SEVENSEG_GENERATED_DIR "${CMAKE_BINARY_DIR}/demo/sevenseg/data")

if(NOT EXISTS ${SEVENSEG_GENERATED_DIR}/sevenseg.c)
    file(MAKE_DIRECTORY ${SEVENSEG_GENERATED_DIR})
    file(WRITE ${SEVENSEG_GENERATED_DIR}/sevenseg.c "/* placeholder - run sevenseg_generate first */\n")
    file(WRITE ${SEVENSEG_GENERATED_DIR}/sevenseg.h "/* placeholder - run sevenseg_generate first */\n")
    message(STATUS "Generated files not found, run sevenseg_generate first")
endif()

add_executable(sevenseg_train
    train_main.c
    ${SEVENSEG_GENERATED_DIR}/sevenseg.c
)
target_link_libraries(sevenseg_train PRIVATE nn_infer_core)
target_include_directories(sevenseg_train PRIVATE
    ${CMAKE_SOURCE_DIR}/src/nn
    ${CMAKE_SOURCE_DIR}/src/nn/types/mlp
    ${SEVENSEG_GENERATED_DIR}
    ${CMAKE_SOURCE_DIR}/src/infer
)
add_dependencies(sevenseg_train sevenseg_generate)

# 第三次编译：推理
set(SEVENSEG_WEIGHTS_SRC "")

if(EXISTS ${SEVENSEG_GENERATED_DIR}/weights.c)
    set(SEVENSEG_WEIGHTS_SRC ${SEVENSEG_GENERATED_DIR}/weights.c)
endif()

add_executable(sevenseg_infer
    infer_main.c
    ${SEVENSEG_GENERATED_DIR}/sevenseg.c
    ${SEVENSEG_WEIGHTS_SRC}
)
target_link_libraries(sevenseg_infer PRIVATE infer_core)
target_include_directories(sevenseg_infer PRIVATE
    ${CMAKE_SOURCE_DIR}/src/infer
    ${CMAKE_SOURCE_DIR}/src/nn/types/transformer
    ${CMAKE_SOURCE_DIR}/src/nn/types/mlp
    ${SEVENSEG_GENERATED_DIR}
)
add_dependencies(sevenseg_infer sevenseg_train)

# 清理目标
add_custom_target(sevenseg_clean
    COMMAND ${CMAKE_COMMAND} -E remove
        ${CMAKE_BINARY_DIR}/demo/sevenseg/Debug/sevenseg_generate.exe
        ${CMAKE_BINARY_DIR}/demo/sevenseg/Debug/sevenseg_train.exe
        ${CMAKE_BINARY_DIR}/demo/sevenseg/Debug/sevenseg_infer.exe
    COMMENT "Cleaning sevenseg demo build artifacts"
)
