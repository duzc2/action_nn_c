add_executable(sevenseg_generate
    generate_main.c
)
target_link_libraries(sevenseg_generate PRIVATE profiler_core)
target_include_directories(sevenseg_generate PRIVATE
    ${CMAKE_SOURCE_DIR}/src/profiler
)

add_executable(sevenseg_train
    train_main.c
)

add_executable(sevenseg_infer
    infer_main.c
)

# 清理目标
add_custom_target(sevenseg_clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_BINARY_DIR}/demo/sevenseg
    COMMENT "Cleaning sevenseg demo build artifacts"
)
