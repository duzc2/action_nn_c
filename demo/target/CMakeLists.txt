add_executable(target_generate
    generate_main.c
)
target_link_libraries(target_generate PRIVATE profiler_core)
target_include_directories(target_generate PRIVATE
    ${CMAKE_SOURCE_DIR}/src/profiler
)

add_executable(target_train
    train_main.c
)

add_executable(target_infer
    infer_main.c
)

# 清理目标
add_custom_target(target_clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_BINARY_DIR}/demo/target
    COMMENT "Cleaning target demo build artifacts"
)
