cmake_minimum_required(VERSION 3.20)
project(action_c C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_subdirectory(src)
add_subdirectory(demo/sevenseg)
add_subdirectory(demo/target)
add_subdirectory(demo/move)
add_subdirectory(demo/transformer)

# 清理所有编译结果（.exe, .obj, .lib 等），保留源码
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_BINARY_DIR}/src
    COMMENT "Cleaning all build artifacts (executables, objects, libraries)"
)

# 清理所有（包括 CMake 缓存和生成的源码文件）
add_custom_target(distclean
    COMMAND ${CMAKE_COMMAND} -E remove_directory
        ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all build artifacts and CMake cache"
)
