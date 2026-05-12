cmake_minimum_required(VERSION 3.20)
project(action_c C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_subdirectory(src)

# 添加测试（仅在启用时）
option(BUILD_TESTING "Enable CTest" ON)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
