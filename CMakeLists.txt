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
