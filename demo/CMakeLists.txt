cmake_minimum_required(VERSION 3.20)

# Demo parent directory
# This file is used for unified demo builds

# Get build output directory from parent
if(NOT DEFINED DEMO_BUILD_ROOT)
    set(DEMO_BUILD_ROOT "${CMAKE_BINARY_DIR}/demo")
endif()

# Subdirectories are configured individually
# See individual demo/xxx/generate/CMakeLists.txt
