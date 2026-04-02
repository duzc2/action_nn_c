cmake_minimum_required(VERSION 3.20)
project(action_c_cs_tools C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

include("${CMAKE_CURRENT_LIST_DIR}/../../demo_common.cmake")

add_library(cs_v1_tool_common STATIC
    cs_tool_common.c
)

target_include_directories(cs_v1_tool_common PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}"
)
target_compile_definitions(cs_v1_tool_common PUBLIC _CRT_SECURE_NO_WARNINGS)

if(WIN32)
    target_link_libraries(cs_v1_tool_common PUBLIC user32 gdi32)
endif()

action_c_apply_strict_warnings(cs_v1_tool_common)

add_executable(cs_capture_session
    cs_capture_session.c
)
target_link_libraries(cs_capture_session PRIVATE cs_v1_tool_common)
action_c_apply_strict_warnings(cs_capture_session)

add_executable(cs_label_session
    cs_label_session.c
)
target_link_libraries(cs_label_session PRIVATE cs_v1_tool_common)
action_c_apply_strict_warnings(cs_label_session)

add_executable(cs_dataset_build
    cs_dataset_build.c
)
target_link_libraries(cs_dataset_build PRIVATE cs_v1_tool_common)
action_c_apply_strict_warnings(cs_dataset_build)

add_executable(cs_dataset_report
    cs_dataset_report.c
)
target_link_libraries(cs_dataset_report PRIVATE cs_v1_tool_common)
action_c_apply_strict_warnings(cs_dataset_report)
