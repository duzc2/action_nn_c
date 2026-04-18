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

add_executable(cs_state_trace
    cs_state_trace.c
)
target_link_libraries(cs_state_trace PRIVATE cs_v1_tool_common)
action_c_apply_strict_warnings(cs_state_trace)

set(CS_RUNTIME_PROBE32_DLL "${CMAKE_CURRENT_BINARY_DIR}/cs_runtime_probe32.dll")
set(CS_RUNTIME_PROBE32_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/build_cs_runtime_probe32.bat")
string(CONCAT CS_RUNTIME_PROBE32_SCRIPT_CONTENT
    "@echo off\r\n"
    "call \"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars32.bat\"\r\n"
    "if errorlevel 1 exit /b 1\r\n"
    "clang-cl /nologo /TC /LD --target=i686-pc-windows-msvc /W4 /WX /I \"${CMAKE_CURRENT_LIST_DIR}\" /Fe:\"${CS_RUNTIME_PROBE32_DLL}\" \"${CMAKE_CURRENT_LIST_DIR}/cs_runtime_probe32.c\"\r\n"
)
file(GENERATE OUTPUT "${CS_RUNTIME_PROBE32_SCRIPT}" CONTENT "${CS_RUNTIME_PROBE32_SCRIPT_CONTENT}")
add_custom_command(
    OUTPUT "${CS_RUNTIME_PROBE32_DLL}"
    COMMAND "${CS_RUNTIME_PROBE32_SCRIPT}"
    DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/cs_runtime_probe32.c"
        "${CMAKE_CURRENT_LIST_DIR}/cs_runtime_probe_shared.h"
    VERBATIM
)
add_custom_target(cs_runtime_probe32 ALL DEPENDS "${CS_RUNTIME_PROBE32_DLL}")
add_dependencies(cs_state_trace cs_runtime_probe32)

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
