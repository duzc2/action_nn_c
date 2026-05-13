cmake_minimum_required(VERSION 3.20)
set(CMAKE_C_EXTENSIONS OFF)
project(action_c_cs_tools C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

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

# cs_runtime_probe32: 32-bit probe DLL, Windows-only
if(NOT WIN32)
    message(WARNING "cs_tools: cs_runtime_probe32 requires Windows. Skipping 32-bit probe DLL build.")
elseif(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    add_library(cs_runtime_probe32 SHARED
        cs_runtime_probe32.c
    )
    target_include_directories(cs_runtime_probe32 PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"
    )
    target_compile_definitions(cs_runtime_probe32 PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(cs_runtime_probe32 PRIVATE
        $<$<OR:$<C_COMPILER_ID:MSVC>,$<C_COMPILER_ID:Clang>>:/W4>
        $<$<OR:$<C_COMPILER_ID:MSVC>,$<C_COMPILER_ID:Clang>>:/WX>
    )
    set_target_properties(cs_runtime_probe32 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    add_dependencies(cs_state_trace cs_runtime_probe32)
else()
    message(WARNING "cs_tools: cs_runtime_probe32 requires MSVC or Clang on Windows. Compiler detected: ${CMAKE_C_COMPILER_ID}. Skipping 32-bit probe DLL build.")
endif()

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
