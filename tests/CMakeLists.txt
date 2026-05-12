# 启用 CTest
enable_testing()

# 添加测试目标辅助宏
function(action_c_add_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE} ${ARGN})
    target_include_directories(${TEST_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )
    if(MSVC)
        target_compile_options(${TEST_NAME} PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    set_property(TARGET ${TEST_NAME} PROPERTY C_STANDARD 11)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Smoke test
action_c_add_test(test_harness_smoke smoke_test.c)

# utils tests
action_c_add_test(test_error utils/test_error.c)
action_c_add_test(test_log utils/test_log.c ${CMAKE_SOURCE_DIR}/src/utils/log.c)
action_c_add_test(test_safe_math utils/test_safe_math.c)
