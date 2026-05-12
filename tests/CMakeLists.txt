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
action_c_add_test(test_arena utils/test_arena.c
    ${CMAKE_SOURCE_DIR}/src/utils/arena.c
    ${CMAKE_SOURCE_DIR}/src/utils/log.c)

# nn tests
action_c_add_test(test_mlp_arena_integration nn/test_mlp_arena_integration.c
    ${CMAKE_SOURCE_DIR}/src/utils/arena.c
    ${CMAKE_SOURCE_DIR}/src/utils/log.c)

action_c_add_test(test_gnn_arena_integration nn/test_gnn_arena_integration.c
    ${CMAKE_SOURCE_DIR}/src/utils/arena.c
    ${CMAKE_SOURCE_DIR}/src/utils/log.c)

action_c_add_test(test_rnn_pointer_swap nn/test_rnn_pointer_swap.c)

action_c_add_test(test_transformer_arena nn/test_transformer_arena.c
    ${CMAKE_SOURCE_DIR}/src/utils/arena.c
    ${CMAKE_SOURCE_DIR}/src/utils/log.c)

# Track B: VTable and deduplication tests
action_c_add_test(test_backend_struct_size nn/test_backend_struct_size.c)

action_c_add_test(test_weight_header nn/test_weight_header.c
    ${CMAKE_SOURCE_DIR}/src/nn/nn_weight_header.c
    ${CMAKE_SOURCE_DIR}/src/utils/log.c)

action_c_add_test(test_infer_registry nn/test_infer_registry.c
    ${CMAKE_SOURCE_DIR}/src/nn/nn_infer_registry.c
    ${CMAKE_SOURCE_DIR}/src/utils/log.c)
target_compile_definitions(test_infer_registry PRIVATE NN_REGISTRY_BUILD_TEST)

# Track B: MLP VTable tests
action_c_add_test(test_mlp_vtable nn/test_mlp_vtable.c
    ${CMAKE_SOURCE_DIR}/src/utils/log.c
    ${CMAKE_SOURCE_DIR}/src/utils/arena.c)
target_link_libraries(test_mlp_vtable nn_train_core)

# Track B: All backends VTable tests
action_c_add_test(test_all_vtables nn/test_all_vtables.c
    ${CMAKE_SOURCE_DIR}/src/utils/log.c
    ${CMAKE_SOURCE_DIR}/src/utils/arena.c)
target_link_libraries(test_all_vtables nn_train_core)
