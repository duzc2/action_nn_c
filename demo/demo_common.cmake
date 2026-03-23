set(ACTION_C_DEMO_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}")
get_filename_component(ACTION_C_ROOT "${ACTION_C_DEMO_COMMON_DIR}/.." ABSOLUTE)

function(action_c_add_core_subdirectory)
    add_subdirectory("${ACTION_C_ROOT}/src" "${CMAKE_BINARY_DIR}/action_c_src")
endfunction()

function(action_c_get_demo_generated_dir OUT_VAR)
    get_filename_component(_generated_dir "${CMAKE_BINARY_DIR}/../data" ABSOLUTE)
    set(${OUT_VAR} "${_generated_dir}" PARENT_SCOPE)
endfunction()

function(action_c_add_demo_clean_target DEMO_NAME)
    add_custom_target(${DEMO_NAME}_clean
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${CMAKE_CURRENT_SOURCE_DIR}/../build/${DEMO_NAME}"
        COMMENT "Clean generated and built files for ${DEMO_NAME} demo"
    )
endfunction()
