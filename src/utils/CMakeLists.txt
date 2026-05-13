add_library(utils_core STATIC
    arena.c
    log.c
)
action_c_apply_strict_warnings(utils_core)
target_include_directories(utils_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)
