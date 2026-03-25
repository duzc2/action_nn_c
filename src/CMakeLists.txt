option(ACTION_C_ENABLE_NN_MLP "Enable the built-in MLP network type" ON)
option(ACTION_C_ENABLE_NN_TRANSFORMER "Enable the built-in transformer network type" ON)

if(NOT ACTION_C_ENABLE_NN_MLP AND NOT ACTION_C_ENABLE_NN_TRANSFORMER)
    message(FATAL_ERROR "At least one network type must be enabled for action_c")
endif()

add_subdirectory(nn)
add_subdirectory(infer)
add_subdirectory(train)
add_subdirectory(profiler)
