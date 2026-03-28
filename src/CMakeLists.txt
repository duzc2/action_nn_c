option(ACTION_C_ENABLE_NN_MLP "Enable the built-in MLP network type" ON)
option(ACTION_C_ENABLE_NN_TRANSFORMER "Enable the built-in transformer network type" ON)
option(ACTION_C_ENABLE_NN_CNN "Enable the built-in CNN network type" OFF)
option(ACTION_C_ENABLE_NN_RNN "Enable the built-in RNN network type" OFF)
option(ACTION_C_ENABLE_NN_GNN "Enable the built-in GNN network type" OFF)

function(action_c_apply_strict_warnings TARGET_NAME)
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /W4 /WX)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endfunction()

if(NOT ACTION_C_ENABLE_NN_MLP AND
   NOT ACTION_C_ENABLE_NN_TRANSFORMER AND
   NOT ACTION_C_ENABLE_NN_CNN AND
   NOT ACTION_C_ENABLE_NN_RNN AND
   NOT ACTION_C_ENABLE_NN_GNN)
    message(FATAL_ERROR "At least one network type must be enabled for action_c")
endif()

add_subdirectory(nn)
add_subdirectory(infer)
add_subdirectory(train)
add_subdirectory(profiler)
