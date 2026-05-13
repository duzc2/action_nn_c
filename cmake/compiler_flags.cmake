# --- Single source of truth: compiler flags ---
# This file is include()'d by all CMakeLists.txt
#
# Provides:
#   action_c_apply_strict_warnings(TARGET)  -- /W4 /WX /utf-8 (MSVC) or -Wall -Wextra -Wpedantic -Werror
#   action_c_apply_hardening(TARGET)        -- GS/guard:cf/sdl (MSVC) or FORTIFY/stack-protector/PIE
#   action_c_enable_lto(TARGET)             -- IPO/LTO when supported
#   action_c_apply_all_flags(TARGET)        -- strict_warnings + hardening + lto

function(action_c_apply_strict_warnings TARGET_NAME)
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /W4 /WX /utf-8)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE
            -Wall -Wextra -Wpedantic -Werror
        )
    endif()
endfunction()

function(action_c_apply_hardening TARGET_NAME)
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE
            /GS /guard:cf /DYNAMICBASE /sdl
        )
        target_link_options(${TARGET_NAME} PRIVATE
            /HIGHENTROPYVA /NXCOMPAT
        )
    else()
        target_compile_options(${TARGET_NAME} PRIVATE
            -D_FORTIFY_SOURCE=2
            -fstack-protector-strong
            -fPIE -fPIC
        )
        target_link_options(${TARGET_NAME} PRIVATE
            -Wl,-z,relro -Wl,-z,now -pie
        )
    endif()
endfunction()

function(action_c_enable_lto TARGET_NAME)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
    if(ipo_supported)
        set_property(TARGET ${TARGET_NAME}
            PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()

function(action_c_apply_all_flags TARGET_NAME)
    action_c_apply_strict_warnings(${TARGET_NAME})
    action_c_apply_hardening(${TARGET_NAME})
    action_c_enable_lto(${TARGET_NAME})
endfunction()
