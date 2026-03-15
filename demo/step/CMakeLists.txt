add_executable(step_demo
    main.c
)

target_include_directories(step_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/train/include
)

target_link_libraries(step_demo PRIVATE dnn_train)

if (UNIX OR MINGW)
    target_link_libraries(step_demo PRIVATE m)
endif()
