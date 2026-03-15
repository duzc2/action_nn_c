add_executable(goal_demo
    main.c
)

target_include_directories(goal_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/train/include
)

target_link_libraries(goal_demo PRIVATE dnn_train)

if (UNIX OR MINGW)
    target_link_libraries(goal_demo PRIVATE m)
endif()
