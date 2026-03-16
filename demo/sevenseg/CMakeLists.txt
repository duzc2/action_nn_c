add_executable(sevenseg_demo
    main.c
)

target_include_directories(sevenseg_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/train/include
)

target_link_libraries(sevenseg_demo PRIVATE dnn_train)

if (UNIX OR MINGW)
    target_link_libraries(sevenseg_demo PRIVATE m)
endif()
