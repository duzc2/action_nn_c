add_executable(transformer_simple_demo
    main.c
)

target_include_directories(transformer_simple_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/train/include
)

target_link_libraries(transformer_simple_demo PRIVATE dnn_train)

if (UNIX OR MINGW)
    target_link_libraries(transformer_simple_demo PRIVATE m)
endif()
