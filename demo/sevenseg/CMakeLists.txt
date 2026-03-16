add_library(dnn_infer_sevenseg STATIC
    ../../src/core/tensor.c
    ../../src/core/ops.c
    ../../src/core/tokenizer.c
    ../../src/core/weights_io.c
    ../../src/core/protocol.c
    ../../src/infer/workflow_runtime.c
    ../../src/tokenizer/tokenizer_runtime.c
    ../../src/model/transformer.c
)
target_include_directories(dnn_infer_sevenseg PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infer/include
)
target_compile_definitions(dnn_infer_sevenseg PUBLIC
    STATE_DIM=1
    OUTPUT_DIM=7
)

add_library(dnn_train_sevenseg STATIC
    ../../src/train/workflow_train.c
    ../../src/train/csv_loader.c
)
target_include_directories(dnn_train_sevenseg PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/train/include
)
target_link_libraries(dnn_train_sevenseg PUBLIC dnn_infer_sevenseg)
target_compile_definitions(dnn_train_sevenseg PUBLIC
    STATE_DIM=1
    OUTPUT_DIM=7
)

add_executable(sevenseg_demo
    main.c
)
target_include_directories(sevenseg_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/train/include
)
target_link_libraries(sevenseg_demo PRIVATE dnn_train_sevenseg)
target_compile_definitions(sevenseg_demo PRIVATE
    STATE_DIM=1
    OUTPUT_DIM=7
)

if (UNIX OR MINGW)
    target_link_libraries(dnn_infer_sevenseg PUBLIC m)
    target_link_libraries(dnn_train_sevenseg PUBLIC m)
    target_link_libraries(sevenseg_demo PRIVATE m)
endif()
