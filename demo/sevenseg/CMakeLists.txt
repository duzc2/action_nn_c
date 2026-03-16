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

add_executable(sevenseg_train
    train_main.c
    sevenseg_shared.c
)
target_include_directories(sevenseg_train PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/train/include
)
target_link_libraries(sevenseg_train PRIVATE dnn_train_sevenseg)
target_compile_definitions(sevenseg_train PRIVATE
    STATE_DIM=1
    OUTPUT_DIM=7
)

set(SEVENSEG_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)
set(SEVENSEG_VOCAB ${SEVENSEG_DATA_DIR}/demo_vocab_sevenseg.txt)
set(SEVENSEG_BIN ${SEVENSEG_DATA_DIR}/demo_weights_sevenseg.bin)
set(SEVENSEG_C_ARRAY ${SEVENSEG_DATA_DIR}/demo_weights_sevenseg_export.c)
set(SEVENSEG_C_FUNC ${SEVENSEG_DATA_DIR}/demo_network_sevenseg_functions.c)

add_custom_command(
    OUTPUT ${SEVENSEG_VOCAB} ${SEVENSEG_BIN} ${SEVENSEG_C_ARRAY} ${SEVENSEG_C_FUNC}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SEVENSEG_DATA_DIR}
    COMMAND $<TARGET_FILE:sevenseg_train> --export-only ${SEVENSEG_DATA_DIR}
    DEPENDS sevenseg_train
    VERBATIM
)

add_custom_target(sevenseg_generated_assets
    DEPENDS ${SEVENSEG_VOCAB} ${SEVENSEG_BIN} ${SEVENSEG_C_ARRAY} ${SEVENSEG_C_FUNC}
)

add_executable(sevenseg_infer_bin
    infer_bin_main.c
    sevenseg_shared.c
)
target_include_directories(sevenseg_infer_bin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/train/include
)
target_link_libraries(sevenseg_infer_bin PRIVATE dnn_train_sevenseg)
target_compile_definitions(sevenseg_infer_bin PRIVATE
    STATE_DIM=1
    OUTPUT_DIM=7
)
add_dependencies(sevenseg_infer_bin sevenseg_generated_assets)

add_executable(sevenseg_infer_c_array
    infer_c_array_main.c
    sevenseg_shared.c
    ${SEVENSEG_C_ARRAY}
)
target_include_directories(sevenseg_infer_c_array PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/train/include
)
target_link_libraries(sevenseg_infer_c_array PRIVATE dnn_train_sevenseg)
target_compile_definitions(sevenseg_infer_c_array PRIVATE
    STATE_DIM=1
    OUTPUT_DIM=7
)
add_dependencies(sevenseg_infer_c_array sevenseg_generated_assets)

add_executable(sevenseg_infer_c_func
    infer_c_func_main.c
    sevenseg_shared.c
    ${SEVENSEG_C_FUNC}
)
target_include_directories(sevenseg_infer_c_func PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/train/include
)
target_link_libraries(sevenseg_infer_c_func PRIVATE dnn_train_sevenseg)
target_compile_definitions(sevenseg_infer_c_func PRIVATE
    STATE_DIM=1
    OUTPUT_DIM=7
)
add_dependencies(sevenseg_infer_c_func sevenseg_generated_assets)

add_executable(sevenseg_benchmark
    benchmark_main.c
    ${SEVENSEG_C_ARRAY}
    ${SEVENSEG_C_FUNC}
)
target_include_directories(sevenseg_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/train/include
)
target_link_libraries(sevenseg_benchmark PRIVATE dnn_train_sevenseg)
target_compile_definitions(sevenseg_benchmark PRIVATE
    STATE_DIM=1
    OUTPUT_DIM=7
)
add_dependencies(sevenseg_benchmark sevenseg_generated_assets)

if (UNIX OR MINGW)
    target_link_libraries(dnn_infer_sevenseg PUBLIC m)
    target_link_libraries(dnn_train_sevenseg PUBLIC m)
    target_link_libraries(sevenseg_train PRIVATE m)
    target_link_libraries(sevenseg_infer_bin PRIVATE m)
    target_link_libraries(sevenseg_infer_c_array PRIVATE m)
    target_link_libraries(sevenseg_infer_c_func PRIVATE m)
    target_link_libraries(sevenseg_benchmark PRIVATE m)
endif()
