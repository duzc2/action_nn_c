cmake_minimum_required(VERSION 3.20)
set(CMAKE_C_EXTENSIONS OFF)
project(test_verify C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_definitions(-D_UNUSED)

include(${CMAKE_CURRENT_LIST_DIR}/../demo/demo_common.cmake)
action_c_enable_network_types(TYPES cnn cnn_dual_pool mlp)
action_c_add_core_subdirectory()

# ── test_stride: unit-level stride verification ──
add_executable(test_stride test_stride.c)
action_c_apply_strict_warnings(test_stride)
target_include_directories(test_stride PRIVATE "${ACTION_C_ROOT}/src")
target_link_libraries(test_stride PRIVATE nn_train_core)

# ── test_network_e2e: full generated network on single sample ──
set(GENERATED_DIR "${ACTION_C_ROOT}/build/demo/edge_video_preprocess/data")
set(GENERATED_INFER_C   "${GENERATED_DIR}/infer.c")
set(GENERATED_TRAIN_C   "${GENERATED_DIR}/train.c")
set(GENERATED_WEIGHTS_SAVE_C "${GENERATED_DIR}/weights_save.c")
set(GENERATED_WEIGHTS_LOAD_C "${GENERATED_DIR}/weights_load.c")
set_source_files_properties(
    "${GENERATED_INFER_C}"
    "${GENERATED_TRAIN_C}"
    "${GENERATED_WEIGHTS_SAVE_C}"
    "${GENERATED_WEIGHTS_LOAD_C}"
    PROPERTIES GENERATED TRUE
)

add_executable(test_network_e2e
    test_network_e2e.c
    "${GENERATED_INFER_C}"
    "${GENERATED_TRAIN_C}"
    "${GENERATED_WEIGHTS_SAVE_C}"
    "${GENERATED_WEIGHTS_LOAD_C}"
)
target_include_directories(test_network_e2e PRIVATE
    "${ACTION_C_ROOT}/src"
    "${GENERATED_DIR}"
)
target_link_libraries(test_network_e2e PRIVATE nn_train_core)

# ── quick_train: trains on CIFAR-10 subset, saves weights.bin ──
add_executable(quick_train
    quick_train.c
    ../demo/edge_video_preprocess/cifar10_dataset.c
    "${GENERATED_INFER_C}"
    "${GENERATED_TRAIN_C}"
    "${GENERATED_WEIGHTS_SAVE_C}"
)
target_include_directories(quick_train PRIVATE
    "${ACTION_C_ROOT}/src"
    "${GENERATED_DIR}"
    "${ACTION_C_ROOT}/demo/edge_video_preprocess"
)
target_link_libraries(quick_train PRIVATE nn_train_core)

# ── test_cnn_backend: standalone CNN backend training test ──
add_executable(test_cnn_backend test_cnn_backend.c)
target_include_directories(test_cnn_backend PRIVATE "${ACTION_C_ROOT}/src")
target_link_libraries(test_cnn_backend PRIVATE nn_train_core)

# ── test_bn_consistency: Bug #5 BN forward/backward cache consistency ──
add_executable(test_bn_consistency test_bn_consistency.c)
target_include_directories(test_bn_consistency PRIVATE "${ACTION_C_ROOT}/src")
target_link_libraries(test_bn_consistency PRIVATE nn_train_core)

# ── test_cnn_full: comprehensive CNN unit test (24 test cases) ──
add_executable(test_cnn_full test_cnn_full.c)
target_include_directories(test_cnn_full PRIVATE "${ACTION_C_ROOT}/src")
target_link_libraries(test_cnn_full PRIVATE nn_train_core)

# ── test_motion_detect: adaptive threshold + temporal smoother ──
add_executable(test_motion_detect
    test_motion_detect.c
    ../demo/edge_video_preprocess/video_processor.c
)
target_include_directories(test_motion_detect PRIVATE
    "${ACTION_C_ROOT}/src"
    "${ACTION_C_ROOT}/demo/edge_video_preprocess"
)
target_link_libraries(test_motion_detect PRIVATE nn_train_core)

# ── test_motion_accuracy: precision/recall/F1 benchmark on synthetic frames ──
add_executable(test_motion_accuracy
    test_motion_accuracy.c
    ../demo/edge_video_preprocess/video_processor.c
)
target_include_directories(test_motion_accuracy PRIVATE
    "${ACTION_C_ROOT}/src"
    "${ACTION_C_ROOT}/demo/edge_video_preprocess"
)
target_link_libraries(test_motion_accuracy PRIVATE nn_train_core)
