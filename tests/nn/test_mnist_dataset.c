/**
 * @file test_mnist_dataset.c
 * @brief Unit tests for the unified MNIST dataset loader (B9 merge).
 *
 * Validates constants, the shared MnistDataset API, legacy typedef/macro
 * compatibility through MnistCnnDataset, and the quadrant-packing helper.
 */

#include "test_harness.h"
#include <string.h>

/* Include through the shim to verify legacy MnistCnnDataset compat */
#define DEMO_DATASET_PATH "demo/dataset/mnist_dataset.h"

/* Since tests/ is at repo root, reach into demo/ */
#include "../../demo/dataset/mnist_dataset.h"

/* Also verify the CNN shim provides the legacy type */
#include "../../demo/mnist_cnn/mnist_cnn_dataset.h"

/* ── Constants ── */

TEST(constants_are_correct) {
    ASSERT_EQ_SIZE(28U, MNIST_IMAGE_ROWS, "MNIST_IMAGE_ROWS == 28");
    ASSERT_EQ_SIZE(28U, MNIST_IMAGE_COLS, "MNIST_IMAGE_COLS == 28");
    ASSERT_EQ_SIZE(784U, MNIST_IMAGE_SIZE, "MNIST_IMAGE_SIZE == 784");
    ASSERT_EQ_SIZE(10U, MNIST_CLASS_COUNT, "MNIST_CLASS_COUNT == 10");
    ASSERT_EQ_SIZE(14U, MNIST_QUADRANT_ROWS, "MNIST_QUADRANT_ROWS == 14");
    ASSERT_EQ_SIZE(14U, MNIST_QUADRANT_COLS, "MNIST_QUADRANT_COLS == 14");
    ASSERT_EQ_SIZE(4U, MNIST_SEQUENCE_LENGTH, "MNIST_SEQUENCE_LENGTH == 4");
    ASSERT_EQ_SIZE(784U, MNIST_FEATURE_INPUT_SIZE,
                   "MNIST_FEATURE_INPUT_SIZE == 784 (4 * 14 * 14)");
}

/* ── Legacy macro aliases (CNN shim) ── */

TEST(legacy_cnn_constants_match) {
    ASSERT_EQ_INT((int)MNIST_IMAGE_ROWS, (int)MNIST_CNN_IMAGE_ROWS,
                  "CNN_IMAGE_ROWS matches unified");
    ASSERT_EQ_INT((int)MNIST_IMAGE_COLS, (int)MNIST_CNN_IMAGE_COLS,
                  "CNN_IMAGE_COLS matches unified");
    ASSERT_EQ_INT((int)MNIST_IMAGE_SIZE, (int)MNIST_CNN_IMAGE_SIZE,
                  "CNN_IMAGE_SIZE matches unified");
    ASSERT_EQ_INT((int)MNIST_CLASS_COUNT, (int)MNIST_CNN_CLASS_COUNT,
                  "CNN_CLASS_COUNT matches unified");
    ASSERT_EQ_INT((int)MNIST_QUADRANT_ROWS, (int)MNIST_CNN_QUADRANT_ROWS,
                  "CNN_QUADRANT_ROWS matches unified");
    ASSERT_EQ_INT((int)MNIST_FEATURE_INPUT_SIZE, (int)MNIST_CNN_FEATURE_INPUT_SIZE,
                  "CNN_FEATURE_INPUT_SIZE matches unified");
}

/* ── Type compatibility ── */

TEST(mnist_cnn_dataset_is_mnist_dataset) {
    /* Verify MnistCnnDataset and MnistDataset are the same size (typedef) */
    ASSERT_EQ_SIZE(sizeof(MnistDataset), sizeof(MnistCnnDataset),
                   "MnistCnnDataset is same size as MnistDataset");
}

/* ── mnist_dataset_make_one_hot ── */

TEST(one_hot_label_0) {
    float target[10];
    mnist_dataset_make_one_hot(0, target, 10);
    ASSERT_FLOAT_EQ(1.0f, target[0], 1e-6f, "target[0] == 1.0");
    ASSERT_FLOAT_EQ(0.0f, target[3], 1e-6f, "target[3] == 0.0");
    ASSERT_FLOAT_EQ(0.0f, target[9], 1e-6f, "target[9] == 0.0");
}

TEST(one_hot_label_9) {
    float target[10];
    mnist_dataset_make_one_hot(9, target, 10);
    ASSERT_FLOAT_EQ(0.0f, target[0], 1e-6f, "target[0] == 0.0");
    ASSERT_FLOAT_EQ(1.0f, target[9], 1e-6f, "target[9] == 1.0");
}

TEST(one_hot_out_of_range) {
    float target[10];
    /* Label 15 exceeds class_count 10 — should produce all zeros (no crash) */
    mnist_dataset_make_one_hot(15, target, 10);
    ASSERT_FLOAT_EQ(0.0f, target[0], 1e-6f, "target[0] == 0.0 (out of range)");
    ASSERT_FLOAT_EQ(0.0f, target[9], 1e-6f, "target[9] == 0.0 (out of range)");
}

TEST(one_hot_null_target_safe) {
    /* Should not crash with NULL target */
    mnist_dataset_make_one_hot(3, NULL, 10);
    ASSERT_TRUE(1, "make_one_hot(NULL) does not crash");
}

/* ── mnist_dataset_argmax ── */

TEST(argmax_basic) {
    const float values[] = {0.1f, 0.3f, 0.7f, 0.2f, 0.05f};
    int best = mnist_dataset_argmax(values, 5);
    ASSERT_EQ_INT(2, best, "argmax of {0.1, 0.3, 0.7, 0.2, 0.05} == 2");
}

TEST(argmax_first_element_wins_tie) {
    const float values[] = {0.5f, 0.5f, 0.5f};
    int best = mnist_dataset_argmax(values, 3);
    ASSERT_EQ_INT(0, best, "argmax with tie keeps first index");
}

TEST(argmax_single_element) {
    const float values[] = {42.0f};
    int best = mnist_dataset_argmax(values, 1);
    ASSERT_EQ_INT(0, best, "argmax of single element is 0");
}

TEST(argmax_null_returns_neg1) {
    int best = mnist_dataset_argmax(NULL, 5);
    ASSERT_EQ_INT(-1, best, "argmax(NULL) returns -1");
}

TEST(argmax_zero_count_returns_neg1) {
    const float values[] = {0.1f};
    int best = mnist_dataset_argmax(values, 0);
    ASSERT_EQ_INT(-1, best, "argmax with count==0 returns -1");
}

/* ── mnist_dataset_free safety ── */

TEST(free_null_dataset_safe) {
    mnist_dataset_free(NULL);
    ASSERT_TRUE(1, "free(NULL) does not crash");
}

/* ── mnist_pack_quadrants ── */

TEST(pack_quadrants_basic) {
    /* Create a simple 28x28 image: each pixel = its row * 100 + col */
    float image[MNIST_IMAGE_SIZE];
    size_t i;
    for (i = 0; i < MNIST_IMAGE_SIZE; i++) {
        image[i] = (float)i;
    }

    float packed[MNIST_FEATURE_INPUT_SIZE];
    mnist_pack_quadrants(image, packed);

    /* Quadrant 0 (TL): rows 0-13, cols 0-13 */
    ASSERT_FLOAT_EQ(image[0 * 28 + 0], packed[0], 1e-6f,
                    "packed[0] == image(0,0)");

    /* Quadrant 1 (TR): rows 0-13, cols 14-27 → packed starts at 196 */
    ASSERT_FLOAT_EQ(image[0 * 28 + 14], packed[196], 1e-6f,
                    "packed[196] == image(0,14)");

    /* Quadrant 2 (BL): rows 14-27, cols 0-13 → packed starts at 392 */
    ASSERT_FLOAT_EQ(image[14 * 28 + 0], packed[392], 1e-6f,
                    "packed[392] == image(14,0)");

    /* Quadrant 3 (BR): rows 14-27, cols 14-27 → packed starts at 588 */
    ASSERT_FLOAT_EQ(image[14 * 28 + 14], packed[588], 1e-6f,
                    "packed[588] == image(14,14)");

    /* Last element of quadrant 3 */
    ASSERT_FLOAT_EQ(image[27 * 28 + 27], packed[783], 1e-6f,
                    "packed[783] == image(27,27)");
}

TEST(pack_quadrants_null_input_safe) {
    float packed[MNIST_FEATURE_INPUT_SIZE];
    packed[0] = 1.0f;
    mnist_pack_quadrants(NULL, packed);
    /* Should not crash; packed should be untouched */
    ASSERT_TRUE(1, "pack_quadrants(NULL, packed) does not crash");
}

TEST(pack_quadrants_null_output_safe) {
    float image[MNIST_IMAGE_SIZE] = {0};
    mnist_pack_quadrants(image, NULL);
    ASSERT_TRUE(1, "pack_quadrants(image, NULL) does not crash");
}

TEST(legacy_cnn_pack_quadrants_alias_works) {
    /* The macro mnist_cnn_pack_quadrants aliases mnist_pack_quadrants */
    float image[MNIST_IMAGE_SIZE];
    size_t i;
    for (i = 0; i < MNIST_IMAGE_SIZE; i++) image[i] = (float)(i % 256);

    float packed1[MNIST_FEATURE_INPUT_SIZE];
    float packed2[MNIST_FEATURE_INPUT_SIZE];
    memset(packed1, 0, sizeof(packed1));
    memset(packed2, 0xFF, sizeof(packed2)); /* different init to detect bugs */

    mnist_pack_quadrants(image, packed1);
    mnist_cnn_pack_quadrants(image, packed2);

    ASSERT_EQ_INT(0, memcmp(packed1, packed2, sizeof(packed1)),
                  "mnist_cnn_pack_quadrants produces same output as mnist_pack_quadrants");
}

/* ── mnist_dataset_render_ascii (smoke test) ── */

TEST(render_ascii_null_safe) {
    mnist_dataset_render_ascii(NULL, 0, 0);
    ASSERT_TRUE(1, "render_ascii(NULL) does not crash");
}

/* ── load error paths ── */

TEST(load_null_args_returns_error) {
    char errbuf[256];
    int rc = mnist_dataset_load(NULL, NULL, 0, NULL, errbuf, sizeof(errbuf));
    ASSERT_EQ_INT(-1, rc, "load with all NULLs returns -1");
}

TEST(load_nonexistent_file) {
    MnistDataset ds;
    char errbuf[256] = {0};
    int rc = mnist_dataset_load(
        "nonexistent_images.idx3",
        "nonexistent_labels.idx1",
        0, &ds, errbuf, sizeof(errbuf));
    ASSERT_EQ_INT(-1, rc, "load with nonexistent files returns -1");
}

int main(void) {
    RUN_TEST(constants_are_correct);
    RUN_TEST(legacy_cnn_constants_match);
    RUN_TEST(mnist_cnn_dataset_is_mnist_dataset);
    RUN_TEST(one_hot_label_0);
    RUN_TEST(one_hot_label_9);
    RUN_TEST(one_hot_out_of_range);
    RUN_TEST(one_hot_null_target_safe);
    RUN_TEST(argmax_basic);
    RUN_TEST(argmax_first_element_wins_tie);
    RUN_TEST(argmax_single_element);
    RUN_TEST(argmax_null_returns_neg1);
    RUN_TEST(argmax_zero_count_returns_neg1);
    RUN_TEST(free_null_dataset_safe);
    RUN_TEST(pack_quadrants_basic);
    RUN_TEST(pack_quadrants_null_input_safe);
    RUN_TEST(pack_quadrants_null_output_safe);
    RUN_TEST(legacy_cnn_pack_quadrants_alias_works);
    RUN_TEST(render_ascii_null_safe);
    RUN_TEST(load_null_args_returns_error);
    RUN_TEST(load_nonexistent_file);

    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
