/**
 * @file train_main.c
 * @brief Hybrid transformer+MLP route demo training entry
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "../demo_runtime_paths.h"

#include <math.h>
#include <stdio.h>

#define HYBRID_ROUTE_INPUT_SIZE 8U
#define HYBRID_ROUTE_OUTPUT_SIZE 2U
#define HYBRID_ROUTE_EPOCHS 32

/**
 * @brief Quantized cue values used to generate the toy dataset.
 */
static const float g_values[] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};

/**
 * @brief Normalize a 2D vector when it has meaningful length.
 */
static void normalize_vector(float* x, float* y) {
    float length = sqrtf((*x * *x) + (*y * *y));

    if (length > 0.00001f) {
        *x /= length;
        *y /= length;
    }
}

/**
 * @brief Build a human-readable teaching rule.
 *
 * The scene being modeled is:
 * - an upstream planner leaves four recent local route cues
 * - the lower controller should continue along the emerging branch
 * - later cues matter more than earlier cues
 *
 * So the label is simply a weighted average of the four route cues.
 */
static void build_expected(const float* input, float* expected) {
    expected[0] =
        0.10f * input[0] +
        0.18f * input[2] +
        0.28f * input[4] +
        0.44f * input[6];
    expected[1] =
        0.10f * input[1] +
        0.18f * input[3] +
        0.28f * input[5] +
        0.44f * input[7];
    normalize_vector(&expected[0], &expected[1]);
}

/**
 * @brief Mean squared error for logging.
 */
static float compute_loss(const float* output, const float* expected, size_t size) {
    float loss = 0.0f;
    size_t index;

    for (index = 0U; index < size; ++index) {
        float diff = output[index] - expected[index];
        loss += diff * diff;
    }
    return loss / (float)size;
}

int main(void) {
    const char* output_file = "../../data/weights.bin";
    void* infer_ctx;
    void* train_ctx;
    float input[HYBRID_ROUTE_INPUT_SIZE];
    float expected[HYBRID_ROUTE_OUTPUT_SIZE];
    float output[HYBRID_ROUTE_OUTPUT_SIZE];
    int epoch;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Hybrid Route Training ===\n");
    printf("Scene: a robot receives 4 recent local route cues from an upstream planner.\n");
    printf("Goal: keep following the intended branch instead of reacting to only one frame.\n");
    printf("Teaching rule: weighted average of 4 cues, with stronger weight on later cues.\n");
    printf("Leaf graph: transformer(sequence_encoder) -> mlp(decision_head)\n\n");

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        return 1;
    }

    train_ctx = train_create(infer_ctx);
    if (train_ctx == NULL) {
        fprintf(stderr, "Failed to create training context\n");
        infer_destroy(infer_ctx);
        return 1;
    }

    for (epoch = 0; epoch < HYBRID_ROUTE_EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        size_t sample_count = 0U;
        size_t a0;

        for (a0 = 0U; a0 < sizeof(g_values) / sizeof(g_values[0]); ++a0) {
            size_t a1;
            for (a1 = 0U; a1 < sizeof(g_values) / sizeof(g_values[0]); ++a1) {
                size_t a2;
                for (a2 = 0U; a2 < sizeof(g_values) / sizeof(g_values[0]); ++a2) {
                    size_t a3;
                    for (a3 = 0U; a3 < sizeof(g_values) / sizeof(g_values[0]); ++a3) {
                        size_t a4;
                        for (a4 = 0U; a4 < sizeof(g_values) / sizeof(g_values[0]); ++a4) {
                            size_t a5;
                            for (a5 = 0U; a5 < sizeof(g_values) / sizeof(g_values[0]); ++a5) {
                                size_t a6;
                                for (a6 = 0U; a6 < sizeof(g_values) / sizeof(g_values[0]); ++a6) {
                                    size_t a7;
                                    for (a7 = 0U; a7 < sizeof(g_values) / sizeof(g_values[0]); ++a7) {
                                        input[0] = g_values[a0];
                                        input[1] = g_values[a1];
                                        input[2] = g_values[a2];
                                        input[3] = g_values[a3];
                                        input[4] = g_values[a4];
                                        input[5] = g_values[a5];
                                        input[6] = g_values[a6];
                                        input[7] = g_values[a7];
                                        build_expected(input, expected);

                                        if (train_step(train_ctx, input, expected) != 0) {
                                            fprintf(stderr, "Training step failed at epoch %d\n", epoch + 1);
                                            train_destroy(train_ctx);
                                            infer_destroy(infer_ctx);
                                            return 1;
                                        }
                                        if (infer_auto_run(infer_ctx, input, output) != 0) {
                                            fprintf(stderr, "Inference check failed at epoch %d\n", epoch + 1);
                                            train_destroy(train_ctx);
                                            infer_destroy(infer_ctx);
                                            return 1;
                                        }
                                        epoch_loss += compute_loss(output, expected, HYBRID_ROUTE_OUTPUT_SIZE);
                                        sample_count += 1U;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (((epoch + 1) % 8) == 0 || epoch == 0 || epoch == (HYBRID_ROUTE_EPOCHS - 1)) {
            printf(
                "Epoch %d/%d - dataset loss: %.4f - trainer loss: %.4f\n",
                epoch + 1,
                HYBRID_ROUTE_EPOCHS,
                epoch_loss / (float)sample_count,
                train_get_loss(train_ctx)
            );
        }
    }

    printf("\nTraining completed.\n");
    printf("Average loss: %.4f\n", train_get_loss(train_ctx));
    if (weights_save_to_file(infer_ctx, output_file) != 0) {
        fprintf(stderr, "Failed to save weights to %s\n", output_file);
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        return 1;
    }
    printf("Weights saved to: %s\n", output_file);

    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    return 0;
}
