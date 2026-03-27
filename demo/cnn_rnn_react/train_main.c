/**
 * @file train_main.c
 * @brief Training entry for the pedestrian-crossing controller demo.
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "../demo_runtime_paths.h"
#include "cnn_rnn_react_scene.h"

#include <stdio.h>

#define CNN_RNN_REACT_EPOCHS 120
#define CNN_RNN_REACT_SAMPLES_PER_EPOCH 128U

/**
 * @brief Mean squared error used for logging.
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
    unsigned int rng_state = 0x43524E4EU;
    int epoch;
    int save_rc;
    float output[CNN_RNN_REACT_OUTPUT_SIZE];

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== CNN + RNN Pedestrian-Crossing Training ===\n");
    printf("Scene: cross the road, reach the opposite sidewalk, and avoid moving lane traffic.\n");
    printf("Input: four recent 30x20 full-map frames. Output: crossing direction + forward/wait decision.\n");
    printf("Outputs: turn_axis + move_axis, both are non-conflicting control dimensions.\n");
    printf("Sequence length: %u recent frames, frame size: %ux%u\n\n",
        (unsigned int)CNN_RNN_REACT_SEQUENCE_LENGTH,
        (unsigned int)CNN_RNN_REACT_FRAME_WIDTH,
        (unsigned int)CNN_RNN_REACT_FRAME_HEIGHT);

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "failed to create inference context\n");
        return 1;
    }

    train_ctx = train_create(infer_ctx);
    if (train_ctx == NULL) {
        fprintf(stderr, "failed to create training context\n");
        infer_destroy(infer_ctx);
        return 1;
    }

    for (epoch = 0; epoch < CNN_RNN_REACT_EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        size_t sample_index;

        for (sample_index = 0U; sample_index < CNN_RNN_REACT_SAMPLES_PER_EPOCH; ++sample_index) {
            CnnRnnReactSample sample;

            cnn_rnn_react_build_random_sample(&sample, &rng_state);

            if (train_step(train_ctx, sample.input, sample.target) != 0) {
                fprintf(stderr, "training step failed at epoch %d\n", epoch + 1);
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                return 1;
            }

            if (infer_auto_run(infer_ctx, sample.input, output) != 0) {
                fprintf(stderr, "inference verification failed at epoch %d\n", epoch + 1);
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                return 1;
            }

            epoch_loss += compute_loss(output, sample.target, CNN_RNN_REACT_OUTPUT_SIZE);
        }

        if (((epoch + 1) % 10) == 0 || epoch == 0 || epoch == (CNN_RNN_REACT_EPOCHS - 1)) {
            printf(
                "Epoch %d/%d - dataset loss: %.4f - trainer loss: %.4f\n",
                epoch + 1,
                CNN_RNN_REACT_EPOCHS,
                epoch_loss / (float)CNN_RNN_REACT_SAMPLES_PER_EPOCH,
                train_get_loss(train_ctx)
            );
        }
    }

    printf("\nTraining completed.\n");
    printf("Average loss: %.4f\n", train_get_loss(train_ctx));
    save_rc = weights_save_to_file(infer_ctx, output_file);
    if (save_rc != 0) {
        fprintf(stderr, "failed to save weights to %s (rc=%d)\n", output_file, save_rc);
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        return 1;
    }
    printf("Weights saved to: %s\n", output_file);

    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    return 0;
}
