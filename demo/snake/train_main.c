/**
 * @file train_main.c
 * @brief Snake Demo training entry
 *
 * Collects teacher game samples, shuffles them, and trains the MLP.
 * Follows the cnn_rnn_react/train_main.c pattern.
 *
 * Network: 8 -> [6] -> 4 (softmax), ADAM, MSE loss
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "../demo_runtime_paths.h"
#include "snake_scene.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SNAKE_TEACHER_GAMES    1000
#define SNAKE_TRAIN_EPOCHS     80
#define SNAKE_MAX_SAMPLES      (SNAKE_TEACHER_GAMES * SNAKE_MAX_STEPS)

typedef struct {
    float features[SNAKE_INPUT_SIZE];
    float target[SNAKE_OUTPUT_SIZE];
} SnakeSample;

/* Fisher-Yates shuffle */
static void snake_shuffle_samples(SnakeSample* samples, size_t count) {
    size_t i;
    for (i = count - 1; i > 0; i--) {
        size_t j = (size_t)rand() % (i + 1);
        SnakeSample tmp = samples[i];
        samples[i] = samples[j];
        samples[j] = tmp;
    }
}

/* Mean squared error for reporting */
static float snake_compute_loss(const float* output, const float* expected, size_t size) {
    float loss = 0.0f;
    size_t i;
    for (i = 0; i < size; ++i) {
        float diff = output[i] - expected[i];
        loss += diff * diff;
    }
    return loss / (float)size;
}

int main(void) {
    const char* output_file = "../../data/weights.bin";
    void* infer_ctx;
    void* train_ctx;
    SnakeSample* samples;
    size_t sample_count = 0;
    size_t alloc_count;
    int save_rc;
    int epoch;
    int game;
    float output[SNAKE_OUTPUT_SIZE];
    unsigned int rng = (unsigned int)time(NULL);

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Snake Network Training ===\n");
    printf("Network: 8 -> [6] -> 4 (softmax)\n");
    printf("Collecting teacher samples from %d games...\n", SNAKE_TEACHER_GAMES);

    /* Allocate sample buffer */
    alloc_count = SNAKE_MAX_SAMPLES;
    samples = (SnakeSample*)malloc(alloc_count * sizeof(SnakeSample));
    if (samples == NULL) {
        fprintf(stderr, "Failed to allocate sample buffer\n");
        return 1;
    }

    /* Play teacher games to collect samples */
    for (game = 0; game < SNAKE_TEACHER_GAMES; game++) {
        SnakeState state;
        unsigned int game_rng = rng + (unsigned int)(game * 2654435761U);

        snake_init_random(&state, &game_rng);

        while (!state.game_over && sample_count < alloc_count) {
            SnakeSample* s = &samples[sample_count];
            snake_encode_state(&state, s->features);
            snake_teacher_target(&state, s->target);

            /* Advance by teacher action */
            SnakeTurn turn = snake_teacher_turn(&state);
            SnakeDirection dir = snake_turn_to_absolute(state.direction, turn);
            snake_step(&state, dir);

            sample_count++;
        }
    }

    printf("Collected %zu training samples.\n", sample_count);
    printf("Shuffling...\n");

    srand(42);
    snake_shuffle_samples(samples, sample_count);

    /* Create inference and training contexts */
    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        free(samples);
        return 1;
    }

    train_ctx = train_create(infer_ctx);
    if (train_ctx == NULL) {
        fprintf(stderr, "Failed to create training context\n");
        infer_destroy(infer_ctx);
        free(samples);
        return 1;
    }

    printf("\nTraining for %d epochs...\n\n", SNAKE_TRAIN_EPOCHS);

    for (epoch = 0; epoch < SNAKE_TRAIN_EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        size_t i;

        for (i = 0; i < sample_count; ++i) {
            if (train_step(train_ctx, samples[i].features, samples[i].target) != 0) {
                fprintf(stderr, "Training step failed at epoch %d, sample %zu\n", epoch + 1, i);
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                free(samples);
                return 1;
            }

            if (infer_auto_run(infer_ctx, samples[i].features, output) != 0) {
                fprintf(stderr, "Inference check failed at epoch %d, sample %zu\n", epoch + 1, i);
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                free(samples);
                return 1;
            }

            epoch_loss += snake_compute_loss(output, samples[i].target, SNAKE_OUTPUT_SIZE);
        }

        if (((epoch + 1) % 10) == 0 || epoch == 0 || epoch == (SNAKE_TRAIN_EPOCHS - 1)) {
            printf("Epoch %d/%d - loss: %.4f - trainer loss: %.4f\n",
                   epoch + 1, SNAKE_TRAIN_EPOCHS,
                   epoch_loss / (float)sample_count,
                   train_get_loss(train_ctx));
        }
    }

    printf("\n=== Training Complete ===\n");
    printf("Final trainer loss: %.4f\n", train_get_loss(train_ctx));

    save_rc = weights_save_to_file(infer_ctx, output_file);
    if (save_rc != 0) {
        fprintf(stderr, "Failed to save weights to %s (rc=%d)\n", output_file, save_rc);
        train_destroy(train_ctx);
        infer_destroy(infer_ctx);
        free(samples);
        return 1;
    }
    printf("Weights saved to: %s\n", output_file);

    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    free(samples);
    return 0;
}
