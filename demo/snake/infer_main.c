/**
 * @file infer_main.c
 * @brief Snake Demo inference entry (playable TUI)
 *
 * Loads trained weights, initializes a random snake game,
 * and runs the neural network inference loop with rendering.
 * Follows road_graph_nav/infer_main.c pattern.
 */

#include "infer.h"
#include "weights_load.h"
#include "../demo_runtime_paths.h"
#include "snake_scene.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Find argmax index in output array */
static int snake_argmax(const float* output, size_t size) {
    size_t i;
    int best = 0;
    float best_val = output[0];
    for (i = 1; i < size; ++i) {
        if (output[i] > best_val) {
            best_val = output[i];
            best = (int)i;
        }
    }
    return best;
}

int main(void) {
    void* infer_ctx;
    SnakeState state;
    float features[SNAKE_INPUT_SIZE];
    float nn_output[SNAKE_OUTPUT_SIZE];
    unsigned int rng = (unsigned int)time(NULL);
    const char* weights_file = "../../data/weights.bin";

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory\n");
        return 1;
    }

    /* Load weights */
    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        return 1;
    }

    if (weights_load_from_file(infer_ctx, weights_file) != 0) {
        fprintf(stderr, "Failed to load weights from %s\n", weights_file);
        fprintf(stderr, "Run snake_train first to generate weights.bin\n");
        infer_destroy(infer_ctx);
        return 1;
    }

    printf("Weights loaded from %s\n", weights_file);

    /* Initialize random game */
    snake_init_random(&state, &rng);

    printf("=== Snake TUI Demo ===\n");
    printf("Controls: NN plays autonomously\n");
    printf("Press Ctrl+C to exit\n");
    snake_sleep_ms(1000);

    /* Main loop */
    while (!state.game_over) {
        /* Encode state */
        snake_encode_state(&state, features);

        /* Run inference */
        if (infer_auto_run(infer_ctx, features, nn_output) != 0) {
            fprintf(stderr, "Inference failed\n");
            break;
        }

        /* Get action from argmax */
        SnakeDirection action = (SnakeDirection)snake_argmax(nn_output, SNAKE_OUTPUT_SIZE);

        /* Render */
        snake_render(&state, nn_output);

        /* Step game */
        snake_step(&state, action);

        /* Sleep for smooth animation */
        snake_sleep_ms(150);
    }

    /* Final render */
    snake_encode_state(&state, features);
    infer_auto_run(infer_ctx, features, nn_output);
    snake_render(&state, nn_output);

    printf("\nFinal score: %d\n", state.score);

    infer_destroy(infer_ctx);
    return 0;
}
