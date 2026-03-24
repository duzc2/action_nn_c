/**
 * @file train_main.c
 * @brief Transformer demo training entry based on generated APIs
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>

/**
 * @brief Dialogue pair definition
 */
typedef struct {
    const char* input;
    const char* expected_output;
} DialoguePair;

static const DialoguePair DIALOGUE_PAIRS[] = {
    {"hello", "Hello. Nice to meet you."},
    {"hi", "Hi there!"},
    {"how are you", "I'm doing well, thank you!"},
    {"what is your name", "I'm a small language model."},
    {"bye", "Goodbye! Have a nice day!"},
    {"yes", "I understand."},
    {"no", "I see."},
    {"thank you", "You're welcome!"},
    {"sorry", "It's okay, no problem."},
    {"help", "How can I assist you today?"}
};
static const int DIALOGUE_COUNT = sizeof(DIALOGUE_PAIRS) / sizeof(DIALOGUE_PAIRS[0]);

int main(void) {
    const char* output_file = "../../data/weights.bin";
    void* infer_ctx;
    void* train_ctx;
    int save_rc;
    int epoch;
    int i;
    int total_epochs = 5;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== Transformer Network Training ===\n");
    printf("Dialogue pairs: %d\n\n", DIALOGUE_COUNT);

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

    for (epoch = 0; epoch < total_epochs; epoch++) {
        printf("Epoch %d/%d\n", epoch + 1, total_epochs);

        for (i = 0; i < DIALOGUE_COUNT; i++) {
            const char* input = DIALOGUE_PAIRS[i].input;
            const char* expected = DIALOGUE_PAIRS[i].expected_output;
            char output[256];

            train_step(train_ctx, input, expected);
            if (infer_auto_run(infer_ctx, input, output) != 0) {
                fprintf(stderr, "inference failed during training\n");
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                return 1;
            }

            if (epoch == 0 && (i % 3) == 0) {
                printf("  sample: \"%s\" -> \"%s\" (target: \"%s\")\n", input, output, expected);
            }
        }

        printf("  average loss: %.4f\n", train_get_loss(train_ctx));
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
