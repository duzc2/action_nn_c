/**
 * @file infer_main.c
 * @brief SevenSeg Demo inference entry
 *
 * Demonstrates usage of profiler-generated inference API.
 * This demo ONLY includes the generated infer.h header,
 * NOT any src/nn implementation headers.
 */

#include "infer.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Render 7-segment display pattern
 */
static void render_sevenseg(const float* output) {
    int seg = (output[0] > 0.5f);
    int sea = (output[1] > 0.5f);
    int seb = (output[2] > 0.5f);
    int sec = (output[3] > 0.5f);
    int sed = (output[4] > 0.5f);
    int see = (output[5] > 0.5f);
    int sef = (output[6] > 0.5f);

    printf(" %s \n", seg ? "---" : "   ");
    printf("%c   %c\n", sef ? '|' : ' ', sea ? '|' : ' ');
    printf(" %s \n", seb ? "---" : "   ");
    printf("%c   %c\n", see ? '|' : ' ', sec ? '|' : ' ');
    printf(" %s \n", sed ? "---" : "   ");
}

/**
 * @brief Encode digit as one-hot vector
 */
static void encode_digit(int digit, float* input) {
    size_t i;
    for (i = 0; i < 10; i++) {
        input[i] = (i == (size_t)digit) ? 1.0f : 0.0f;
    }
}

int main(void) {
    void* infer_ctx;
    float input[10];
    float output[7];
    int ch;
    int digit;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("=== SevenSeg Network Inference ===\n\n");

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        return 1;
    }

    printf("Enter digit (0-9) to see prediction, 'q' to quit:\n\n");

    while ((ch = getchar()) != EOF) {
        if (ch == 'q' || ch == 'Q') break;
        if (ch < '0' || ch > '9') {
            if (ch != '\n' && ch != '\r') printf("Invalid input.\n");
            continue;
        }
        digit = ch - '0';
        encode_digit(digit, input);

        infer_auto_run(infer_ctx, input, output);

        printf("Digit: %d -> [%.2f %.2f %.2f %.2f %.2f %.2f %.2f]\n",
               digit, output[0], output[1], output[2], output[3],
               output[4], output[5], output[6]);
        render_sevenseg(output);
        printf("\n");
    }

    infer_destroy(infer_ctx);
    return 0;
}
