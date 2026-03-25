#include "infer.h"
#include "weights_load.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>

static void trim_line_endings(char* text) {
    size_t length;

    if (text == NULL) {
        return;
    }

    length = strlen(text);
    while (length > 0U &&
           (text[length - 1U] == '\n' || text[length - 1U] == '\r')) {
        text[length - 1U] = '\0';
        length -= 1U;
    }
}

static char* trim_surrounding_spaces(char* text) {
    char* start = text;
    char* end;

    if (text == NULL) {
        return NULL;
    }

    while (*start != '\0' &&
           (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n')) {
        start += 1;
    }

    end = start + strlen(start);
    while (end > start &&
           (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) {
        end -= 1;
    }
    *end = '\0';

    return start;
}

int main(void) {
    char line[256];
    char ans[256];
    char* question;
    const char* weights_file = "../../data/weights.bin";
    void* infer_ctx;

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "failed to switch working directory to executable directory\n");
        return 1;
    }

    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "failed to create inference context\n");
        return 1;
    }

    if (weights_load_from_file(infer_ctx, weights_file) != 0) {
        fprintf(stderr, "failed to load weights from %s\n", weights_file);
        infer_destroy(infer_ctx);
        return 1;
    }

    printf("tiny transformer chat, input quit to exit\n");
    while (fgets(line, sizeof(line), stdin) != NULL) {
        trim_line_endings(line);
        question = trim_surrounding_spaces(line);
        if (question == NULL || question[0] == '\0') {
            continue;
        }
        if (strcmp(question, "quit") == 0) {
            break;
        }
        if (infer_auto_run(infer_ctx, question, ans) != 0) {
            printf("infer failed\n");
            continue;
        }
        printf("%s\n", ans);
    }

    infer_destroy(infer_ctx);
    return 0;
}
