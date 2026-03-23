#include "infer.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    char line[256];
    char ans[256];
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

    printf("tiny transformer chat, input quit to exit\n");
    while (fgets(line, sizeof(line), stdin) != NULL) {
        size_t len = strlen(line);
        if (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[len - 1] = '\0';
        }
        if (strcmp(line, "quit") == 0) {
            break;
        }
        if (infer_auto_run(infer_ctx, line, ans) != 0) {
            printf("infer failed\n");
            continue;
        }
        printf("%s\n", ans);
    }

    infer_destroy(infer_ctx);
    return 0;
}
