#include "infer_runtime.h"
#include "transformer_infer_ops.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    char line[256];
    char ans[256];
    NNInferRequest request;
    TransformerInferContext infer_ctx;
    printf("tiny transformer chat, input quit to exit\n");
    while (fgets(line, sizeof(line), stdin) != NULL) {
        size_t len = strlen(line);
        if (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[len - 1] = '\0';
        }
        if (strcmp(line, "quit") == 0) {
            break;
        }
        request.network_type = "transformer";
        request.context = &infer_ctx;
        infer_ctx.question = line;
        infer_ctx.answer = ans;
        infer_ctx.answer_capacity = sizeof(ans);
        if (nn_infer_runtime_step(&request) != 0) {
            printf("infer failed\n");
            continue;
        }
        printf("%s\n", ans);
    }
    return 0;
}
