/**
 * @file generate_main.c
 * @brief Transformer Demo code generation entry
 */

#include "profiler.h"

#include <stdio.h>

int main(void) {
    ProfilerGenerateRequest req;
    int rc = 0;
    req.network_name = "transformer";
    req.network_type = "transformer";
    req.output_dir = "../../data";
    rc = profiler_generate(&req);
    if (rc != 0) {
        fprintf(stderr, "generate transformer spec failed: %d\n", rc);
        return 1;
    }
    printf("transformer spec generated\n");
    return 0;
}
