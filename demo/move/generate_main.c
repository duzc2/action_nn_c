/**
 * @file generate_main.c
 * @brief Move Demo code generation entry
 */

#include "profiler.h"

#include <stdio.h>

int main(void) {
    ProfilerGenerateRequest req;

    req.network_name = "move";
    req.network_type = "mlp";
    req.output_dir = "../../data";

    int rc = profiler_generate(&req);
    if (rc != 0) {
        fprintf(stderr, "profiler_generate failed: %d\n", rc);
        return 1;
    }

    printf("move spec generated\n");
    return 0;
}
