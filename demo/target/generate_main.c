#include "profiler.h"

#include <stdio.h>

int main(void) {
    ProfilerGenerateRequest req;
    int rc = 0;
    req.network_name = "target";
    req.network_type = "mlp";
    req.output_dir = "demo/target/data";
    rc = profiler_generate(&req);
    if (rc != 0) {
        fprintf(stderr, "generate target spec failed: %d\n", rc);
        return 1;
    }
    printf("target spec generated\n");
    return 0;
}
