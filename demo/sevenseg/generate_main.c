#include "profiler.h"

#include <stdio.h>

int main(void) {
    ProfilerGenerateRequest req;
    int rc = 0;
    req.network_name = "sevenseg";
    req.network_type = "mlp";
    req.output_dir = "demo/sevenseg/data";
    rc = profiler_generate(&req);
    if (rc != 0) {
        fprintf(stderr, "generate sevenseg spec failed: %d\n", rc);
        return 1;
    }
    printf("sevenseg spec generated\n");
    return 0;
}
