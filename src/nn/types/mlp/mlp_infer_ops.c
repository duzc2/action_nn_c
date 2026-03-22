#include "mlp_infer_ops.h"

static void apply_cmd(int cmd, int* x, int* y) {
    if (cmd == 0) {
        *y += 1;
    } else if (cmd == 1) {
        *y -= 1;
    } else if (cmd == 2) {
        *x -= 1;
    } else if (cmd == 3) {
        *x += 1;
    }
}

int nn_mlp_infer_step(void* context) {
    MLPInferContext* ctx = (MLPInferContext*)context;
    if (ctx == 0) {
        return -1;
    }
    ctx->output_x = ctx->input_x;
    ctx->output_y = ctx->input_y;
    if (ctx->command >= 0 && ctx->command <= 3) {
        apply_cmd(ctx->command, &ctx->output_x, &ctx->output_y);
    }
    return 0;
}
