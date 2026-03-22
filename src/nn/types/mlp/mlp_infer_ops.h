#ifndef MLP_INFER_OPS_H
#define MLP_INFER_OPS_H

#include <stddef.h>

typedef struct {
    int input_x;
    int input_y;
    int command;
    int output_x;
    int output_y;
} MLPInferContext;

int nn_mlp_infer_step(void* context);

#endif
