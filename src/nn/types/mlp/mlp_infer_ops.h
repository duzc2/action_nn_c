#ifndef MLP_INFER_OPS_H
#define MLP_INFER_OPS_H

#include <stddef.h>
#include <stdio.h>

typedef struct {
    int input_x;
    int input_y;
    int command;
    int output_x;
    int output_y;
} MlpInferContext;

int nn_mlp_infer_step(void* context);

int nn_mlp_load_weights(void* context, FILE* fp);
int nn_mlp_save_weights(void* context, FILE* fp);

#endif
