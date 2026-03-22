#ifndef TRANSFORMER_INFER_OPS_H
#define TRANSFORMER_INFER_OPS_H

#include <stddef.h>
#include <stdio.h>

typedef struct {
    const char* question;
    char* answer;
    size_t answer_capacity;
} TransformerInferContext;

int nn_transformer_infer_step(void* context);

int nn_transformer_load_weights(void* context, FILE* fp);
int nn_transformer_save_weights(void* context, FILE* fp);

#endif
