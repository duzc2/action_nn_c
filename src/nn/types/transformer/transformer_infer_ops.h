#ifndef TRANSFORMER_INFER_OPS_H
#define TRANSFORMER_INFER_OPS_H

#include <stddef.h>

typedef struct {
    const char* question;
    char* answer;
    size_t answer_capacity;
} TransformerInferContext;

int nn_transformer_infer_step(void* context);

#endif
