#ifndef TOKENIZER_RUNTIME_H
#define TOKENIZER_RUNTIME_H

#include <stddef.h>

#include "tokenizer.h"

int tokenizer_runtime_encode_text(Tokenizer* tokenizer,
                                  const char* text,
                                  int* out_ids,
                                  size_t out_capacity,
                                  size_t* out_count);

#endif
