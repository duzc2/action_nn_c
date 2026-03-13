#include "../include/tokenizer_runtime.h"

int tokenizer_runtime_encode_text(Tokenizer* tokenizer,
                                  const char* text,
                                  int* out_ids,
                                  size_t out_capacity,
                                  size_t* out_count) {
    if (tokenizer == NULL || tokenizer->vocab == NULL) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    return tokenizer_encode(tokenizer, text, out_ids, out_capacity, out_count);
}
