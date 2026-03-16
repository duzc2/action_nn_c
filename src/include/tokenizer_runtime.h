#ifndef TOKENIZER_RUNTIME_H
#define TOKENIZER_RUNTIME_H

#include <stddef.h>

#include "tokenizer.h"

/**
 * @brief 运行时文本编码入口。
 *
 * 设计目的：
 * - 对 tokenizer_encode 做轻量封装，统一运行时模块对编码接口的依赖点。
 * 关键保护点：
 * - 由实现层先校验 tokenizer 与词表指针，避免空指针进入底层编码器。
 */
int tokenizer_runtime_encode_text(Tokenizer* tokenizer,
                                  const char* text,
                                  int* out_ids,
                                  size_t out_capacity,
                                  size_t* out_count);

#endif
