#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "../include/tokenizer.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief 词表二进制文件头，用于校验格式版本和快速拒绝非法输入。
 */
typedef struct VocabBinaryHeader {
    unsigned char magic[4];
    uint32_t version;
    uint32_t token_count;
} VocabBinaryHeader;

/**
 * @brief 将 C 字符串复制到新分配内存中。
 *
 * 关键保护点：
 * - C99 不保证存在 strdup，本函数提供可移植替代。
 * - 使用显式长度与 memcpy，避免重复扫描字符串。
 *
 * @param src 输入字符串
 * @return char* 成功返回新字符串，失败返回 NULL
 */
static char* tokenizer_strdup(const char* src) {
    size_t len = 0U;
    char* dst = NULL;
    if (src == NULL) {
        return NULL;
    }
    len = strlen(src);
    dst = (char*)malloc(len + 1U);
    if (dst == NULL) {
        return NULL;
    }
    memcpy(dst, src, len + 1U);
    return dst;
}

/**
 * @brief 判断字符是否为空白分隔符。
 *
 * @param ch 输入字符
 * @return int 1=空白，0=非空白
 */
static int tokenizer_is_space(char ch) {
    return (isspace((unsigned char)ch) != 0) ? 1 : 0;
}

/**
 * @brief 初始化词表。
 *
 * @param vocab      词表对象
 * @param capacity   最大 token 数
 * @return int       TokenizerStatus
 */
int vocab_init(Vocabulary* vocab, size_t capacity) {
    if (vocab == NULL || capacity == 0U) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    vocab->tokens = (char**)calloc(capacity, sizeof(char*));
    if (vocab->tokens == NULL) {
        return TOKENIZER_STATUS_OOM;
    }
    vocab->size = 0U;
    vocab->capacity = capacity;
    return TOKENIZER_STATUS_OK;
}

/**
 * @brief 释放词表资源。
 *
 * @param vocab 词表对象
 */
void vocab_free(Vocabulary* vocab) {
    size_t i = 0U;
    if (vocab == NULL) {
        return;
    }
    if (vocab->tokens != NULL) {
        for (i = 0U; i < vocab->size; ++i) {
            free(vocab->tokens[i]);
            vocab->tokens[i] = NULL;
        }
        free(vocab->tokens);
    }
    vocab->tokens = NULL;
    vocab->size = 0U;
    vocab->capacity = 0U;
}

/**
 * @brief 查询 token 对应的 id。
 *
 * @param vocab    词表对象
 * @param token    token 字符串
 * @param out_id   输出 token id
 * @return int     TokenizerStatus
 */
int vocab_find_id(const Vocabulary* vocab, const char* token, int* out_id) {
    size_t i = 0U;
    if (vocab == NULL || token == NULL || out_id == NULL) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0U; i < vocab->size; ++i) {
        if (vocab->tokens[i] != NULL && strcmp(vocab->tokens[i], token) == 0) {
            *out_id = (int)i;
            return TOKENIZER_STATUS_OK;
        }
    }
    return TOKENIZER_STATUS_NOT_FOUND;
}

/**
 * @brief 向词表新增 token；若 token 已存在，返回已有 id。
 *
 * @param vocab    词表对象
 * @param token    token 字符串
 * @param out_id   输出 token id（可为 NULL）
 * @return int     TokenizerStatus
 */
int vocab_add_token(Vocabulary* vocab, const char* token, int* out_id) {
    int id = -1;
    int rc = 0;
    char* copied = NULL;
    if (vocab == NULL || token == NULL || token[0] == '\0') {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    rc = vocab_find_id(vocab, token, &id);
    if (rc == TOKENIZER_STATUS_OK) {
        if (out_id != NULL) {
            *out_id = id;
        }
        return TOKENIZER_STATUS_OK;
    }
    if (rc != TOKENIZER_STATUS_NOT_FOUND) {
        return rc;
    }
    if (vocab->size >= vocab->capacity) {
        return TOKENIZER_STATUS_BUFFER_TOO_SMALL;
    }
    copied = tokenizer_strdup(token);
    if (copied == NULL) {
        return TOKENIZER_STATUS_OOM;
    }
    vocab->tokens[vocab->size] = copied;
    if (out_id != NULL) {
        *out_id = (int)vocab->size;
    }
    vocab->size += 1U;
    return TOKENIZER_STATUS_OK;
}

/**
 * @brief 根据 id 查询 token 字符串。
 *
 * @param vocab  词表对象
 * @param id     token id
 * @return const char* 成功返回 token 指针；失败返回 NULL
 */
const char* vocab_get_token(const Vocabulary* vocab, int id) {
    if (vocab == NULL || id < 0) {
        return NULL;
    }
    if ((size_t)id >= vocab->size) {
        return NULL;
    }
    return vocab->tokens[id];
}

/**
 * @brief 将词表保存为二进制文件。
 *
 * 二进制布局：
 * - Header: magic("VOCB"), version(1), token_count(uint32)
 * - Repeated:
 *   - token_len(uint32)
 *   - token_bytes[ token_len ]（不含 '\0'）
 *
 * @param vocab       词表对象
 * @param file_path   输出文件路径
 * @return int        TokenizerStatus
 */
int vocab_save_binary(const Vocabulary* vocab, const char* file_path) {
    FILE* fp = NULL;
    VocabBinaryHeader header;
    size_t i = 0U;
    if (vocab == NULL || file_path == NULL) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    if (vocab->size > (size_t)UINT32_MAX) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    fp = fopen(file_path, "wb");
    if (fp == NULL) {
        return TOKENIZER_STATUS_IO_ERROR;
    }
    header.magic[0] = (unsigned char)'V';
    header.magic[1] = (unsigned char)'O';
    header.magic[2] = (unsigned char)'C';
    header.magic[3] = (unsigned char)'B';
    header.version = (uint32_t)1U;
    header.token_count = (uint32_t)vocab->size;

    if (fwrite(&header, sizeof(header), 1U, fp) != 1U) {
        fclose(fp);
        return TOKENIZER_STATUS_IO_ERROR;
    }
    for (i = 0U; i < vocab->size; ++i) {
        uint32_t token_len = 0U;
        const char* token = vocab->tokens[i];
        if (token == NULL) {
            fclose(fp);
            return TOKENIZER_STATUS_FORMAT_ERROR;
        }
        if (strlen(token) > (size_t)UINT32_MAX) {
            fclose(fp);
            return TOKENIZER_STATUS_INVALID_ARGUMENT;
        }
        token_len = (uint32_t)strlen(token);
        if (fwrite(&token_len, sizeof(token_len), 1U, fp) != 1U) {
            fclose(fp);
            return TOKENIZER_STATUS_IO_ERROR;
        }
        if (token_len > 0U && fwrite(token, 1U, token_len, fp) != token_len) {
            fclose(fp);
            return TOKENIZER_STATUS_IO_ERROR;
        }
    }
    fclose(fp);
    return TOKENIZER_STATUS_OK;
}

/**
 * @brief 从二进制文件加载词表。
 *
 * @param file_path   输入文件路径
 * @param out_vocab   输出词表对象（函数内部会初始化）
 * @return int        TokenizerStatus
 */
int vocab_load_binary(const char* file_path, Vocabulary* out_vocab) {
    FILE* fp = NULL;
    VocabBinaryHeader header;
    uint32_t i = 0U;
    int rc = 0;
    if (file_path == NULL || out_vocab == NULL) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    memset(out_vocab, 0, sizeof(*out_vocab));
    fp = fopen(file_path, "rb");
    if (fp == NULL) {
        return TOKENIZER_STATUS_IO_ERROR;
    }
    if (fread(&header, sizeof(header), 1U, fp) != 1U) {
        fclose(fp);
        return TOKENIZER_STATUS_IO_ERROR;
    }
    if (header.magic[0] != (unsigned char)'V' ||
        header.magic[1] != (unsigned char)'O' ||
        header.magic[2] != (unsigned char)'C' ||
        header.magic[3] != (unsigned char)'B' ||
        header.version != (uint32_t)1U) {
        fclose(fp);
        return TOKENIZER_STATUS_FORMAT_ERROR;
    }
    rc = vocab_init(out_vocab, (size_t)header.token_count);
    if (rc != TOKENIZER_STATUS_OK) {
        fclose(fp);
        return rc;
    }

    for (i = 0U; i < header.token_count; ++i) {
        uint32_t token_len = 0U;
        char* token = NULL;
        if (fread(&token_len, sizeof(token_len), 1U, fp) != 1U) {
            vocab_free(out_vocab);
            fclose(fp);
            return TOKENIZER_STATUS_IO_ERROR;
        }
        token = (char*)malloc((size_t)token_len + 1U);
        if (token == NULL) {
            vocab_free(out_vocab);
            fclose(fp);
            return TOKENIZER_STATUS_OOM;
        }
        if (token_len > 0U && fread(token, 1U, token_len, fp) != token_len) {
            free(token);
            vocab_free(out_vocab);
            fclose(fp);
            return TOKENIZER_STATUS_IO_ERROR;
        }
        token[token_len] = '\0';
        out_vocab->tokens[out_vocab->size] = token;
        out_vocab->size += 1U;
    }
    fclose(fp);
    return TOKENIZER_STATUS_OK;
}

int vocab_load_text(const char* file_path, Vocabulary* out_vocab) {
    FILE* fp = NULL;
    char line[512];
    int rc = 0;
    size_t line_count = 0U;
    if (file_path == NULL || out_vocab == NULL) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    memset(out_vocab, 0, sizeof(*out_vocab));
    fp = fopen(file_path, "r");
    if (fp == NULL) {
        return TOKENIZER_STATUS_IO_ERROR;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t n = strlen(line);
        while (n > 0U && (line[n - 1U] == '\n' || line[n - 1U] == '\r')) {
            line[--n] = '\0';
        }
        if (n > 0U) {
            line_count += 1U;
        }
    }
    if (line_count == 0U) {
        fclose(fp);
        return TOKENIZER_STATUS_FORMAT_ERROR;
    }
    if (fseek(fp, 0L, SEEK_SET) != 0) {
        fclose(fp);
        return TOKENIZER_STATUS_IO_ERROR;
    }
    rc = vocab_init(out_vocab, line_count);
    if (rc != TOKENIZER_STATUS_OK) {
        fclose(fp);
        return rc;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t n = strlen(line);
        while (n > 0U && (line[n - 1U] == '\n' || line[n - 1U] == '\r')) {
            line[--n] = '\0';
        }
        if (n == 0U) {
            continue;
        }
        rc = vocab_add_token(out_vocab, line, NULL);
        if (rc != TOKENIZER_STATUS_OK) {
            vocab_free(out_vocab);
            fclose(fp);
            return rc;
        }
    }
    fclose(fp);
    return TOKENIZER_STATUS_OK;
}

/**
 * @brief 初始化 Tokenizer。
 *
 * @param tokenizer  Tokenizer 对象
 * @param vocab      词表对象
 * @param unk_id     未登录词 id
 * @return int       TokenizerStatus
 */
int tokenizer_init(Tokenizer* tokenizer, const Vocabulary* vocab, int unk_id) {
    if (tokenizer == NULL || vocab == NULL || unk_id < 0 || (size_t)unk_id >= vocab->size) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    tokenizer->vocab = vocab;
    tokenizer->unk_id = unk_id;
    return TOKENIZER_STATUS_OK;
}

/**
 * @brief 文本编码：按空白分词并映射为 token id。
 *
 * 关键算法说明：
 * - 采用线性扫描，从非空白段提取 token。
 * - 每个 token 先查词表，未命中则写入 unk_id。
 * - 通过 out_capacity 限制输出，防止缓冲区溢出。
 *
 * @param tokenizer      Tokenizer 对象
 * @param text           输入文本
 * @param out_ids        输出 id 数组
 * @param out_capacity   输出数组容量（元素数）
 * @param out_count      实际输出数量
 * @return int           TokenizerStatus
 */
int tokenizer_encode(const Tokenizer* tokenizer,
                     const char* text,
                     int* out_ids,
                     size_t out_capacity,
                     size_t* out_count) {
    size_t pos = 0U;
    size_t written = 0U;
    if (tokenizer == NULL || tokenizer->vocab == NULL || text == NULL || out_count == NULL) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    *out_count = 0U;
    while (text[pos] != '\0') {
        size_t start = 0U;
        size_t end = 0U;
        int id = -1;
        size_t token_len = 0U;
        char* token = NULL;

        while (text[pos] != '\0' && tokenizer_is_space(text[pos])) {
            pos += 1U;
        }
        if (text[pos] == '\0') {
            break;
        }

        if (written >= out_capacity || out_ids == NULL) {
            return TOKENIZER_STATUS_BUFFER_TOO_SMALL;
        }

        start = pos;
        while (text[pos] != '\0' && !tokenizer_is_space(text[pos])) {
            pos += 1U;
        }
        end = pos;
        token_len = end - start;
        token = (char*)malloc(token_len + 1U);
        if (token == NULL) {
            return TOKENIZER_STATUS_OOM;
        }
        memcpy(token, text + start, token_len);
        token[token_len] = '\0';

        if (vocab_find_id(tokenizer->vocab, token, &id) != TOKENIZER_STATUS_OK) {
            id = tokenizer->unk_id;
        }
        free(token);
        out_ids[written] = id;
        written += 1U;
    }
    *out_count = written;
    return TOKENIZER_STATUS_OK;
}

/**
 * @brief token id 解码：将 id 序列拼接为以空格分隔的文本。
 *
 * @param vocab          词表对象
 * @param ids            输入 id 数组
 * @param id_count       id 数量
 * @param out_text       输出文本缓冲区
 * @param out_capacity   输出缓冲区容量（字节）
 * @return int           TokenizerStatus
 */
int tokenizer_decode(const Vocabulary* vocab,
                     const int* ids,
                     size_t id_count,
                     char* out_text,
                     size_t out_capacity) {
    size_t i = 0U;
    size_t used = 0U;
    if (vocab == NULL || ids == NULL || out_text == NULL || out_capacity == 0U) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    out_text[0] = '\0';
    for (i = 0U; i < id_count; ++i) {
        const char* token = vocab_get_token(vocab, ids[i]);
        size_t token_len = 0U;
        if (token == NULL) {
            return TOKENIZER_STATUS_NOT_FOUND;
        }
        token_len = strlen(token);
        if (i > 0U) {
            if (used + 1U >= out_capacity) {
                return TOKENIZER_STATUS_BUFFER_TOO_SMALL;
            }
            out_text[used] = ' ';
            used += 1U;
        }
        if (used + token_len >= out_capacity) {
            return TOKENIZER_STATUS_BUFFER_TOO_SMALL;
        }
        memcpy(out_text + used, token, token_len);
        used += token_len;
        out_text[used] = '\0';
    }
    return TOKENIZER_STATUS_OK;
}
