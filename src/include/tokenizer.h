#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>

/**
 * @brief Tokenizer/词表模块统一错误码，便于上层做确定性错误处理。
 */
typedef enum TokenizerStatus {
    TOKENIZER_STATUS_OK = 0,
    TOKENIZER_STATUS_INVALID_ARGUMENT = -1,
    TOKENIZER_STATUS_OOM = -2,
    TOKENIZER_STATUS_NOT_FOUND = -3,
    TOKENIZER_STATUS_IO_ERROR = -4,
    TOKENIZER_STATUS_FORMAT_ERROR = -5,
    TOKENIZER_STATUS_BUFFER_TOO_SMALL = -6
} TokenizerStatus;

/**
 * @brief 词表对象，维护 token 字符串与 token id 的一一映射。
 *
 * 设计说明：
 * - 词表独立于模型参数，支持独立保存和加载。
 * - token 的 id 等于其在 tokens 数组中的索引，确保编码/解码一致。
 */
typedef struct Vocabulary {
    char** tokens;
    size_t size;
    size_t capacity;
} Vocabulary;

/**
 * @brief Tokenizer 对象，使用词表将文本转换为 token id 序列。
 *
 * 设计说明：
 * - 按空白字符分词（空格/制表/换行等）。
 * - 未登录词统一映射到 unk_id。
 */
typedef struct Tokenizer {
    const Vocabulary* vocab;
    int unk_id;
} Tokenizer;

/**
 * @brief 初始化词表。
 *
 * @param vocab      词表对象
 * @param capacity   最大 token 数
 * @return int       TokenizerStatus
 */
int vocab_init(Vocabulary* vocab, size_t capacity);

/**
 * @brief 释放词表资源。
 *
 * @param vocab 词表对象
 */
void vocab_free(Vocabulary* vocab);

/**
 * @brief 向词表新增 token；若 token 已存在，返回已有 id。
 *
 * @param vocab    词表对象
 * @param token    token 字符串
 * @param out_id   输出 token id（可为 NULL）
 * @return int     TokenizerStatus
 */
int vocab_add_token(Vocabulary* vocab, const char* token, int* out_id);

/**
 * @brief 查询 token 对应的 id。
 *
 * @param vocab    词表对象
 * @param token    token 字符串
 * @param out_id   输出 token id
 * @return int     TokenizerStatus
 */
int vocab_find_id(const Vocabulary* vocab, const char* token, int* out_id);

/**
 * @brief 根据 id 查询 token 字符串。
 *
 * @param vocab  词表对象
 * @param id     token id
 * @return const char* 成功返回 token 指针；失败返回 NULL
 */
const char* vocab_get_token(const Vocabulary* vocab, int id);

/**
 * @brief 将词表保存为二进制文件。
 *
 * @param vocab       词表对象
 * @param file_path   输出文件路径
 * @return int        TokenizerStatus
 */
int vocab_save_binary(const Vocabulary* vocab, const char* file_path);

/**
 * @brief 从二进制文件加载词表。
 *
 * @param file_path   输入文件路径
 * @param out_vocab   输出词表对象（函数内部会初始化）
 * @return int        TokenizerStatus
 */
int vocab_load_binary(const char* file_path, Vocabulary* out_vocab);

int vocab_load_text(const char* file_path, Vocabulary* out_vocab);

/**
 * @brief 初始化 Tokenizer。
 *
 * @param tokenizer  Tokenizer 对象
 * @param vocab      词表对象
 * @param unk_id     未登录词 id
 * @return int       TokenizerStatus
 */
int tokenizer_init(Tokenizer* tokenizer, const Vocabulary* vocab, int unk_id);

/**
 * @brief 文本编码：按空白分词并映射为 token id。
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
                     size_t* out_count);

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
                     size_t out_capacity);

#endif /* TOKENIZER_H */
