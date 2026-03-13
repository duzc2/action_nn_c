#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stddef.h>

/**
 * @brief 协议错误码，统一 Raw/Token 模式的返回状态。
 */
typedef enum ProtocolStatus {
    PROTOCOL_STATUS_OK = 0,
    PROTOCOL_STATUS_INVALID_ARGUMENT = -1,
    PROTOCOL_STATUS_BUFFER_TOO_SMALL = -2,
    PROTOCOL_STATUS_FORMAT_ERROR = -3
} ProtocolStatus;

/**
 * @brief 协议帧类型：Raw 文本或 Token 序列。
 */
typedef enum ProtocolMode {
    PROTOCOL_MODE_RAW = 0,
    PROTOCOL_MODE_TOKEN = 1
} ProtocolMode;

/**
 * @brief 解码后的统一帧结构，供训练闭环主程序直接使用。
 *
 * 设计说明：
 * - Raw 模式下，raw_text 指向调用方提供的缓存。
 * - Token 模式下，token_ids 写入调用方提供的数组。
 */
typedef struct ProtocolFrame {
    ProtocolMode mode;
    char* raw_text;
    int* token_ids;
    size_t token_count;
} ProtocolFrame;

/**
 * @brief 将 Raw 文本编码为协议帧字符串。
 *
 * 格式：
 * - RAW|<text>\n
 *
 * @param text         原始文本
 * @param out_buffer   输出缓冲区
 * @param out_capacity 输出缓冲区容量（字节）
 * @param out_size     实际写入字节数（可为 NULL）
 * @return int         ProtocolStatus
 */
int protocol_encode_raw(const char* text,
                        char* out_buffer,
                        size_t out_capacity,
                        size_t* out_size);

/**
 * @brief 将 Token 序列编码为协议帧字符串。
 *
 * 格式：
 * - TOK|<count>|id0,id1,...\n
 *
 * @param token_ids    token id 数组
 * @param token_count  token 数量
 * @param out_buffer   输出缓冲区
 * @param out_capacity 输出缓冲区容量（字节）
 * @param out_size     实际写入字节数（可为 NULL）
 * @return int         ProtocolStatus
 */
int protocol_encode_token(const int* token_ids,
                          size_t token_count,
                          char* out_buffer,
                          size_t out_capacity,
                          size_t* out_size);

/**
 * @brief 解析协议帧。
 *
 * 使用方式：
 * - 调用方提前准备 raw_buffer 与 token_buffer。
 * - 若为 Raw 帧，结果写入 raw_buffer，并将 frame->raw_text 指向它。
 * - 若为 Token 帧，结果写入 token_buffer，并设置 frame->token_ids/token_count。
 *
 * @param packet               输入帧字符串（以 '\0' 结尾）
 * @param frame                输出帧对象
 * @param raw_buffer           Raw 文本缓冲区
 * @param raw_buffer_capacity  Raw 文本缓冲区容量
 * @param token_buffer         Token 缓冲区
 * @param token_capacity       Token 缓冲区容量（元素数）
 * @return int                 ProtocolStatus
 */
int protocol_decode_packet(const char* packet,
                           ProtocolFrame* frame,
                           char* raw_buffer,
                           size_t raw_buffer_capacity,
                           int* token_buffer,
                           size_t token_capacity);

#endif /* PROTOCOL_H */
