#include "../include/protocol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief 安全拼接文本到缓冲区，统一处理容量检查。
 *
 * @param dst           目标缓冲区
 * @param dst_capacity  目标容量
 * @param inout_used    已使用字节数（输入/输出）
 * @param src           追加文本
 * @return int          ProtocolStatus
 */
static int protocol_append_text(char* dst,
                                size_t dst_capacity,
                                size_t* inout_used,
                                const char* src) {
    size_t add_len = 0U;
    if (dst == NULL || inout_used == NULL || src == NULL) {
        return PROTOCOL_STATUS_INVALID_ARGUMENT;
    }
    add_len = strlen(src);
    if (*inout_used + add_len >= dst_capacity) {
        return PROTOCOL_STATUS_BUFFER_TOO_SMALL;
    }
    memcpy(dst + *inout_used, src, add_len);
    *inout_used += add_len;
    dst[*inout_used] = '\0';
    return PROTOCOL_STATUS_OK;
}

/**
 * @brief 解析十进制整数，失败时返回格式错误。
 *
 * @param text      输入文本
 * @param out_value 输出整数
 * @param out_next  解析停止位置（可为 NULL）
 * @return int      ProtocolStatus
 */
static int protocol_parse_int(const char* text, int* out_value, const char** out_next) {
    char* end_ptr = NULL;
    long value = 0L;
    if (text == NULL || out_value == NULL) {
        return PROTOCOL_STATUS_INVALID_ARGUMENT;
    }
    value = strtol(text, &end_ptr, 10);
    if (end_ptr == text) {
        return PROTOCOL_STATUS_FORMAT_ERROR;
    }
    *out_value = (int)value;
    if (out_next != NULL) {
        *out_next = end_ptr;
    }
    return PROTOCOL_STATUS_OK;
}

/**
 * @brief 将 Raw 文本编码为协议帧字符串。
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
                        size_t* out_size) {
    size_t used = 0U;
    int rc = 0;
    if (text == NULL || out_buffer == NULL || out_capacity == 0U) {
        return PROTOCOL_STATUS_INVALID_ARGUMENT;
    }
    out_buffer[0] = '\0';
    rc = protocol_append_text(out_buffer, out_capacity, &used, "RAW|");
    if (rc != PROTOCOL_STATUS_OK) {
        return rc;
    }
    rc = protocol_append_text(out_buffer, out_capacity, &used, text);
    if (rc != PROTOCOL_STATUS_OK) {
        return rc;
    }
    rc = protocol_append_text(out_buffer, out_capacity, &used, "\n");
    if (rc != PROTOCOL_STATUS_OK) {
        return rc;
    }
    if (out_size != NULL) {
        *out_size = used;
    }
    return PROTOCOL_STATUS_OK;
}

/**
 * @brief 将 Token 序列编码为协议帧字符串。
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
                          size_t* out_size) {
    size_t i = 0U;
    size_t used = 0U;
    int rc = 0;
    if (token_ids == NULL || out_buffer == NULL || out_capacity == 0U) {
        return PROTOCOL_STATUS_INVALID_ARGUMENT;
    }
    out_buffer[0] = '\0';
    rc = protocol_append_text(out_buffer, out_capacity, &used, "TOK|");
    if (rc != PROTOCOL_STATUS_OK) {
        return rc;
    }

    /* 关键保护点：
     * - 先输出 token 个数，解码端可提前做容量校验，避免写越界。 */
    {
        char count_buf[32];
        int n = snprintf(count_buf, sizeof(count_buf), "%zu|", token_count);
        if (n < 0 || (size_t)n >= sizeof(count_buf)) {
            return PROTOCOL_STATUS_FORMAT_ERROR;
        }
        rc = protocol_append_text(out_buffer, out_capacity, &used, count_buf);
        if (rc != PROTOCOL_STATUS_OK) {
            return rc;
        }
    }

    for (i = 0U; i < token_count; ++i) {
        char id_buf[32];
        int n = snprintf(id_buf, sizeof(id_buf), "%d%s", token_ids[i], (i + 1U < token_count) ? "," : "");
        if (n < 0 || (size_t)n >= sizeof(id_buf)) {
            return PROTOCOL_STATUS_FORMAT_ERROR;
        }
        rc = protocol_append_text(out_buffer, out_capacity, &used, id_buf);
        if (rc != PROTOCOL_STATUS_OK) {
            return rc;
        }
    }

    rc = protocol_append_text(out_buffer, out_capacity, &used, "\n");
    if (rc != PROTOCOL_STATUS_OK) {
        return rc;
    }
    if (out_size != NULL) {
        *out_size = used;
    }
    return PROTOCOL_STATUS_OK;
}

/**
 * @brief 解析 Token 帧的 payload 段。
 *
 * @param payload          count|id0,id1,...
 * @param token_buffer     token 输出数组
 * @param token_capacity   token 输出容量
 * @param out_token_count  实际 token 数
 * @return int             ProtocolStatus
 */
static int protocol_decode_token_payload(const char* payload,
                                         int* token_buffer,
                                         size_t token_capacity,
                                         size_t* out_token_count) {
    const char* p = payload;
    const char* sep = NULL;
    int count_i = 0;
    size_t i = 0U;
    int rc = 0;
    if (payload == NULL || token_buffer == NULL || out_token_count == NULL) {
        return PROTOCOL_STATUS_INVALID_ARGUMENT;
    }

    sep = strchr(p, '|');
    if (sep == NULL) {
        return PROTOCOL_STATUS_FORMAT_ERROR;
    }
    {
        char count_text[32];
        size_t len = (size_t)(sep - p);
        if (len == 0U || len >= sizeof(count_text)) {
            return PROTOCOL_STATUS_FORMAT_ERROR;
        }
        memcpy(count_text, p, len);
        count_text[len] = '\0';
        rc = protocol_parse_int(count_text, &count_i, NULL);
        if (rc != PROTOCOL_STATUS_OK || count_i < 0) {
            return PROTOCOL_STATUS_FORMAT_ERROR;
        }
    }
    if ((size_t)count_i > token_capacity) {
        return PROTOCOL_STATUS_BUFFER_TOO_SMALL;
    }

    p = sep + 1;
    for (i = 0U; i < (size_t)count_i; ++i) {
        const char* next = NULL;
        int token_id = 0;
        rc = protocol_parse_int(p, &token_id, &next);
        if (rc != PROTOCOL_STATUS_OK) {
            return PROTOCOL_STATUS_FORMAT_ERROR;
        }
        token_buffer[i] = token_id;
        if (i + 1U < (size_t)count_i) {
            if (*next != ',') {
                return PROTOCOL_STATUS_FORMAT_ERROR;
            }
            p = next + 1;
        } else {
            p = next;
        }
    }
    if (*p != '\0') {
        return PROTOCOL_STATUS_FORMAT_ERROR;
    }
    *out_token_count = (size_t)count_i;
    return PROTOCOL_STATUS_OK;
}

/**
 * @brief 解析协议帧。
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
                           size_t token_capacity) {
    size_t packet_len = 0U;
    if (packet == NULL || frame == NULL || raw_buffer == NULL || token_buffer == NULL) {
        return PROTOCOL_STATUS_INVALID_ARGUMENT;
    }

    packet_len = strlen(packet);
    if (packet_len == 0U) {
        return PROTOCOL_STATUS_FORMAT_ERROR;
    }

    frame->mode = PROTOCOL_MODE_RAW;
    frame->raw_text = NULL;
    frame->token_ids = NULL;
    frame->token_count = 0U;

    if (strncmp(packet, "RAW|", 4U) == 0) {
        const char* payload = packet + 4;
        size_t payload_len = strlen(payload);
        if (payload_len > 0U && payload[payload_len - 1U] == '\n') {
            payload_len -= 1U;
        }
        if (payload_len + 1U > raw_buffer_capacity) {
            return PROTOCOL_STATUS_BUFFER_TOO_SMALL;
        }
        memcpy(raw_buffer, payload, payload_len);
        raw_buffer[payload_len] = '\0';
        frame->mode = PROTOCOL_MODE_RAW;
        frame->raw_text = raw_buffer;
        return PROTOCOL_STATUS_OK;
    }

    if (strncmp(packet, "TOK|", 4U) == 0) {
        const char* payload = packet + 4;
        size_t payload_len = strlen(payload);
        int rc = 0;
        if (payload_len > 0U && payload[payload_len - 1U] == '\n') {
            char* mutable_payload = NULL;
            payload_len -= 1U;
            if (payload_len + 1U > raw_buffer_capacity) {
                return PROTOCOL_STATUS_BUFFER_TOO_SMALL;
            }
            memcpy(raw_buffer, payload, payload_len);
            raw_buffer[payload_len] = '\0';
            mutable_payload = raw_buffer;
            rc = protocol_decode_token_payload(mutable_payload, token_buffer, token_capacity, &frame->token_count);
            if (rc != PROTOCOL_STATUS_OK) {
                return rc;
            }
            frame->mode = PROTOCOL_MODE_TOKEN;
            frame->token_ids = token_buffer;
            frame->raw_text = NULL;
            return PROTOCOL_STATUS_OK;
        }
        return PROTOCOL_STATUS_FORMAT_ERROR;
    }

    return PROTOCOL_STATUS_FORMAT_ERROR;
}
