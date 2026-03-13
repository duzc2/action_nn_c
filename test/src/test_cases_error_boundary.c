#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "../include/test_cases.h"

#include <stdio.h>
#include <string.h>

#include "../../src/include/csv_loader.h"
#include "../../src/include/ops.h"
#include "../../src/include/protocol.h"
#include "../../src/include/tensor.h"
#include "../../src/include/tokenizer.h"
#include "../../src/include/weights_io.h"

/**
 * @brief 生成测试临时文件路径。
 *
 * @param name 文件名
 * @param out 输出缓冲区
 * @param cap 输出缓冲区容量
 */
static void make_test_file_path(const char* name, char* out, size_t cap) {
    (void)snprintf(out, cap, "%s", name);
}

/**
 * @brief 错误测试：传入非法参数时返回预期错误码。
 *
 * @return int 0=通过，非0=失败
 */
static int test_error_invalid_arguments(void) {
    size_t shape[1] = {4U};
    float in_data[4] = {0};
    float out_data[4] = {0};
    Tensor in;
    Tensor out;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&in, in_data, 1U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out, out_data, 1U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_INVALID_ARGUMENT, op_actuator(&in, NULL, &out));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_INVALID_ARGUMENT, tensor_pool_init(NULL, out_data, 4U));
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_INVALID_ARGUMENT, protocol_encode_raw(NULL, (char*)out_data, 4U, NULL));
    return 0;
}

/**
 * @brief 错误测试：矩阵乘法形状不匹配时返回错误。
 *
 * @return int 0=通过，非0=失败
 */
static int test_error_matmul_shape_mismatch(void) {
    size_t a_shape[2] = {2U, 3U};
    size_t b_shape[2] = {4U, 2U};
    size_t out_shape[2] = {2U, 2U};
    float a_data[6] = {0};
    float b_data[8] = {0};
    float out_data[4] = {0};
    Tensor a;
    Tensor b;
    Tensor out;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&a, a_data, 2U, a_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&b, b_data, 2U, b_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out, out_data, 2U, out_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_SHAPE_MISMATCH, op_matmul_2d(&a, &b, &out));
    return 0;
}

/**
 * @brief 错误测试：协议格式错误数据应被拒绝。
 *
 * @return int 0=通过，非0=失败
 */
static int test_error_protocol_format_invalid_packets(void) {
    ProtocolFrame frame;
    char raw[64];
    int ids[8];
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_FORMAT_ERROR,
                      protocol_decode_packet("BAD|hello\n", &frame, raw, sizeof(raw), ids, 8U));
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_FORMAT_ERROR,
                      protocol_decode_packet("TOK|2|1;\n", &frame, raw, sizeof(raw), ids, 8U));
    return 0;
}

/**
 * @brief 错误测试：Tokenizer 解码不存在 id 返回错误。
 *
 * @return int 0=通过，非0=失败
 */
static int test_error_tokenizer_decode_not_found(void) {
    Vocabulary vocab;
    int ids[2] = {0, 9};
    char out[64];
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_init(&vocab, 4U));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "<unk>", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_NOT_FOUND, tokenizer_decode(&vocab, ids, 2U, out, sizeof(out)));
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 错误测试：导出权重时非法符号名应返回错误。
 *
 * @return int 0=通过，非0=失败
 */
static int test_error_weights_export_invalid_symbol(void) {
    float data[2] = {1.0f, 2.0f};
    TFW_ASSERT_INT_EQ(WEIGHTS_IO_STATUS_INVALID_ARGUMENT,
                      weights_export_c_source("test_invalid_symbol.c", "1bad", data, 2U));
    return 0;
}

/**
 * @brief 边界测试：最小张量与最小协议帧。
 *
 * @return int 0=通过，非0=失败
 */
static int test_boundary_minimal_inputs(void) {
    size_t shape[1] = {1U};
    float in_data[1] = {2.0f};
    float weight_data[1] = {3.0f};
    float out_data[1] = {0.0f};
    Tensor in;
    Tensor w;
    Tensor out;
    char packet[32];
    ProtocolFrame frame;
    char raw[32];
    int tokens[4];
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&in, in_data, 1U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&w, weight_data, 1U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out, out_data, 1U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, op_rmsnorm_last_dim(&in, &w, 1e-6f, &out));
    TFW_ASSERT_TRUE(out_data[0] > 2.9f && out_data[0] < 3.1f);
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_OK, protocol_encode_raw("a", packet, sizeof(packet), NULL));
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_OK,
                      protocol_decode_packet(packet, &frame, raw, sizeof(raw), tokens, 4U));
    TFW_ASSERT_TRUE(frame.mode == PROTOCOL_MODE_RAW);
    TFW_ASSERT_TRUE(strcmp(frame.raw_text, "a") == 0);
    return 0;
}

/**
 * @brief 边界测试：验证最大维度(4维)张量可初始化。
 *
 * @return int 0=通过，非0=失败
 */
static int test_boundary_tensor_max_dims(void) {
    size_t shape[4] = {1U, 1U, 1U, 1U};
    float data[1] = {7.0f};
    Tensor t;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&t, data, 4U, shape));
    TFW_ASSERT_SIZE_EQ(1U, t.numel);
    TFW_ASSERT_SIZE_EQ(1U, t.stride[0]);
    TFW_ASSERT_SIZE_EQ(1U, t.stride[3]);
    return 0;
}

/**
 * @brief 边界测试：空 RAW payload 可正确解码为空字符串。
 *
 * @return int 0=通过，非0=失败
 */
static int test_boundary_protocol_empty_raw_payload(void) {
    ProtocolFrame frame;
    char raw[16];
    int token_ids[4];
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_OK,
                      protocol_decode_packet("RAW|\n", &frame, raw, sizeof(raw), token_ids, 4U));
    TFW_ASSERT_TRUE(frame.mode == PROTOCOL_MODE_RAW);
    TFW_ASSERT_TRUE(frame.raw_text != NULL);
    TFW_ASSERT_TRUE(frame.raw_text[0] == '\0');
    return 0;
}

/**
 * @brief 边界测试：Tokenizer 输出容量不足时返回错误。
 *
 * @return int 0=通过，非0=失败
 */
static int test_boundary_tokenizer_encode_buffer_too_small(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    int ids[1] = {0};
    size_t count = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_init(&vocab, 8U));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "<unk>", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "go", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tokenizer_init(&tokenizer, &vocab, 0));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_BUFFER_TOO_SMALL,
                      tokenizer_encode(&tokenizer, "go go", ids, 1U, &count));
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 边界测试：CSV 加载应跳过非法行并保留合法行。
 *
 * @return int 0=通过，非0=失败
 */
static int test_boundary_csv_loader_skip_invalid_lines(void) {
    char csv_path[128];
    FILE* fp = NULL;
    CsvDataset dataset;
    memset(&dataset, 0, sizeof(dataset));
    make_test_file_path("test_boundary_dataset.csv", csv_path, sizeof(csv_path));
    fp = fopen(csv_path, "w");
    TFW_ASSERT_TRUE(fp != NULL);
    fprintf(fp, "bad_line_without_fields\n");
    fprintf(fp, "walk,1,2,3,4,5,6,7,8,0.1,0.2,0.3,0.4\n");
    fprintf(fp, "too_short,1,2\n");
    fclose(fp);
    TFW_ASSERT_INT_EQ(0, csv_load_dataset(csv_path, &dataset));
    TFW_ASSERT_SIZE_EQ(1U, dataset.count);
    TFW_ASSERT_TRUE(strcmp(dataset.samples[0].command, "walk") == 0);
    csv_free_dataset(&dataset);
    (void)remove(csv_path);
    return 0;
}

/**
 * @brief 返回“错误 + 边界”测试分组。
 *
 * @return TestCaseGroup 分组对象
 */
TestCaseGroup testcases_get_error_boundary_group(void) {
    static const TestCase cases[] = {
        {"invalid_arguments", "错误", "验证非法参数返回预期错误码", "NULL参数/非法对象", "0(PASS)", test_error_invalid_arguments},
        {"matmul_shape_mismatch", "错误", "验证矩阵乘法形状不匹配的错误返回", "A[2x3],B[4x2]", "0(PASS)", test_error_matmul_shape_mismatch},
        {"protocol_format_invalid_packets", "错误", "验证协议格式错误包被拒绝", "BAD/TOK非法分隔符", "0(PASS)", test_error_protocol_format_invalid_packets},
        {"tokenizer_decode_not_found", "错误", "验证解码非法id返回NOT_FOUND", "id=9超范围", "0(PASS)", test_error_tokenizer_decode_not_found},
        {"weights_export_invalid_symbol", "错误", "验证导出权重非法符号名返回错误", "symbol=1bad", "0(PASS)", test_error_weights_export_invalid_symbol},
        {"minimal_inputs", "边界", "验证最小输入规模下计算与协议解析边界", "1维最小张量与最短RAW帧", "0(PASS)", test_boundary_minimal_inputs},
        {"tensor_max_dims", "边界", "验证4维张量初始化与stride边界", "shape=[1,1,1,1]", "0(PASS)", test_boundary_tensor_max_dims},
        {"protocol_empty_raw_payload", "边界", "验证空RAW payload可被正确解析", "packet=RAW|\\n", "0(PASS)", test_boundary_protocol_empty_raw_payload},
        {"tokenizer_encode_buffer_too_small", "边界", "验证Tokenizer编码输出缓冲区不足时返回错误", "text='go go',capacity=1", "0(PASS)", test_boundary_tokenizer_encode_buffer_too_small},
        {"csv_loader_skip_invalid_lines", "边界", "验证CSV加载时跳过非法行", "混合合法/非法行", "0(PASS)", test_boundary_csv_loader_skip_invalid_lines}
    };
    TestCaseGroup group;
    group.cases = cases;
    group.count = sizeof(cases) / sizeof(cases[0]);
    return group;
}
