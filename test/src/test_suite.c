#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "../include/test_framework.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/include/csv_loader.h"
#include "../../src/include/model.h"
#include "../../src/include/ops.h"
#include "../../src/include/protocol.h"
#include "../../src/include/tensor.h"
#include "../../src/include/tokenizer.h"
#include "../../src/include/weights_io.h"

/**
 * @brief 创建可重复使用的临时文件路径。
 *
 * 设计说明：
 * - 使用固定文件名，保证在 CTest 的工作目录中可预测。
 * - 每次测试前覆盖写入，测试后删除，避免污染仓库文件。
 *
 * @param name 文件名
 * @param out  输出路径缓冲区
 * @param cap  输出路径容量
 */
static void make_test_file_path(const char* name, char* out, size_t cap) {
    (void)snprintf(out, cap, "%s", name);
}

/**
 * @brief 单元测试：tensor 计算元素数量与 shape 比较。
 *
 * @return int 0=通过，非0=失败
 */
static int test_unit_tensor_numel_and_shape(void) {
    size_t shape[2] = {2U, 3U};
    size_t numel = 0U;
    float data_a[6];
    float data_b[6];
    Tensor a;
    Tensor b;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_calc_numel(2U, shape, &numel));
    TFW_ASSERT_SIZE_EQ(6U, numel);
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&a, data_a, 2U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&b, data_b, 2U, shape));
    TFW_ASSERT_TRUE(tensor_same_shape(&a, &b));
    return 0;
}

/**
 * @brief 单元测试：2D 矩阵乘法结果正确。
 *
 * @return int 0=通过，非0=失败
 */
static int test_unit_matmul_basic(void) {
    size_t a_shape[2] = {2U, 2U};
    size_t b_shape[2] = {2U, 2U};
    size_t out_shape[2] = {2U, 2U};
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float out_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    Tensor a;
    Tensor b;
    Tensor out;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&a, a_data, 2U, a_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&b, b_data, 2U, b_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out, out_data, 2U, out_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, op_matmul_2d(&a, &b, &out));
    TFW_ASSERT_FLOAT_NEAR(19.0f, out_data[0], 1e-5f);
    TFW_ASSERT_FLOAT_NEAR(22.0f, out_data[1], 1e-5f);
    TFW_ASSERT_FLOAT_NEAR(43.0f, out_data[2], 1e-5f);
    TFW_ASSERT_FLOAT_NEAR(50.0f, out_data[3], 1e-5f);
    return 0;
}

/**
 * @brief 正确性测试：softmax 每行和应接近 1。
 *
 * @return int 0=通过，非0=失败
 */
static int test_correctness_softmax_sum_is_one(void) {
    size_t shape[2] = {2U, 3U};
    float in_data[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};
    float out_data[6] = {0};
    Tensor in;
    Tensor out;
    float row0 = 0.0f;
    float row1 = 0.0f;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&in, in_data, 2U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out, out_data, 2U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, op_softmax_last_dim(&in, &out));
    row0 = out_data[0] + out_data[1] + out_data[2];
    row1 = out_data[3] + out_data[4] + out_data[5];
    TFW_ASSERT_FLOAT_NEAR(1.0f, row0, 1e-5f);
    TFW_ASSERT_FLOAT_NEAR(1.0f, row1, 1e-5f);
    return 0;
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
 * @brief 临界测试：最小张量与最小协议帧。
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
 * @brief 压力测试：高频循环编码解码，验证稳定性与内存边界。
 *
 * @return int 0=通过，非0=失败
 */
static int test_stress_protocol_roundtrip(void) {
    size_t i = 0U;
    int ids[4] = {1, 2, 3, 4};
    char packet[128];
    ProtocolFrame frame;
    char raw[128];
    int out_ids[8];
    for (i = 0U; i < 10000U; ++i) {
        TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_OK,
                          protocol_encode_token(ids, 4U, packet, sizeof(packet), NULL));
        TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_OK,
                          protocol_decode_packet(packet, &frame, raw, sizeof(raw), out_ids, 8U));
        TFW_ASSERT_TRUE(frame.mode == PROTOCOL_MODE_TOKEN);
        TFW_ASSERT_SIZE_EQ(4U, frame.token_count);
        TFW_ASSERT_INT_EQ(1, frame.token_ids[0]);
        TFW_ASSERT_INT_EQ(4, frame.token_ids[3]);
    }
    testfw_log_info("压力测试完成：protocol 循环 10000 次。");
    return 0;
}

/**
 * @brief 集成测试：词表+Tokenizer+协议端到端闭环。
 *
 * @return int 0=通过，非0=失败
 */
static int test_integration_tokenizer_protocol_pipeline(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    int rc = 0;
    int ids[8];
    size_t count = 0U;
    char packet[128];
    ProtocolFrame frame;
    char raw_buf[128];
    int token_buf[16];
    char decoded[128];
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_init(&vocab, 8U));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "<unk>", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "go", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "left", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tokenizer_init(&tokenizer, &vocab, 0));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK,
                      tokenizer_encode(&tokenizer, "go left", ids, 8U, &count));
    TFW_ASSERT_SIZE_EQ(2U, count);
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_OK,
                      protocol_encode_token(ids, count, packet, sizeof(packet), NULL));
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_OK,
                      protocol_decode_packet(packet, &frame, raw_buf, sizeof(raw_buf), token_buf, 16U));
    TFW_ASSERT_TRUE(frame.mode == PROTOCOL_MODE_TOKEN);
    rc = tokenizer_decode(&vocab, frame.token_ids, frame.token_count, decoded, sizeof(decoded));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, rc);
    TFW_ASSERT_TRUE(strcmp(decoded, "go left") == 0);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 常规测试：CSV 数据加载与模型前向主流程。
 *
 * @return int 0=通过，非0=失败
 */
static int test_regular_csv_and_model_flow(void) {
    char csv_path[128];
    FILE* fp = NULL;
    CsvDataset dataset;
    float embed_table[EMBED_DIM * 2];
    EmbeddingLayer embedding;
    TransformerBlock block;
    int token_ids[2] = {0, 1};
    float vectors_in[EMBED_DIM * 2];
    float vectors_out[EMBED_DIM * 2];
    size_t i = 0U;

    memset(&dataset, 0, sizeof(dataset));
    for (i = 0U; i < EMBED_DIM * 2U; ++i) {
        embed_table[i] = (float)(i % 7U) * 0.1f;
    }

    make_test_file_path("test_regular_dataset.csv", csv_path, sizeof(csv_path));
    fp = fopen(csv_path, "w");
    TFW_ASSERT_TRUE(fp != NULL);
    fprintf(fp, "walk,1,2,3,4,5,6,7,8,0.1,0.2,0.3,0.4\n");
    fprintf(fp, "jump,2,3,4,5,6,7,8,9,0.2,0.3,0.4,0.5\n");
    fclose(fp);

    TFW_ASSERT_INT_EQ(0, csv_load_dataset(csv_path, &dataset));
    TFW_ASSERT_SIZE_EQ(2U, dataset.count);

    embedding.table = embed_table;
    embedding.vocab_size = 2U;
    embedding.embed_dim = EMBED_DIM;
    block.attention.embed_dim = EMBED_DIM;
    block.attention.num_heads = 4U;
    block.moe.num_experts = 2U;
    block.moe.k_top = 1U;
    model_embedding_forward(&embedding, token_ids, 2U, vectors_in);
    model_transformer_block_forward(&block, vectors_in, 2U, vectors_out);
    TFW_ASSERT_FLOAT_NEAR(vectors_in[0], vectors_out[0], 1e-6f);
    TFW_ASSERT_FLOAT_NEAR(vectors_in[EMBED_DIM], vectors_out[EMBED_DIM], 1e-6f);

    csv_free_dataset(&dataset);
    (void)remove(csv_path);
    return 0;
}

/**
 * @brief 集成测试：权重文件保存/加载与 C 源导出。
 *
 * @return int 0=通过，非0=失败
 */
static int test_integration_weights_io_roundtrip(void) {
    char bin_path[128];
    char c_path[128];
    float weights[4] = {0.25f, -0.5f, 1.0f, 2.0f};
    float* loaded = NULL;
    size_t loaded_count = 0U;
    make_test_file_path("test_weights.bin", bin_path, sizeof(bin_path));
    make_test_file_path("test_weights_export.c", c_path, sizeof(c_path));
    TFW_ASSERT_INT_EQ(WEIGHTS_IO_STATUS_OK, weights_save_binary(bin_path, weights, 4U));
    TFW_ASSERT_INT_EQ(WEIGHTS_IO_STATUS_OK, weights_load_binary(bin_path, &loaded, &loaded_count));
    TFW_ASSERT_SIZE_EQ(4U, loaded_count);
    TFW_ASSERT_FLOAT_NEAR(weights[0], loaded[0], 1e-6f);
    TFW_ASSERT_FLOAT_NEAR(weights[3], loaded[3], 1e-6f);
    TFW_ASSERT_INT_EQ(WEIGHTS_IO_STATUS_OK,
                      weights_export_c_source(c_path, "demo_weights", loaded, loaded_count));
    free(loaded);
    (void)remove(bin_path);
    (void)remove(c_path);
    return 0;
}

/**
 * @brief 程序入口：注册并执行全量测试。
 *
 * 分类覆盖：
 * - 单元测试
 * - 集成测试
 * - 正确性测试
 * - 错误测试
 * - 临界测试
 * - 压力测试
 * - 常规测试
 *
 * @return int 进程返回码
 */
int main(void) {
    const TestCase cases[] = {
        {"tensor_numel_and_shape", "单元", "验证张量元素计数与形状比较逻辑正确", "shape=[2,3]", "0(PASS)", test_unit_tensor_numel_and_shape},
        {"matmul_basic", "单元", "验证2x2矩阵乘法结果正确性", "A/B固定矩阵", "0(PASS)", test_unit_matmul_basic},
        {"softmax_sum_is_one", "正确性", "验证softmax每行归一化求和为1", "输入2x3矩阵", "0(PASS)", test_correctness_softmax_sum_is_one},
        {"invalid_arguments", "错误", "验证非法参数返回预期错误码", "NULL参数/非法对象", "0(PASS)", test_error_invalid_arguments},
        {"minimal_inputs", "临界", "验证最小输入规模下计算与协议解析边界", "1维最小张量与最短RAW帧", "0(PASS)", test_boundary_minimal_inputs},
        {"protocol_roundtrip_stress", "压力", "验证协议高频循环稳定性与边界安全", "10000次编码解码循环", "0(PASS)", test_stress_protocol_roundtrip},
        {"tokenizer_protocol_pipeline", "集成", "验证词表、tokenizer与协议端到端打通", "go left样本", "0(PASS)", test_integration_tokenizer_protocol_pipeline},
        {"weights_io_roundtrip", "集成", "验证权重二进制读写与C源码导出", "4个浮点权重样本", "0(PASS)", test_integration_weights_io_roundtrip},
        {"csv_and_model_flow", "常规", "验证CSV加载与模型前向主路径", "2行CSV+2 token输入", "0(PASS)", test_regular_csv_and_model_flow}
    };
    return testfw_run_all(cases, sizeof(cases) / sizeof(cases[0]));
}
