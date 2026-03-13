#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "../include/test_cases.h"

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
 * @param name 文件名
 * @param out  输出路径缓冲区
 * @param cap  输出路径容量
 */
static void make_test_file_path(const char* name, char* out, size_t cap) {
    (void)snprintf(out, cap, "%s", name);
}

/**
 * @brief 压力测试：高频循环编码解码，验证稳定性与边界安全。
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
 * @brief 压力测试：重复矩阵乘法，验证计算稳定性。
 *
 * @return int 0=通过，非0=失败
 */
static int test_stress_matmul_repeat(void) {
    size_t i = 0U;
    size_t a_shape[2] = {2U, 2U};
    size_t b_shape[2] = {2U, 2U};
    size_t out_shape[2] = {2U, 2U};
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {0.5f, 1.0f, -1.0f, 2.0f};
    float out_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    Tensor a;
    Tensor b;
    Tensor out;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&a, a_data, 2U, a_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&b, b_data, 2U, b_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out, out_data, 2U, out_shape));
    for (i = 0U; i < 5000U; ++i) {
        TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, op_matmul_2d(&a, &b, &out));
        TFW_ASSERT_FLOAT_NEAR(-1.5f, out_data[0], 1e-6f);
        TFW_ASSERT_FLOAT_NEAR(5.0f, out_data[1], 1e-6f);
        TFW_ASSERT_FLOAT_NEAR(-2.5f, out_data[2], 1e-6f);
        TFW_ASSERT_FLOAT_NEAR(11.0f, out_data[3], 1e-6f);
    }
    testfw_log_info("压力测试完成：matmul 循环 5000 次。");
    return 0;
}

/**
 * @brief 压力测试：长文本编码，验证分词在较大输入下稳定。
 *
 * @return int 0=通过，非0=失败
 */
static int test_stress_tokenizer_long_text(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    char text[4096];
    int out_ids[1024];
    size_t count = 0U;
    size_t i = 0U;
    size_t used = 0U;
    memset(&vocab, 0, sizeof(vocab));
    memset(text, 0, sizeof(text));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_init(&vocab, 8U));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "<unk>", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "go", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "left", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tokenizer_init(&tokenizer, &vocab, 0));
    for (i = 0U; i < 400U; ++i) {
        const char* token = (i % 2U == 0U) ? "go" : "left";
        int n = snprintf(text + used, sizeof(text) - used, "%s%s", token, (i + 1U < 400U) ? " " : "");
        TFW_ASSERT_TRUE(n > 0);
        used += (size_t)n;
    }
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tokenizer_encode(&tokenizer, text, out_ids, 1024U, &count));
    TFW_ASSERT_SIZE_EQ(400U, count);
    TFW_ASSERT_INT_EQ(1, out_ids[0]);
    TFW_ASSERT_INT_EQ(2, out_ids[1]);
    TFW_ASSERT_INT_EQ(2, out_ids[399]);
    vocab_free(&vocab);
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
    FILE* fp = NULL;
    char text_buf[256];
    size_t read_n = 0U;
    make_test_file_path("test_weights.bin", bin_path, sizeof(bin_path));
    make_test_file_path("test_weights_export.c", c_path, sizeof(c_path));
    TFW_ASSERT_INT_EQ(WEIGHTS_IO_STATUS_OK, weights_save_binary(bin_path, weights, 4U));
    TFW_ASSERT_INT_EQ(WEIGHTS_IO_STATUS_OK, weights_load_binary(bin_path, &loaded, &loaded_count));
    TFW_ASSERT_SIZE_EQ(4U, loaded_count);
    TFW_ASSERT_FLOAT_NEAR(weights[0], loaded[0], 1e-6f);
    TFW_ASSERT_FLOAT_NEAR(weights[3], loaded[3], 1e-6f);
    TFW_ASSERT_INT_EQ(WEIGHTS_IO_STATUS_OK,
                      weights_export_c_source(c_path, "demo_weights", loaded, loaded_count));
    fp = fopen(c_path, "r");
    TFW_ASSERT_TRUE(fp != NULL);
    read_n = fread(text_buf, 1U, sizeof(text_buf) - 1U, fp);
    text_buf[read_n] = '\0';
    fclose(fp);
    TFW_ASSERT_TRUE(strstr(text_buf, "demo_weights_count") != NULL);
    TFW_ASSERT_TRUE(strstr(text_buf, "demo_weights[4]") != NULL);
    free(loaded);
    (void)remove(bin_path);
    (void)remove(c_path);
    return 0;
}

/**
 * @brief 集成测试：CSV 数据加载与模型前向主流程。
 *
 * @return int 0=通过，非0=失败
 */
static int test_integration_csv_and_model_flow(void) {
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
 * @brief 集成测试：词表保存/加载后编码结果应保持一致。
 *
 * @return int 0=通过，非0=失败
 */
static int test_integration_vocab_binary_roundtrip(void) {
    Vocabulary vocab;
    Vocabulary loaded;
    Tokenizer tokenizer;
    int ids[4];
    size_t count = 0U;
    char file_path[128];
    memset(&vocab, 0, sizeof(vocab));
    memset(&loaded, 0, sizeof(loaded));
    make_test_file_path("test_vocab.bin", file_path, sizeof(file_path));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_init(&vocab, 8U));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "<unk>", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "open", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "door", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_save_binary(&vocab, file_path));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_load_binary(file_path, &loaded));
    TFW_ASSERT_SIZE_EQ(vocab.size, loaded.size);
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tokenizer_init(&tokenizer, &loaded, 0));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tokenizer_encode(&tokenizer, "open door", ids, 4U, &count));
    TFW_ASSERT_SIZE_EQ(2U, count);
    TFW_ASSERT_INT_EQ(1, ids[0]);
    TFW_ASSERT_INT_EQ(2, ids[1]);
    vocab_free(&vocab);
    vocab_free(&loaded);
    (void)remove(file_path);
    return 0;
}

/**
 * @brief 集成测试：未知词经编码、协议传输、解码后应映射到 <unk>。
 *
 * @return int 0=通过，非0=失败
 */
static int test_integration_unknown_token_pipeline(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    int ids[8] = {0};
    size_t count = 0U;
    char packet[128];
    ProtocolFrame frame;
    int token_buf[8] = {0};
    char raw_buf[64];
    char decoded[64];
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_init(&vocab, 8U));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "<unk>", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "go", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tokenizer_init(&tokenizer, &vocab, 0));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK,
                      tokenizer_encode(&tokenizer, "go mystery", ids, 8U, &count));
    TFW_ASSERT_SIZE_EQ(2U, count);
    TFW_ASSERT_INT_EQ(1, ids[0]);
    TFW_ASSERT_INT_EQ(0, ids[1]);
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_OK,
                      protocol_encode_token(ids, count, packet, sizeof(packet), NULL));
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_OK,
                      protocol_decode_packet(packet, &frame, raw_buf, sizeof(raw_buf), token_buf, 8U));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK,
                      tokenizer_decode(&vocab, frame.token_ids, frame.token_count, decoded, sizeof(decoded)));
    TFW_ASSERT_TRUE(strcmp(decoded, "go <unk>") == 0);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 返回“压力 + 集成”测试分组。
 *
 * @return TestCaseGroup 分组对象
 */
TestCaseGroup testcases_get_stress_integration_group(void) {
    static const TestCase cases[] = {
        {"protocol_roundtrip_stress", "压力", "验证协议高频循环稳定性与边界安全", "10000次编码解码循环", "0(PASS)", test_stress_protocol_roundtrip},
        {"matmul_repeat_stress", "压力", "验证矩阵乘法重复计算稳定性", "2x2循环5000次", "0(PASS)", test_stress_matmul_repeat},
        {"tokenizer_long_text_stress", "压力", "验证Tokenizer在长文本输入下稳定性", "400 token长文本", "0(PASS)", test_stress_tokenizer_long_text},
        {"tokenizer_protocol_pipeline", "集成", "验证词表、tokenizer与协议端到端打通", "go left样本", "0(PASS)", test_integration_tokenizer_protocol_pipeline},
        {"weights_io_roundtrip", "集成", "验证权重二进制读写与C源码导出", "4个浮点权重样本", "0(PASS)", test_integration_weights_io_roundtrip},
        {"csv_and_model_flow", "集成", "验证CSV加载与模型前向主路径", "2行CSV+2 token输入", "0(PASS)", test_integration_csv_and_model_flow},
        {"vocab_binary_roundtrip", "集成", "验证词表二进制保存加载与编码一致性", "open door样本", "0(PASS)", test_integration_vocab_binary_roundtrip},
        {"unknown_token_pipeline", "集成", "验证未知词在协议链路中映射到<unk>", "go mystery样本", "0(PASS)", test_integration_unknown_token_pipeline}
    };
    TestCaseGroup group;
    group.cases = cases;
    group.count = sizeof(cases) / sizeof(cases[0]);
    return group;
}
