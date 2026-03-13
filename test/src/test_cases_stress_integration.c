#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "../include/test_cases.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../../src/include/csv_loader.h"
#include "../../src/include/model.h"
#include "../../src/include/ops.h"
#include "../../src/include/protocol.h"
#include "../../src/include/tensor.h"
#include "../../src/include/tokenizer.h"
#include "../../src/include/weights_io.h"

enum {
    TINY_TOKEN_WEIGHT_COUNT = VOCAB_SIZE * OUTPUT_DIM,
    TINY_STATE_WEIGHT_COUNT = STATE_DIM * OUTPUT_DIM,
    TINY_BIAS_COUNT = OUTPUT_DIM,
    TINY_TOTAL_WEIGHT_COUNT = TINY_TOKEN_WEIGHT_COUNT + TINY_STATE_WEIGHT_COUNT + TINY_BIAS_COUNT
};

typedef struct TinySample {
    int ids[8];
    size_t token_count;
    float state[STATE_DIM];
    float target[OUTPUT_DIM];
} TinySample;

static unsigned int tiny_lcg_next(unsigned int* seed) {
    *seed = (*seed * 1664525U) + 1013904223U;
    return *seed;
}

static float tiny_rand_range(unsigned int* seed, float lo, float hi) {
    unsigned int v = tiny_lcg_next(seed);
    float t = (float)(v & 0x00FFFFFFU) / (float)0x01000000U;
    return lo + (hi - lo) * t;
}

static float tiny_clamp(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static int tiny_init_vocab_tokenizer(Vocabulary* vocab, Tokenizer* tokenizer) {
    static const char* tokens[] = {"<unk>", "move", "left", "right", "stop", "fast", "slow"};
    size_t i = 0U;
    int rc = 0;
    if (vocab == NULL || tokenizer == NULL) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    rc = vocab_init(vocab, VOCAB_SIZE);
    if (rc != TOKENIZER_STATUS_OK) {
        return rc;
    }
    for (i = 0U; i < sizeof(tokens) / sizeof(tokens[0]); ++i) {
        rc = vocab_add_token(vocab, tokens[i], NULL);
        if (rc != TOKENIZER_STATUS_OK) {
            vocab_free(vocab);
            return rc;
        }
    }
    rc = tokenizer_init(tokenizer, vocab, 0);
    if (rc != TOKENIZER_STATUS_OK) {
        vocab_free(vocab);
        return rc;
    }
    return TOKENIZER_STATUS_OK;
}

static void tiny_init_weights(float* weights, size_t count) {
    size_t i = 0U;
    for (i = 0U; i < count; ++i) {
        weights[i] = ((float)((int)(i % 19U) - 9)) * 0.001f;
    }
}

static void tiny_predict_logits(const float* weights,
                                const int* token_ids,
                                size_t token_count,
                                const float* state,
                                float* out_logits) {
    const size_t token_base = 0U;
    const size_t state_base = TINY_TOKEN_WEIGHT_COUNT;
    const size_t bias_base = TINY_TOKEN_WEIGHT_COUNT + TINY_STATE_WEIGHT_COUNT;
    size_t i = 0U;
    size_t j = 0U;
    for (j = 0U; j < OUTPUT_DIM; ++j) {
        out_logits[j] = weights[bias_base + j];
    }
    for (i = 0U; i < token_count; ++i) {
        size_t id = (token_ids[i] >= 0) ? (size_t)token_ids[i] : 0U;
        if (id >= VOCAB_SIZE) {
            id = 0U;
        }
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            out_logits[j] += weights[token_base + id * OUTPUT_DIM + j] / (float)token_count;
        }
    }
    for (i = 0U; i < STATE_DIM; ++i) {
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            out_logits[j] += state[i] * weights[state_base + i * OUTPUT_DIM + j];
        }
    }
}

static int tiny_activate(const float* logits, float* out_act) {
    const int activations[OUTPUT_DIM] = IO_MAPPING_ACTIVATIONS;
    Tensor in;
    Tensor out;
    size_t shape[1] = {OUTPUT_DIM};
    if (tensor_init_view(&in, (float*)logits, 1U, shape) != TENSOR_STATUS_OK) {
        return -1;
    }
    if (tensor_init_view(&out, out_act, 1U, shape) != TENSOR_STATUS_OK) {
        return -2;
    }
    if (op_actuator(&in, activations, &out) != TENSOR_STATUS_OK) {
        return -3;
    }
    return 0;
}

static int tiny_train_one(float* weights, const TinySample* sample, float lr, float* out_loss) {
    const size_t token_base = 0U;
    const size_t state_base = TINY_TOKEN_WEIGHT_COUNT;
    const size_t bias_base = TINY_TOKEN_WEIGHT_COUNT + TINY_STATE_WEIGHT_COUNT;
    float logits[OUTPUT_DIM] = {0};
    float pred[OUTPUT_DIM] = {0};
    float grad[OUTPUT_DIM] = {0};
    size_t i = 0U;
    size_t j = 0U;
    if (sample == NULL || sample->token_count == 0U) {
        return -1;
    }
    tiny_predict_logits(weights, sample->ids, sample->token_count, sample->state, logits);
    if (tiny_activate(logits, pred) != 0) {
        return -2;
    }
    *out_loss = 0.0f;
    for (j = 0U; j < OUTPUT_DIM; ++j) {
        float err = pred[j] - sample->target[j];
        grad[j] = err;
        *out_loss += err * err;
    }
    *out_loss /= (float)OUTPUT_DIM;
    for (i = 0U; i < sample->token_count; ++i) {
        size_t id = (sample->ids[i] >= 0) ? (size_t)sample->ids[i] : 0U;
        if (id >= VOCAB_SIZE) {
            id = 0U;
        }
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            weights[token_base + id * OUTPUT_DIM + j] -= (lr * grad[j]) / (float)sample->token_count;
        }
    }
    for (i = 0U; i < STATE_DIM; ++i) {
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            weights[state_base + i * OUTPUT_DIM + j] -= lr * grad[j] * sample->state[i];
        }
    }
    for (j = 0U; j < OUTPUT_DIM; ++j) {
        weights[bias_base + j] -= lr * grad[j];
    }
    return 0;
}

static int tiny_make_sample(Tokenizer* tokenizer,
                            unsigned int* seed,
                            TinySample* out_sample,
                            float x_min,
                            float x_max,
                            float y_min,
                            float y_max) {
    float x = tiny_rand_range(seed, x_min, x_max);
    float y = tiny_rand_range(seed, y_min, y_max);
    float tx = tiny_rand_range(seed, x_min, x_max);
    float ty = tiny_rand_range(seed, y_min, y_max);
    float dx = tx - x;
    float dy = ty - y;
    const char* cmd = NULL;
    int rc = 0;
    if (fabsf(dx) < 0.2f && fabsf(dy) < 0.2f) {
        cmd = "move stop";
    } else if (dx > 0.0f) {
        cmd = (fabsf(dx) > 3.0f) ? "move right fast" : "move right slow";
    } else {
        cmd = (fabsf(dx) > 3.0f) ? "move left fast" : "move left slow";
    }
    memset(out_sample, 0, sizeof(*out_sample));
    out_sample->state[0] = x / 20.0f;
    out_sample->state[1] = y / 20.0f;
    out_sample->state[2] = dx / 20.0f;
    out_sample->state[3] = dy / 20.0f;
    out_sample->state[4] = tiny_clamp((fabsf(dx) + fabsf(dy)) / 40.0f, 0.0f, 1.0f);
    out_sample->state[5] = tiny_clamp(fabsf(dx) / 20.0f, 0.0f, 1.0f);
    out_sample->state[6] = tiny_clamp(fabsf(dy) / 20.0f, 0.0f, 1.0f);
    out_sample->state[7] = 1.0f;
    rc = tokenizer_encode(tokenizer, cmd, out_sample->ids, 8U, &out_sample->token_count);
    if (rc != TOKENIZER_STATUS_OK || out_sample->token_count == 0U) {
        return -1;
    }
    out_sample->target[0] = (dx < -0.2f) ? 1.0f : 0.0f;
    out_sample->target[1] = (dx > 0.2f) ? 1.0f : 0.0f;
    out_sample->target[2] = tiny_clamp(dx / 6.0f, -1.0f, 1.0f);
    out_sample->target[3] = tiny_clamp(dy / 6.0f, -1.0f, 1.0f);
    return 0;
}

static int test_integration_generalization_unseen_values(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    TinySample train_set[320];
    TinySample eval_set[220];
    float weights[TINY_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 123456789U;
    size_t i = 0U;
    size_t epoch = 0U;
    float init_loss = 0.0f;
    float final_loss = 0.0f;
    float mae_x = 0.0f;
    float mae_y = 0.0f;
    size_t dir_total = 0U;
    size_t dir_hit = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tiny_init_vocab_tokenizer(&vocab, &tokenizer));
    tiny_init_weights(weights, TINY_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 320U; ++i) {
        TFW_ASSERT_INT_EQ(0, tiny_make_sample(&tokenizer, &seed, &train_set[i], -10.0f, 10.0f, -10.0f, 10.0f));
    }
    for (i = 0U; i < 220U; ++i) {
        TFW_ASSERT_INT_EQ(0, tiny_make_sample(&tokenizer, &seed, &eval_set[i], -15.0f, 15.0f, -15.0f, 15.0f));
    }
    for (i = 0U; i < 320U; ++i) {
        float l = 0.0f;
        TFW_ASSERT_INT_EQ(0, tiny_train_one(weights, &train_set[i], 0.0f, &l));
        init_loss += l;
    }
    init_loss /= 320.0f;
    for (epoch = 0U; epoch < 25U; ++epoch) {
        for (i = 0U; i < 320U; ++i) {
            float l = 0.0f;
            TFW_ASSERT_INT_EQ(0, tiny_train_one(weights, &train_set[i], 0.05f, &l));
            final_loss += l;
        }
    }
    final_loss /= (320.0f * 25.0f);
    for (i = 0U; i < 220U; ++i) {
        float logits[OUTPUT_DIM] = {0};
        float act[OUTPUT_DIM] = {0};
        float pred_dir = 0.0f;
        float gt_dir = eval_set[i].target[2];
        tiny_predict_logits(weights, eval_set[i].ids, eval_set[i].token_count, eval_set[i].state, logits);
        TFW_ASSERT_INT_EQ(0, tiny_activate(logits, act));
        mae_x += fabsf(act[2] - eval_set[i].target[2]);
        mae_y += fabsf(act[3] - eval_set[i].target[3]);
        if (fabsf(gt_dir) > 0.12f) {
            dir_total += 1U;
            pred_dir = (act[2] > 0.05f) ? 1.0f : ((act[2] < -0.05f) ? -1.0f : 0.0f);
            if ((gt_dir > 0.0f && pred_dir > 0.0f) || (gt_dir < 0.0f && pred_dir < 0.0f)) {
                dir_hit += 1U;
            }
        }
        TFW_ASSERT_TRUE(fabsf(act[2]) <= 1.01f);
        TFW_ASSERT_TRUE(fabsf(act[3]) <= 1.01f);
    }
    mae_x /= 220.0f;
    mae_y /= 220.0f;
    testfw_log_info("generalization metrics: init_loss=%.4f final_loss=%.4f mae_x=%.4f mae_y=%.4f dir=%zu/%zu",
                    (double)init_loss,
                    (double)final_loss,
                    (double)mae_x,
                    (double)mae_y,
                    dir_hit,
                    dir_total);
    TFW_ASSERT_TRUE(final_loss < init_loss * 0.45f);
    TFW_ASSERT_TRUE(mae_x < 0.32f);
    TFW_ASSERT_TRUE(mae_y < 0.34f);
    TFW_ASSERT_TRUE(dir_total > 100U);
    TFW_ASSERT_TRUE((double)dir_hit / (double)dir_total > 0.86);
    vocab_free(&vocab);
    return 0;
}

static int test_stress_noise_robustness_grid(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    float weights[TINY_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 987654321U;
    size_t i = 0U;
    size_t k = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tiny_init_vocab_tokenizer(&vocab, &tokenizer));
    tiny_init_weights(weights, TINY_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 280U; ++i) {
        TinySample s;
        float l = 0.0f;
        TFW_ASSERT_INT_EQ(0, tiny_make_sample(&tokenizer, &seed, &s, -12.0f, 12.0f, -12.0f, 12.0f));
        TFW_ASSERT_INT_EQ(0, tiny_train_one(weights, &s, 0.06f, &l));
    }
    for (k = 0U; k < 1200U; ++k) {
        TinySample base;
        TinySample noise;
        float logits_a[OUTPUT_DIM] = {0};
        float logits_b[OUTPUT_DIM] = {0};
        float act_a[OUTPUT_DIM] = {0};
        float act_b[OUTPUT_DIM] = {0};
        size_t d = 0U;
        TFW_ASSERT_INT_EQ(0, tiny_make_sample(&tokenizer, &seed, &base, -15.0f, 15.0f, -15.0f, 15.0f));
        noise = base;
        for (d = 0U; d < STATE_DIM; ++d) {
            float eps = tiny_rand_range(&seed, -0.025f, 0.025f);
            noise.state[d] = tiny_clamp(noise.state[d] + eps, -1.5f, 1.5f);
        }
        tiny_predict_logits(weights, base.ids, base.token_count, base.state, logits_a);
        tiny_predict_logits(weights, noise.ids, noise.token_count, noise.state, logits_b);
        TFW_ASSERT_INT_EQ(0, tiny_activate(logits_a, act_a));
        TFW_ASSERT_INT_EQ(0, tiny_activate(logits_b, act_b));
        TFW_ASSERT_TRUE(fabsf(act_a[2] - act_b[2]) < 0.35f);
        TFW_ASSERT_TRUE(fabsf(act_a[3] - act_b[3]) < 0.35f);
        TFW_ASSERT_TRUE(fabsf(act_b[2]) <= 1.01f);
        TFW_ASSERT_TRUE(fabsf(act_b[3]) <= 1.01f);
    }
    testfw_log_info("noise robustness: validated 1200 perturbed samples");
    vocab_free(&vocab);
    return 0;
}

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
        {"noise_robustness_grid", "压力", "验证状态微扰下动作输出平滑且有界", "1200组扰动样本", "0(PASS)", test_stress_noise_robustness_grid},
        {"tokenizer_protocol_pipeline", "集成", "验证词表、tokenizer与协议端到端打通", "go left样本", "0(PASS)", test_integration_tokenizer_protocol_pipeline},
        {"weights_io_roundtrip", "集成", "验证权重二进制读写与C源码导出", "4个浮点权重样本", "0(PASS)", test_integration_weights_io_roundtrip},
        {"csv_and_model_flow", "集成", "验证CSV加载与模型前向主路径", "2行CSV+2 token输入", "0(PASS)", test_integration_csv_and_model_flow},
        {"vocab_binary_roundtrip", "集成", "验证词表二进制保存加载与编码一致性", "open door样本", "0(PASS)", test_integration_vocab_binary_roundtrip},
        {"unknown_token_pipeline", "集成", "验证未知词在协议链路中映射到<unk>", "go mystery样本", "0(PASS)", test_integration_unknown_token_pipeline},
        {"generalization_unseen_values", "集成", "验证训练后对未见值输入仍输出合理动作", "训练320样本+评估220样本", "0(PASS)", test_integration_generalization_unseen_values}
    };
    TestCaseGroup group;
    group.cases = cases;
    group.count = sizeof(cases) / sizeof(cases[0]);
    return group;
}
