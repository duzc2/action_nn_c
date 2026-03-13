#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/config_user.h"
#include "../include/ops.h"
#include "../include/platform_driver.h"
#include "../include/protocol.h"
#include "../include/tensor.h"
#include "../include/tokenizer.h"
#include "../include/weights_io.h"

/**
 * @brief 最小训练样本定义，支持 Raw/Token 两种输入协议。
 */
typedef struct TrainSample {
    ProtocolMode mode;
    const char* raw_text;
    int token_ids[8];
    size_t token_count;
    float target[OUTPUT_DIM];
} TrainSample;

/**
 * @brief 初始化最小词表，供 Raw 协议下文本编码使用。
 *
 * @param vocab 输出词表对象
 * @return int TokenizerStatus
 */
static int build_demo_vocab(Vocabulary* vocab) {
    static const char* tokens[] = {
        "<unk>", "move", "left", "right", "stop", "fast"
    };
    size_t i = 0U;
    int rc = vocab_init(vocab, VOCAB_SIZE);
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
    return TOKENIZER_STATUS_OK;
}

/**
 * @brief 构造最小训练样本集。
 *
 * 设计说明：
 * - 样本 0/1 使用 Raw 协议，覆盖文本->token 流程。
 * - 样本 2/3 使用 Token 协议，覆盖 token 流直达流程。
 *
 * @param out_samples 输出数组
 * @param out_count   输出数量
 */
static void build_demo_samples(TrainSample* out_samples, size_t* out_count) {
    if (out_samples == NULL || out_count == NULL) {
        return;
    }
    memset(out_samples, 0, sizeof(TrainSample) * 4U);

    out_samples[0].mode = PROTOCOL_MODE_RAW;
    out_samples[0].raw_text = "move left";
    out_samples[0].target[0] = 1.0f;
    out_samples[0].target[1] = 0.0f;
    out_samples[0].target[2] = -0.6f;
    out_samples[0].target[3] = 0.0f;

    out_samples[1].mode = PROTOCOL_MODE_RAW;
    out_samples[1].raw_text = "move right";
    out_samples[1].target[0] = 0.0f;
    out_samples[1].target[1] = 1.0f;
    out_samples[1].target[2] = 0.6f;
    out_samples[1].target[3] = 0.0f;

    out_samples[2].mode = PROTOCOL_MODE_TOKEN;
    out_samples[2].token_ids[0] = 1; /* move */
    out_samples[2].token_ids[1] = 4; /* stop */
    out_samples[2].token_count = 2U;
    out_samples[2].target[0] = 0.0f;
    out_samples[2].target[1] = 0.0f;
    out_samples[2].target[2] = 0.0f;
    out_samples[2].target[3] = 0.0f;

    out_samples[3].mode = PROTOCOL_MODE_TOKEN;
    out_samples[3].token_ids[0] = 1; /* move */
    out_samples[3].token_ids[1] = 5; /* fast */
    out_samples[3].token_count = 2U;
    out_samples[3].target[0] = 1.0f;
    out_samples[3].target[1] = 0.0f;
    out_samples[3].target[2] = 0.8f;
    out_samples[3].target[3] = 0.2f;

    *out_count = 4U;
}

/**
 * @brief 通过 Raw/Token 协议统一解出 token id 序列。
 *
 * @param sample      输入样本
 * @param tokenizer   Tokenizer 实例（Raw 模式需要）
 * @param out_ids     输出 token id
 * @param out_cap     输出容量
 * @param out_count   输出数量
 * @return int        0 成功，非 0 失败
 */
static int resolve_sample_tokens(const TrainSample* sample,
                                 const Tokenizer* tokenizer,
                                 int* out_ids,
                                 size_t out_cap,
                                 size_t* out_count) {
    char packet[256];
    char raw_buffer[128];
    int token_buffer[32];
    ProtocolFrame frame;
    size_t packet_size = 0U;
    int rc = 0;
    if (sample == NULL || tokenizer == NULL || out_ids == NULL || out_count == NULL) {
        return -1;
    }
    memset(packet, 0, sizeof(packet));
    memset(raw_buffer, 0, sizeof(raw_buffer));
    memset(token_buffer, 0, sizeof(token_buffer));
    memset(&frame, 0, sizeof(frame));

    if (sample->mode == PROTOCOL_MODE_RAW) {
        rc = protocol_encode_raw(sample->raw_text, packet, sizeof(packet), &packet_size);
    } else {
        rc = protocol_encode_token(sample->token_ids, sample->token_count, packet, sizeof(packet), &packet_size);
    }
    if (rc != PROTOCOL_STATUS_OK || packet_size == 0U) {
        return -2;
    }

    rc = protocol_decode_packet(packet,
                                &frame,
                                raw_buffer,
                                sizeof(raw_buffer),
                                token_buffer,
                                sizeof(token_buffer) / sizeof(token_buffer[0]));
    if (rc != PROTOCOL_STATUS_OK) {
        return -3;
    }

    if (frame.mode == PROTOCOL_MODE_RAW) {
        rc = tokenizer_encode(tokenizer, frame.raw_text, out_ids, out_cap, out_count);
        return (rc == TOKENIZER_STATUS_OK) ? 0 : -4;
    }
    if (frame.mode == PROTOCOL_MODE_TOKEN) {
        if (frame.token_count > out_cap) {
            return -5;
        }
        memcpy(out_ids, frame.token_ids, sizeof(int) * frame.token_count);
        *out_count = frame.token_count;
        return 0;
    }
    return -6;
}

/**
 * @brief 根据 token 平均嵌入计算模型原始输出（logits）。
 *
 * 最小算法说明：
 * - 使用词表大小 x OUTPUT_DIM 的线性表作为“可训练参数”。
 * - 输入 token 的行向量做平均，得到单步动作 logits。
 *
 * @param weights      训练参数表
 * @param vocab_size   词表大小
 * @param token_ids    输入 token
 * @param token_count  输入 token 数
 * @param out_logits   输出 logits
 */
static void model_predict_logits(const float* weights,
                                 size_t vocab_size,
                                 const int* token_ids,
                                 size_t token_count,
                                 float* out_logits) {
    size_t i = 0U;
    size_t j = 0U;
    if (weights == NULL || token_ids == NULL || out_logits == NULL || token_count == 0U) {
        return;
    }
    for (j = 0U; j < OUTPUT_DIM; ++j) {
        out_logits[j] = 0.0f;
    }
    for (i = 0U; i < token_count; ++i) {
        size_t id = (token_ids[i] >= 0) ? (size_t)token_ids[i] : 0U;
        if (id >= vocab_size) {
            id = 0U;
        }
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            out_logits[j] += weights[id * OUTPUT_DIM + j];
        }
    }
    for (j = 0U; j < OUTPUT_DIM; ++j) {
        out_logits[j] /= (float)token_count;
    }
}

/**
 * @brief 对 logits 应用执行头激活，得到最终动作输出。
 *
 * @param logits   输入 logits
 * @param out_act  输出动作
 * @return int     0 成功，非 0 失败
 */
static int activate_actuators(const float* logits, float* out_act) {
    const int activations[OUTPUT_DIM] = IO_MAPPING_ACTIVATIONS;
    Tensor in_tensor;
    Tensor out_tensor;
    size_t shape[1] = { OUTPUT_DIM };
    int rc = 0;
    if (logits == NULL || out_act == NULL) {
        return -1;
    }
    rc = tensor_init_view(&in_tensor, (float*)logits, 1U, shape);
    if (rc != TENSOR_STATUS_OK) {
        return -2;
    }
    rc = tensor_init_view(&out_tensor, out_act, 1U, shape);
    if (rc != TENSOR_STATUS_OK) {
        return -3;
    }
    rc = op_actuator(&in_tensor, activations, &out_tensor);
    return (rc == TENSOR_STATUS_OK) ? 0 : -4;
}

/**
 * @brief 执行一次最小训练步（前向+均方误差+SGD）。
 *
 * @param weights      参数表
 * @param vocab_size   词表大小
 * @param token_ids    输入 token
 * @param token_count  token 数量
 * @param target       监督目标
 * @param lr           学习率
 * @param out_act      输出动作（用于驱动桩）
 * @param out_loss     输出 loss
 */
static void train_one_step(float* weights,
                           size_t vocab_size,
                           const int* token_ids,
                           size_t token_count,
                           const float* target,
                           float lr,
                           float* out_act,
                           float* out_loss) {
    float logits[OUTPUT_DIM];
    float pred[OUTPUT_DIM];
    float grad[OUTPUT_DIM];
    size_t i = 0U;
    size_t j = 0U;
    if (weights == NULL || token_ids == NULL || target == NULL || out_act == NULL || out_loss == NULL) {
        return;
    }

    model_predict_logits(weights, vocab_size, token_ids, token_count, logits);
    (void)activate_actuators(logits, pred);

    *out_loss = 0.0f;
    for (j = 0U; j < OUTPUT_DIM; ++j) {
        float err = pred[j] - target[j];
        grad[j] = err;
        *out_loss += err * err;
        out_act[j] = pred[j];
    }
    *out_loss /= (float)OUTPUT_DIM;

    /* 关键算法说明：
     * - 这里是最小可运行训练闭环，采用“按 token 平均分配梯度”的简单 SGD。
     * - 目标是验证训练/协议/驱动串联流程，不追求复杂优化器。 */
    for (i = 0U; i < token_count; ++i) {
        size_t id = (token_ids[i] >= 0) ? (size_t)token_ids[i] : 0U;
        if (id >= vocab_size) {
            id = 0U;
        }
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            weights[id * OUTPUT_DIM + j] -= (lr * grad[j]) / (float)token_count;
        }
    }
}

/**
 * @brief 最小训练闭环主程序入口。
 *
 * @param argc 参数数量
 * @param argv 参数数组
 * @return int 退出码，0 表示成功
 */
int main(int argc, char** argv) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    DriverStub pc_driver;
    DriverStub esp32_driver;
    TrainSample samples[4];
    size_t sample_count = 0U;
    float weights[VOCAB_SIZE * OUTPUT_DIM];
    size_t epoch = 0U;
    const size_t epochs = 5U;
    const float lr = 0.2f;
    int rc = 0;
    float* loaded_weights = NULL;
    size_t loaded_count = 0U;
    (void)argc;
    (void)argv;

    memset(&vocab, 0, sizeof(vocab));
    memset(&tokenizer, 0, sizeof(tokenizer));
    memset(&pc_driver, 0, sizeof(pc_driver));
    memset(&esp32_driver, 0, sizeof(esp32_driver));
    memset(weights, 0, sizeof(weights));

    rc = build_demo_vocab(&vocab);
    if (rc != TOKENIZER_STATUS_OK) {
        fprintf(stderr, "build_demo_vocab failed: %d\n", rc);
        return 1;
    }
    rc = tokenizer_init(&tokenizer, &vocab, 0);
    if (rc != TOKENIZER_STATUS_OK) {
        fprintf(stderr, "tokenizer_init failed: %d\n", rc);
        vocab_free(&vocab);
        return 1;
    }
    if (driver_stub_init(&pc_driver, DRIVER_TYPE_PC) != DRIVER_STATUS_OK ||
        driver_stub_init(&esp32_driver, DRIVER_TYPE_ESP32) != DRIVER_STATUS_OK) {
        fprintf(stderr, "driver init failed\n");
        vocab_free(&vocab);
        return 1;
    }

    build_demo_samples(samples, &sample_count);
    printf("start minimal train loop: samples=%zu, epochs=%zu\n", sample_count, epochs);

    for (epoch = 0U; epoch < epochs; ++epoch) {
        size_t i = 0U;
        float epoch_loss = 0.0f;
        for (i = 0U; i < sample_count; ++i) {
            int ids[16];
            size_t id_count = 0U;
            float act[OUTPUT_DIM];
            float loss = 0.0f;
            memset(ids, 0, sizeof(ids));
            memset(act, 0, sizeof(act));

            if (resolve_sample_tokens(&samples[i], &tokenizer, ids, 16U, &id_count) != 0) {
                fprintf(stderr, "resolve_sample_tokens failed at sample=%zu\n", i);
                driver_stub_shutdown(&pc_driver);
                driver_stub_shutdown(&esp32_driver);
                vocab_free(&vocab);
                return 2;
            }
            train_one_step(weights,
                           vocab.size,
                           ids,
                           id_count,
                           samples[i].target,
                           lr,
                           act,
                           &loss);
            epoch_loss += loss;

            if ((i % 2U) == 0U) {
                (void)driver_stub_apply(&pc_driver, act, OUTPUT_DIM);
            } else {
                (void)driver_stub_apply(&esp32_driver, act, OUTPUT_DIM);
            }
        }
        printf("epoch=%zu avg_loss=%.6f\n", epoch + 1U, (double)(epoch_loss / (float)sample_count));
    }

    rc = weights_save_binary("build/min_weights.bin", weights, VOCAB_SIZE * OUTPUT_DIM);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        fprintf(stderr, "weights_save_binary failed: %d\n", rc);
        driver_stub_shutdown(&pc_driver);
        driver_stub_shutdown(&esp32_driver);
        vocab_free(&vocab);
        return 3;
    }
    rc = weights_load_binary("build/min_weights.bin", &loaded_weights, &loaded_count);
    if (rc != WEIGHTS_IO_STATUS_OK || loaded_weights == NULL || loaded_count != (size_t)(VOCAB_SIZE * OUTPUT_DIM)) {
        fprintf(stderr, "weights_load_binary failed: %d count=%zu\n", rc, loaded_count);
        driver_stub_shutdown(&pc_driver);
        driver_stub_shutdown(&esp32_driver);
        vocab_free(&vocab);
        return 4;
    }
    rc = weights_export_c_source("build/weights.c", "g_model_weights", loaded_weights, loaded_count);
    free(loaded_weights);
    loaded_weights = NULL;
    if (rc != WEIGHTS_IO_STATUS_OK) {
        fprintf(stderr, "weights_export_c_source failed: %d\n", rc);
        driver_stub_shutdown(&pc_driver);
        driver_stub_shutdown(&esp32_driver);
        vocab_free(&vocab);
        return 5;
    }
    printf("weights exported: build/min_weights.bin and build/weights.c\n");

    driver_stub_shutdown(&pc_driver);
    driver_stub_shutdown(&esp32_driver);
    vocab_free(&vocab);
    printf("minimal train loop completed\n");
    return 0;
}
