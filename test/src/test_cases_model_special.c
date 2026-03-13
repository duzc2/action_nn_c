#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "../include/test_cases.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../../src/include/config_user.h"
#include "../../src/include/ops.h"
#include "../../src/include/protocol.h"
#include "../../src/include/tensor.h"
#include "../../src/include/tokenizer.h"

/**
 * @brief 模型专项测试样本。
 *
 * 设计说明：
 * - 输入由 token 序列与连续状态组成，模拟“离散指令 + 连续控制状态”的融合场景。
 * - target 维持与 OUTPUT_DIM 一致，便于直接复用 actuator 输出维度。
 */
typedef struct ModelSpecialSample {
    int ids[8];
    size_t token_count;
    float state[STATE_DIM];
    float target[OUTPUT_DIM];
} ModelSpecialSample;

/**
 * @brief 模型评估指标。
 *
 * 关键指标：
 * - mse / mae_x / mae_y：回归误差。
 * - dir_acc：方向符号一致率，衡量决策一致性。
 * - max_abs_out：输出边界检查，约束在激活函数定义域附近。
 */
typedef struct ModelEvalMetrics {
    float mse;
    float mae_x;
    float mae_y;
    double dir_acc;
    float max_abs_out;
    size_t dir_hit;
    size_t dir_total;
} ModelEvalMetrics;

enum {
    MS_TOKEN_WEIGHT_COUNT = VOCAB_SIZE * OUTPUT_DIM,
    MS_STATE_WEIGHT_COUNT = STATE_DIM * OUTPUT_DIM,
    MS_BIAS_COUNT = OUTPUT_DIM,
    MS_TOTAL_WEIGHT_COUNT = MS_TOKEN_WEIGHT_COUNT + MS_STATE_WEIGHT_COUNT + MS_BIAS_COUNT
};

/**
 * @brief 生成可复现的伪随机序列。
 *
 * @param seed 随机种子（输入输出）
 * @return unsigned int 下一随机值
 */
static unsigned int ms_lcg_next(unsigned int* seed) {
    *seed = (*seed * 1664525U) + 1013904223U;
    return *seed;
}

/**
 * @brief 生成区间随机数。
 *
 * @param seed 随机种子（输入输出）
 * @param lo   下界
 * @param hi   上界
 * @return float 区间随机值
 */
static float ms_rand_range(unsigned int* seed, float lo, float hi) {
    unsigned int v = ms_lcg_next(seed);
    float t = (float)(v & 0x00FFFFFFU) / (float)0x01000000U;
    return lo + (hi - lo) * t;
}

/**
 * @brief 浮点裁剪函数，防止状态值越界。
 *
 * @param v  输入值
 * @param lo 下界
 * @param hi 上界
 * @return float 裁剪后数值
 */
static float ms_clamp(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

/**
 * @brief 初始化专项测试词表与 tokenizer。
 *
 * @param vocab     输出词表
 * @param tokenizer 输出 tokenizer
 * @return int TokenizerStatus
 */
static int ms_init_vocab_tokenizer(Vocabulary* vocab, Tokenizer* tokenizer) {
    static const char* tokens[] = {"<unk>", "move", "left", "right", "stop", "fast", "slow"};
    size_t i = 0U;
    if (vocab == NULL || tokenizer == NULL) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    if (vocab_init(vocab, VOCAB_SIZE) != TOKENIZER_STATUS_OK) {
        return TOKENIZER_STATUS_OOM;
    }
    for (i = 0U; i < sizeof(tokens) / sizeof(tokens[0]); ++i) {
        if (vocab_add_token(vocab, tokens[i], NULL) != TOKENIZER_STATUS_OK) {
            vocab_free(vocab);
            return TOKENIZER_STATUS_OOM;
        }
    }
    if (tokenizer_init(tokenizer, vocab, 0) != TOKENIZER_STATUS_OK) {
        vocab_free(vocab);
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    return TOKENIZER_STATUS_OK;
}

/**
 * @brief 初始化模型参数，保证训练前状态确定。
 *
 * @param weights 参数数组
 * @param count   参数数量
 */
static void ms_init_weights(float* weights, size_t count) {
    size_t i = 0U;
    for (i = 0U; i < count; ++i) {
        weights[i] = ((float)((int)(i % 17U) - 8)) * 0.0015f;
    }
}

/**
 * @brief 前向计算 logits。
 *
 * @param weights     参数数组
 * @param token_ids   token 序列
 * @param token_count token 数量
 * @param state       状态向量
 * @param out_logits  输出 logits
 */
static void ms_predict_logits(const float* weights,
                              const int* token_ids,
                              size_t token_count,
                              const float* state,
                              float* out_logits) {
    const size_t token_base = 0U;
    const size_t state_base = MS_TOKEN_WEIGHT_COUNT;
    const size_t bias_base = MS_TOKEN_WEIGHT_COUNT + MS_STATE_WEIGHT_COUNT;
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

/**
 * @brief 执行输出激活映射。
 *
 * @param logits 输入 logits
 * @param out_act 输出激活值
 * @return int 0=成功，非0=失败
 */
static int ms_activate(const float* logits, float* out_act) {
    const int activations[OUTPUT_DIM] = IO_MAPPING_ACTIVATIONS;
    Tensor in;
    Tensor out;
    size_t shape[1] = {OUTPUT_DIM};
    if (tensor_init_view(&in, (float*)logits, 1U, shape) != TENSOR_STATUS_OK) {
        return 1;
    }
    if (tensor_init_view(&out, out_act, 1U, shape) != TENSOR_STATUS_OK) {
        return 2;
    }
    if (op_actuator(&in, activations, &out) != TENSOR_STATUS_OK) {
        return 3;
    }
    return 0;
}

/**
 * @brief 根据状态随机生成训练/评估样本。
 *
 * @param tokenizer tokenizer
 * @param seed      随机种子
 * @param out       输出样本
 * @param range_xy  坐标范围绝对值上限
 * @return int 0=成功，非0=失败
 */
static int ms_make_sample(const Tokenizer* tokenizer,
                          unsigned int* seed,
                          ModelSpecialSample* out,
                          float range_xy) {
    float x = ms_rand_range(seed, -range_xy, range_xy);
    float y = ms_rand_range(seed, -range_xy, range_xy);
    float tx = ms_rand_range(seed, -range_xy, range_xy);
    float ty = ms_rand_range(seed, -range_xy, range_xy);
    float dx = tx - x;
    float dy = ty - y;
    const char* cmd = NULL;
    if (fabsf(dx) < 0.2f && fabsf(dy) < 0.2f) {
        cmd = "move stop";
    } else if (dx > 0.0f) {
        cmd = (fabsf(dx) > 3.0f) ? "move right fast" : "move right slow";
    } else {
        cmd = (fabsf(dx) > 3.0f) ? "move left fast" : "move left slow";
    }
    memset(out, 0, sizeof(*out));
    out->state[0] = x / 20.0f;
    out->state[1] = y / 20.0f;
    out->state[2] = dx / 20.0f;
    out->state[3] = dy / 20.0f;
    out->state[4] = ms_clamp((fabsf(dx) + fabsf(dy)) / 40.0f, 0.0f, 1.0f);
    out->state[5] = ms_clamp(fabsf(dx) / 20.0f, 0.0f, 1.0f);
    out->state[6] = ms_clamp(fabsf(dy) / 20.0f, 0.0f, 1.0f);
    out->state[7] = 1.0f;
    if (tokenizer_encode(tokenizer, cmd, out->ids, 8U, &out->token_count) != TOKENIZER_STATUS_OK) {
        return 1;
    }
    out->target[0] = (dx < -0.2f) ? 1.0f : 0.0f;
    out->target[1] = (dx > 0.2f) ? 1.0f : 0.0f;
    out->target[2] = ms_clamp(dx / 6.0f, -1.0f, 1.0f);
    out->target[3] = ms_clamp(dy / 6.0f, -1.0f, 1.0f);
    return 0;
}

/**
 * @brief 单样本训练步。
 *
 * @param weights 模型参数
 * @param sample  样本
 * @param lr      学习率
 * @param out_loss 输出 loss
 * @return int 0=成功，非0=失败
 */
static int ms_train_step(float* weights, const ModelSpecialSample* sample, float lr, float* out_loss) {
    const size_t token_base = 0U;
    const size_t state_base = MS_TOKEN_WEIGHT_COUNT;
    const size_t bias_base = MS_TOKEN_WEIGHT_COUNT + MS_STATE_WEIGHT_COUNT;
    float logits[OUTPUT_DIM] = {0.0f};
    float pred[OUTPUT_DIM] = {0.0f};
    float grad[OUTPUT_DIM] = {0.0f};
    size_t i = 0U;
    size_t j = 0U;
    if (sample == NULL || sample->token_count == 0U || out_loss == NULL) {
        return 1;
    }
    ms_predict_logits(weights, sample->ids, sample->token_count, sample->state, logits);
    if (ms_activate(logits, pred) != 0) {
        return 2;
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

/**
 * @brief 批量训练指定轮数。
 *
 * @param weights 参数
 * @param train_set 训练集
 * @param train_n 训练样本数量
 * @param epochs 轮数
 * @param lr 学习率
 * @return int 0=成功，非0=失败
 */
static int ms_train_model(float* weights,
                          const ModelSpecialSample* train_set,
                          size_t train_n,
                          size_t epochs,
                          float lr) {
    size_t e = 0U;
    size_t i = 0U;
    for (e = 0U; e < epochs; ++e) {
        for (i = 0U; i < train_n; ++i) {
            float loss = 0.0f;
            if (ms_train_step(weights, &train_set[i], lr, &loss) != 0) {
                return 1;
            }
        }
    }
    return 0;
}

/**
 * @brief 评估样本集合并聚合指标。
 *
 * @param weights 参数
 * @param set 样本集合
 * @param n 样本数量
 * @param out 输出指标
 * @return int 0=成功，非0=失败
 */
static int ms_eval_metrics(const float* weights,
                           const ModelSpecialSample* set,
                           size_t n,
                           ModelEvalMetrics* out) {
    size_t i = 0U;
    memset(out, 0, sizeof(*out));
    for (i = 0U; i < n; ++i) {
        float logits[OUTPUT_DIM] = {0.0f};
        float act[OUTPUT_DIM] = {0.0f};
        float err_x = 0.0f;
        float err_y = 0.0f;
        ms_predict_logits(weights, set[i].ids, set[i].token_count, set[i].state, logits);
        if (ms_activate(logits, act) != 0) {
            return 1;
        }
        err_x = act[2] - set[i].target[2];
        err_y = act[3] - set[i].target[3];
        out->mse += (err_x * err_x + err_y * err_y) * 0.5f;
        out->mae_x += fabsf(err_x);
        out->mae_y += fabsf(err_y);
        if (fabsf(act[2]) > out->max_abs_out) {
            out->max_abs_out = fabsf(act[2]);
        }
        if (fabsf(act[3]) > out->max_abs_out) {
            out->max_abs_out = fabsf(act[3]);
        }
        if (fabsf(set[i].target[2]) > 0.10f) {
            float pred_sign = (act[2] > 0.02f) ? 1.0f : ((act[2] < -0.02f) ? -1.0f : 0.0f);
            float gt_sign = (set[i].target[2] > 0.0f) ? 1.0f : -1.0f;
            out->dir_total += 1U;
            if ((pred_sign > 0.0f && gt_sign > 0.0f) || (pred_sign < 0.0f && gt_sign < 0.0f)) {
                out->dir_hit += 1U;
            }
        }
    }
    out->mse /= (float)n;
    out->mae_x /= (float)n;
    out->mae_y /= (float)n;
    out->dir_acc = (out->dir_total > 0U) ? ((double)out->dir_hit / (double)out->dir_total) : 0.0;
    return 0;
}

/**
 * @brief 一致性测试辅助：反转 token 顺序，不改变 token 多重集合。
 *
 * @param sample 样本（输入输出）
 */
static void ms_reverse_tokens(ModelSpecialSample* sample) {
    size_t i = 0U;
    for (i = 0U; i < sample->token_count / 2U; ++i) {
        int t = sample->ids[i];
        sample->ids[i] = sample->ids[sample->token_count - 1U - i];
        sample->ids[sample->token_count - 1U - i] = t;
    }
}

static const char* ms_select_control_command(float dx, float dy) {
    if (fabsf(dx) < 0.30f && fabsf(dy) < 0.30f) {
        return "move stop";
    }
    if (dx >= 0.0f) {
        return (fabsf(dx) > 2.0f) ? "move right fast" : "move right slow";
    }
    return (fabsf(dx) > 2.0f) ? "move left fast" : "move left slow";
}

static void ms_build_control_state(float x, float y, float gx, float gy, float* out_state) {
    float dx = gx - x;
    float dy = gy - y;
    out_state[0] = x / 20.0f;
    out_state[1] = y / 20.0f;
    out_state[2] = dx / 20.0f;
    out_state[3] = dy / 20.0f;
    out_state[4] = ms_clamp((fabsf(dx) + fabsf(dy)) / 40.0f, 0.0f, 1.0f);
    out_state[5] = ms_clamp(fabsf(dx) / 20.0f, 0.0f, 1.0f);
    out_state[6] = ms_clamp(fabsf(dy) / 20.0f, 0.0f, 1.0f);
    out_state[7] = 1.0f;
}

static int ms_infer_action(const float* weights,
                           const Tokenizer* tokenizer,
                           const char* command,
                           const float* state,
                           float* out_act) {
    int ids[8] = {0};
    size_t token_count = 0U;
    float logits[OUTPUT_DIM] = {0.0f};
    if (tokenizer_encode(tokenizer, command, ids, 8U, &token_count) != TOKENIZER_STATUS_OK || token_count == 0U) {
        return 1;
    }
    ms_predict_logits(weights, ids, token_count, state, logits);
    if (ms_activate(logits, out_act) != 0) {
        return 2;
    }
    return 0;
}

static float ms_clamp_step(float step, float remain) {
    if (fabsf(step) > fabsf(remain)) {
        return remain;
    }
    return step;
}

static void ms_apply_control_step(float* x, float* y, float gx, float gy, const float* act) {
    float dx = gx - *x;
    float dy = gy - *y;
    float sx = act[2] * 0.55f;
    float sy = 0.0f;
    if (fabsf(sx) < 0.08f && fabsf(dx) > 0.5f) {
        sx = (dx > 0.0f) ? 0.10f : -0.10f;
    }
    sx = ms_clamp_step(sx, dx);
    if (dy > 0.0f) {
        sy = ((act[3] + 1.0f) * 0.5f) * 0.42f;
        if (sy < 0.05f) {
            sy = 0.05f;
        }
        sy = ms_clamp_step(sy, dy);
    } else if (dy < 0.0f) {
        sy = -0.05f;
        sy = ms_clamp_step(sy, dy);
    }
    *x += sx;
    *y += sy;
}

/**
 * @brief 泛化测试：同分布插值区间性能验证。
 */
static int test_model_generalization_interpolation(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[280];
    ModelSpecialSample eval_set[220];
    ModelEvalMetrics metrics;
    float weights[MS_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 20260314U;
    size_t i = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 280U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 10.0f));
    }
    for (i = 0U; i < 220U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &eval_set[i], 10.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 280U, 22U, 0.05f));
    TFW_ASSERT_INT_EQ(0, ms_eval_metrics(weights, eval_set, 220U, &metrics));
    testfw_log_info("model/generalization/interpolation: mse=%.4f mae_x=%.4f mae_y=%.4f dir=%.3f max_abs=%.4f",
                    (double)metrics.mse,
                    (double)metrics.mae_x,
                    (double)metrics.mae_y,
                    metrics.dir_acc,
                    (double)metrics.max_abs_out);
    TFW_ASSERT_TRUE(metrics.mse < 0.17f);
    TFW_ASSERT_TRUE(metrics.mae_x < 0.36f);
    TFW_ASSERT_TRUE(metrics.mae_y < 0.38f);
    TFW_ASSERT_TRUE(metrics.dir_acc > 0.74);
    TFW_ASSERT_TRUE(metrics.max_abs_out <= 1.01f);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 泛化测试：跨区间外推性能验证。
 */
static int test_model_generalization_cross_range(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[240];
    ModelSpecialSample eval_set[200];
    ModelEvalMetrics metrics;
    float weights[MS_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 20260315U;
    size_t i = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 240U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 8.0f));
    }
    for (i = 0U; i < 200U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &eval_set[i], 14.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 240U, 22U, 0.055f));
    TFW_ASSERT_INT_EQ(0, ms_eval_metrics(weights, eval_set, 200U, &metrics));
    testfw_log_info("model/generalization/cross_range: mse=%.4f mae_x=%.4f mae_y=%.4f dir=%.3f max_abs=%.4f",
                    (double)metrics.mse,
                    (double)metrics.mae_x,
                    (double)metrics.mae_y,
                    metrics.dir_acc,
                    (double)metrics.max_abs_out);
    TFW_ASSERT_TRUE(metrics.mse < 0.26f);
    TFW_ASSERT_TRUE(metrics.mae_x < 0.46f);
    TFW_ASSERT_TRUE(metrics.mae_y < 0.48f);
    TFW_ASSERT_TRUE(metrics.dir_acc > 0.64);
    TFW_ASSERT_TRUE(metrics.max_abs_out <= 1.01f);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief OOD 测试：远分布输入下边界与误差阈值验证。
 */
static int test_model_ood_far_range_bounds(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[220];
    ModelSpecialSample eval_set[220];
    ModelEvalMetrics metrics;
    float weights[MS_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 20260316U;
    size_t i = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 220U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 8.0f));
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &eval_set[i], 20.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 220U, 20U, 0.055f));
    TFW_ASSERT_INT_EQ(0, ms_eval_metrics(weights, eval_set, 220U, &metrics));
    testfw_log_info("model/ood/far_range: mse=%.4f mae_x=%.4f mae_y=%.4f dir=%.3f max_abs=%.4f",
                    (double)metrics.mse,
                    (double)metrics.mae_x,
                    (double)metrics.mae_y,
                    metrics.dir_acc,
                    (double)metrics.max_abs_out);
    TFW_ASSERT_TRUE(metrics.mse < 0.42f);
    TFW_ASSERT_TRUE(metrics.mae_x < 0.62f);
    TFW_ASSERT_TRUE(metrics.mae_y < 0.64f);
    TFW_ASSERT_TRUE(metrics.dir_acc > 0.52);
    TFW_ASSERT_TRUE(metrics.max_abs_out <= 1.01f);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 对抗扰动测试：状态小扰动下方向翻转率与误差增量限制。
 */
static int test_model_adversarial_state_perturbation(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[240];
    ModelSpecialSample eval_set[180];
    float weights[MS_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 20260317U;
    float mae_base = 0.0f;
    float mae_adv = 0.0f;
    size_t i = 0U;
    size_t d = 0U;
    size_t flip = 0U;
    size_t total = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 240U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 10.0f));
    }
    for (i = 0U; i < 180U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &eval_set[i], 12.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 240U, 20U, 0.05f));
    for (i = 0U; i < 180U; ++i) {
        float base_logits[OUTPUT_DIM] = {0.0f};
        float base_act[OUTPUT_DIM] = {0.0f};
        float adv_state[STATE_DIM];
        float adv_logits[OUTPUT_DIM] = {0.0f};
        float adv_act[OUTPUT_DIM] = {0.0f};
        memcpy(adv_state, eval_set[i].state, sizeof(adv_state));
        ms_predict_logits(weights, eval_set[i].ids, eval_set[i].token_count, eval_set[i].state, base_logits);
        TFW_ASSERT_INT_EQ(0, ms_activate(base_logits, base_act));
        for (d = 0U; d < STATE_DIM; ++d) {
            float p_state[STATE_DIM];
            float n_state[STATE_DIM];
            float p_logits[OUTPUT_DIM] = {0.0f};
            float n_logits[OUTPUT_DIM] = {0.0f};
            float p_act[OUTPUT_DIM] = {0.0f};
            float n_act[OUTPUT_DIM] = {0.0f};
            float p_err = 0.0f;
            float n_err = 0.0f;
            memcpy(p_state, eval_set[i].state, sizeof(p_state));
            memcpy(n_state, eval_set[i].state, sizeof(n_state));
            p_state[d] = ms_clamp(p_state[d] + 0.06f, -2.0f, 2.0f);
            n_state[d] = ms_clamp(n_state[d] - 0.06f, -2.0f, 2.0f);
            ms_predict_logits(weights, eval_set[i].ids, eval_set[i].token_count, p_state, p_logits);
            ms_predict_logits(weights, eval_set[i].ids, eval_set[i].token_count, n_state, n_logits);
            TFW_ASSERT_INT_EQ(0, ms_activate(p_logits, p_act));
            TFW_ASSERT_INT_EQ(0, ms_activate(n_logits, n_act));
            p_err = fabsf(p_act[2] - eval_set[i].target[2]);
            n_err = fabsf(n_act[2] - eval_set[i].target[2]);
            adv_state[d] = (p_err >= n_err) ? p_state[d] : n_state[d];
        }
        ms_predict_logits(weights, eval_set[i].ids, eval_set[i].token_count, adv_state, adv_logits);
        TFW_ASSERT_INT_EQ(0, ms_activate(adv_logits, adv_act));
        mae_base += fabsf(base_act[2] - eval_set[i].target[2]);
        mae_adv += fabsf(adv_act[2] - eval_set[i].target[2]);
        if (fabsf(eval_set[i].target[2]) > 0.12f) {
            float s0 = (base_act[2] > 0.02f) ? 1.0f : ((base_act[2] < -0.02f) ? -1.0f : 0.0f);
            float s1 = (adv_act[2] > 0.02f) ? 1.0f : ((adv_act[2] < -0.02f) ? -1.0f : 0.0f);
            total += 1U;
            if ((s0 > 0.0f && s1 < 0.0f) || (s0 < 0.0f && s1 > 0.0f)) {
                flip += 1U;
            }
        }
        TFW_ASSERT_TRUE(fabsf(adv_act[2]) <= 1.01f);
        TFW_ASSERT_TRUE(fabsf(adv_act[3]) <= 1.01f);
    }
    mae_base /= 180.0f;
    mae_adv /= 180.0f;
    testfw_log_info("model/adversarial/state: mae_base=%.4f mae_adv=%.4f flip=%zu/%zu",
                    (double)mae_base,
                    (double)mae_adv,
                    flip,
                    total);
    TFW_ASSERT_TRUE(mae_adv - mae_base < 0.28f);
    TFW_ASSERT_TRUE(total > 90U);
    TFW_ASSERT_TRUE((double)flip / (double)total < 0.30);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 对抗扰动测试：速度 token 翻转对方向输出应保持一致。
 */
static int test_model_adversarial_token_speed_flip(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[220];
    ModelSpecialSample eval_set[180];
    float weights[MS_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 20260318U;
    int id_fast = 0;
    int id_slow = 0;
    size_t i = 0U;
    size_t consistent = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_find_id(&vocab, "fast", &id_fast));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_find_id(&vocab, "slow", &id_slow));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 220U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 10.0f));
    }
    for (i = 0U; i < 180U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &eval_set[i], 14.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 220U, 20U, 0.05f));
    for (i = 0U; i < 180U; ++i) {
        ModelSpecialSample modified = eval_set[i];
        float a_logits[OUTPUT_DIM] = {0.0f};
        float b_logits[OUTPUT_DIM] = {0.0f};
        float a_act[OUTPUT_DIM] = {0.0f};
        float b_act[OUTPUT_DIM] = {0.0f};
        size_t k = 0U;
        for (k = 0U; k < modified.token_count; ++k) {
            if (modified.ids[k] == id_fast) {
                modified.ids[k] = id_slow;
            } else if (modified.ids[k] == id_slow) {
                modified.ids[k] = id_fast;
            }
        }
        ms_predict_logits(weights, eval_set[i].ids, eval_set[i].token_count, eval_set[i].state, a_logits);
        ms_predict_logits(weights, modified.ids, modified.token_count, modified.state, b_logits);
        TFW_ASSERT_INT_EQ(0, ms_activate(a_logits, a_act));
        TFW_ASSERT_INT_EQ(0, ms_activate(b_logits, b_act));
        if ((a_act[2] >= 0.0f && b_act[2] >= 0.0f) || (a_act[2] < 0.0f && b_act[2] < 0.0f)) {
            consistent += 1U;
        }
    }
    testfw_log_info("model/adversarial/token_flip: consistent=%zu/%d", consistent, 180);
    TFW_ASSERT_TRUE((double)consistent / 180.0 > 0.88);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 稳定性测试：同输入重复推理必须数值稳定且确定。
 */
static int test_model_stability_repeat_inference(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[200];
    ModelSpecialSample sample;
    float weights[MS_TOTAL_WEIGHT_COUNT];
    float ref[OUTPUT_DIM] = {0.0f};
    unsigned int seed = 20260319U;
    size_t i = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 200U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 10.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 200U, 18U, 0.05f));
    TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &sample, 10.0f));
    for (i = 0U; i < 1200U; ++i) {
        float logits[OUTPUT_DIM] = {0.0f};
        float act[OUTPUT_DIM] = {0.0f};
        ms_predict_logits(weights, sample.ids, sample.token_count, sample.state, logits);
        TFW_ASSERT_INT_EQ(0, ms_activate(logits, act));
        if (i == 0U) {
            memcpy(ref, act, sizeof(ref));
        } else {
            TFW_ASSERT_FLOAT_NEAR(ref[0], act[0], 1e-7f);
            TFW_ASSERT_FLOAT_NEAR(ref[1], act[1], 1e-7f);
            TFW_ASSERT_FLOAT_NEAR(ref[2], act[2], 1e-7f);
            TFW_ASSERT_FLOAT_NEAR(ref[3], act[3], 1e-7f);
        }
    }
    testfw_log_info("model/stability/repeat_inference: repeat=1200 max_diff<=1e-7");
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 一致性测试：token 顺序变化（同多重集合）输出一致。
 */
static int test_model_consistency_token_order_invariance(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[220];
    ModelSpecialSample eval_set[140];
    float weights[MS_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 20260320U;
    size_t i = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 220U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 10.0f));
    }
    for (i = 0U; i < 140U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &eval_set[i], 12.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 220U, 20U, 0.05f));
    for (i = 0U; i < 140U; ++i) {
        ModelSpecialSample reversed = eval_set[i];
        float logits_a[OUTPUT_DIM] = {0.0f};
        float logits_b[OUTPUT_DIM] = {0.0f};
        float act_a[OUTPUT_DIM] = {0.0f};
        float act_b[OUTPUT_DIM] = {0.0f};
        ms_reverse_tokens(&reversed);
        ms_predict_logits(weights, eval_set[i].ids, eval_set[i].token_count, eval_set[i].state, logits_a);
        ms_predict_logits(weights, reversed.ids, reversed.token_count, reversed.state, logits_b);
        TFW_ASSERT_INT_EQ(0, ms_activate(logits_a, act_a));
        TFW_ASSERT_INT_EQ(0, ms_activate(logits_b, act_b));
        TFW_ASSERT_FLOAT_NEAR(act_a[2], act_b[2], 1e-6f);
        TFW_ASSERT_FLOAT_NEAR(act_a[3], act_b[3], 1e-6f);
    }
    testfw_log_info("model/consistency/token_order: checked=140");
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 一致性测试：协议编解码前后 token 序列不变，模型输出一致。
 */
static int test_model_consistency_protocol_roundtrip(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[220];
    ModelSpecialSample sample;
    float weights[MS_TOTAL_WEIGHT_COUNT];
    char packet[256];
    ProtocolFrame frame;
    char raw[64];
    int token_buf[16];
    unsigned int seed = 20260321U;
    size_t i = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 220U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 10.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 220U, 20U, 0.05f));
    TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &sample, 12.0f));
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_OK,
                      protocol_encode_token(sample.ids, sample.token_count, packet, sizeof(packet), NULL));
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_OK,
                      protocol_decode_packet(packet, &frame, raw, sizeof(raw), token_buf, 16U));
    TFW_ASSERT_SIZE_EQ(sample.token_count, frame.token_count);
    {
        float logits_a[OUTPUT_DIM] = {0.0f};
        float logits_b[OUTPUT_DIM] = {0.0f};
        float act_a[OUTPUT_DIM] = {0.0f};
        float act_b[OUTPUT_DIM] = {0.0f};
        ms_predict_logits(weights, sample.ids, sample.token_count, sample.state, logits_a);
        ms_predict_logits(weights, frame.token_ids, frame.token_count, sample.state, logits_b);
        TFW_ASSERT_INT_EQ(0, ms_activate(logits_a, act_a));
        TFW_ASSERT_INT_EQ(0, ms_activate(logits_b, act_b));
        TFW_ASSERT_FLOAT_NEAR(act_a[2], act_b[2], 1e-7f);
        TFW_ASSERT_FLOAT_NEAR(act_a[3], act_b[3], 1e-7f);
    }
    testfw_log_info("model/consistency/protocol_roundtrip: token_count=%zu", sample.token_count);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 一致性测试：固定种子重复训练，参数应可复现。
 */
static int test_model_consistency_seed_reproducibility(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[200];
    float w1[MS_TOTAL_WEIGHT_COUNT];
    float w2[MS_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 20260322U;
    size_t i = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    for (i = 0U; i < 200U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 10.0f));
    }
    ms_init_weights(w1, MS_TOTAL_WEIGHT_COUNT);
    ms_init_weights(w2, MS_TOTAL_WEIGHT_COUNT);
    TFW_ASSERT_INT_EQ(0, ms_train_model(w1, train_set, 200U, 18U, 0.05f));
    TFW_ASSERT_INT_EQ(0, ms_train_model(w2, train_set, 200U, 18U, 0.05f));
    for (i = 0U; i < MS_TOTAL_WEIGHT_COUNT; ++i) {
        TFW_ASSERT_FLOAT_NEAR(w1[i], w2[i], 1e-7f);
    }
    testfw_log_info("model/consistency/seed_reproducibility: weight_count=%d", MS_TOTAL_WEIGHT_COUNT);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 极限长时序闭环：10k 帧多目标循环，验证长期漂移与稳定收敛。
 */
static int test_model_extreme_long_horizon_closed_loop(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[320];
    float weights[MS_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 20260323U;
    const float goals[][2] = {
        {15.0f, 15.0f}, {-12.0f, 10.0f}, {8.0f, -9.0f}, {0.0f, 0.0f}, {14.0f, -14.0f}, {-15.0f, -15.0f}
    };
    size_t i = 0U;
    size_t frame = 0U;
    size_t goal_idx = 0U;
    size_t reached = 0U;
    float x = 0.0f;
    float y = 0.0f;
    float dist_sum = 0.0f;
    float dist_max = 0.0f;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 320U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 12.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 320U, 24U, 0.05f));
    for (frame = 0U; frame < 10000U; ++frame) {
        float gx = goals[goal_idx][0];
        float gy = goals[goal_idx][1];
        float state[STATE_DIM];
        float act[OUTPUT_DIM] = {0.0f};
        float dx = gx - x;
        float dy = gy - y;
        float dist = sqrtf(dx * dx + dy * dy);
        const char* cmd = NULL;
        if (dist < 0.45f) {
            reached += 1U;
            goal_idx = (goal_idx + 1U) % (sizeof(goals) / sizeof(goals[0]));
            gx = goals[goal_idx][0];
            gy = goals[goal_idx][1];
        }
        cmd = ms_select_control_command(gx - x, gy - y);
        ms_build_control_state(x, y, gx, gy, state);
        TFW_ASSERT_INT_EQ(0, ms_infer_action(weights, &tokenizer, cmd, state, act));
        ms_apply_control_step(&x, &y, gx, gy, act);
        dx = gx - x;
        dy = gy - y;
        dist = sqrtf(dx * dx + dy * dy);
        dist_sum += dist;
        if (dist > dist_max) {
            dist_max = dist;
        }
        TFW_ASSERT_TRUE(fabsf(x) < 40.0f);
        TFW_ASSERT_TRUE(fabsf(y) < 40.0f);
    }
    testfw_log_info("model/extreme/long_horizon: reached=%zu avg_dist=%.4f max_dist=%.4f final=(%.3f,%.3f)",
                    reached,
                    (double)(dist_sum / 10000.0f),
                    (double)dist_max,
                    (double)x,
                    (double)y);
    TFW_ASSERT_TRUE(reached >= 55U);
    TFW_ASSERT_TRUE((dist_sum / 10000.0f) < 12.0f);
    TFW_ASSERT_TRUE(dist_max < 45.0f);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 极限切换：高频多目标切换，验证重新规划与切换抖动控制。
 */
static int test_model_extreme_multi_goal_switching(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[300];
    float weights[MS_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 20260324U;
    size_t i = 0U;
    size_t frame = 0U;
    size_t switches = 0U;
    size_t near_target = 0U;
    float x = -3.0f;
    float y = 2.0f;
    float gx = 10.0f;
    float gy = 10.0f;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 300U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 12.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 300U, 22U, 0.05f));
    for (frame = 0U; frame < 2400U; ++frame) {
        float state[STATE_DIM];
        float act[OUTPUT_DIM] = {0.0f};
        float dx = gx - x;
        float dy = gy - y;
        float dist = sqrtf(dx * dx + dy * dy);
        const char* cmd = NULL;
        if ((frame % 120U) == 0U) {
            gx = ms_rand_range(&seed, -15.0f, 15.0f);
            gy = ms_rand_range(&seed, -15.0f, 15.0f);
            switches += 1U;
        }
        if (dist < 0.70f) {
            near_target += 1U;
        }
        cmd = ms_select_control_command(gx - x, gy - y);
        ms_build_control_state(x, y, gx, gy, state);
        TFW_ASSERT_INT_EQ(0, ms_infer_action(weights, &tokenizer, cmd, state, act));
        ms_apply_control_step(&x, &y, gx, gy, act);
        TFW_ASSERT_TRUE(fabsf(x) < 45.0f);
        TFW_ASSERT_TRUE(fabsf(y) < 45.0f);
    }
    testfw_log_info("model/extreme/multi_goal_switching: switches=%zu near_target=%zu final=(%.3f,%.3f)",
                    switches,
                    near_target,
                    (double)x,
                    (double)y);
    TFW_ASSERT_TRUE(switches >= 18U);
    TFW_ASSERT_TRUE(near_target >= 320U);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 极限鲁棒：观测延迟与噪声联合注入，验证闭环仍能到达目标并保持有界。
 */
static int test_model_extreme_delay_noise_joint(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[320];
    float weights[MS_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 20260325U;
    float x = 0.0f;
    float y = 0.0f;
    const float gx = 15.0f;
    const float gy = 15.0f;
    float hist[4][STATE_DIM];
    size_t frame = 0U;
    size_t stable = 0U;
    size_t i = 0U;
    memset(&vocab, 0, sizeof(vocab));
    memset(hist, 0, sizeof(hist));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 320U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 12.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 320U, 24U, 0.05f));
    for (frame = 0U; frame < 2600U; ++frame) {
        float cur_state[STATE_DIM];
        float delayed[STATE_DIM];
        float act[OUTPUT_DIM] = {0.0f};
        const char* cmd = NULL;
        size_t d = 0U;
        ms_build_control_state(x, y, gx, gy, cur_state);
        for (d = 0U; d < STATE_DIM; ++d) {
            hist[frame % 4U][d] = cur_state[d];
        }
        for (d = 0U; d < STATE_DIM; ++d) {
            float noise = ms_rand_range(&seed, -0.025f, 0.025f);
            delayed[d] = ms_clamp(hist[(frame + 2U) % 4U][d] + noise, -1.8f, 1.8f);
        }
        cmd = ms_select_control_command(gx - x, gy - y);
        TFW_ASSERT_INT_EQ(0, ms_infer_action(weights, &tokenizer, cmd, delayed, act));
        ms_apply_control_step(&x, &y, gx, gy, act);
        if (fabsf(gx - x) < 0.8f && fabsf(gy - y) < 0.8f) {
            stable += 1U;
        }
        TFW_ASSERT_TRUE(fabsf(x) < 35.0f);
        TFW_ASSERT_TRUE(fabsf(y) < 35.0f);
    }
    testfw_log_info("model/extreme/delay_noise_joint: stable=%zu final=(%.3f,%.3f) remain=(%.3f,%.3f)",
                    stable,
                    (double)x,
                    (double)y,
                    (double)(gx - x),
                    (double)(gy - y));
    TFW_ASSERT_TRUE(stable >= 350U);
    TFW_ASSERT_TRUE(fabsf(gx - x) < 1.2f);
    TFW_ASSERT_TRUE(fabsf(gy - y) < 1.2f);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 故障注入恢复：注入掉帧/错误命令/动作冻结，验证系统恢复能力。
 */
static int test_model_extreme_fault_injection_recovery(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    ModelSpecialSample train_set[320];
    float weights[MS_TOTAL_WEIGHT_COUNT];
    unsigned int seed = 20260326U;
    float x = -10.0f;
    float y = -10.0f;
    const float gx = 12.0f;
    const float gy = 12.0f;
    size_t frame = 0U;
    size_t fault_hits = 0U;
    size_t recovered = 0U;
    float frozen_act[OUTPUT_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};
    size_t i = 0U;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < 320U; ++i) {
        TFW_ASSERT_INT_EQ(0, ms_make_sample(&tokenizer, &seed, &train_set[i], 12.0f));
    }
    TFW_ASSERT_INT_EQ(0, ms_train_model(weights, train_set, 320U, 24U, 0.05f));
    for (frame = 0U; frame < 3200U; ++frame) {
        float state[STATE_DIM];
        float act[OUTPUT_DIM] = {0.0f};
        const char* cmd = ms_select_control_command(gx - x, gy - y);
        float r = ms_rand_range(&seed, 0.0f, 1.0f);
        ms_build_control_state(x, y, gx, gy, state);
        if (r < 0.06f) {
            cmd = "move stop";
            fault_hits += 1U;
        } else if (r < 0.12f) {
            cmd = "unknown speed token";
            fault_hits += 1U;
        } else if (r < 0.16f) {
            act[0] = frozen_act[0];
            act[1] = frozen_act[1];
            act[2] = frozen_act[2];
            act[3] = frozen_act[3];
            fault_hits += 1U;
            ms_apply_control_step(&x, &y, gx, gy, act);
            if (fabsf(gx - x) < 0.9f && fabsf(gy - y) < 0.9f) {
                recovered += 1U;
            }
            continue;
        }
        TFW_ASSERT_INT_EQ(0, ms_infer_action(weights, &tokenizer, cmd, state, act));
        frozen_act[0] = act[0];
        frozen_act[1] = act[1];
        frozen_act[2] = act[2];
        frozen_act[3] = act[3];
        ms_apply_control_step(&x, &y, gx, gy, act);
        if (fabsf(gx - x) < 0.9f && fabsf(gy - y) < 0.9f) {
            recovered += 1U;
        }
        TFW_ASSERT_TRUE(fabsf(x) < 50.0f);
        TFW_ASSERT_TRUE(fabsf(y) < 50.0f);
    }
    testfw_log_info("model/extreme/fault_recovery: faults=%zu recovered=%zu final=(%.3f,%.3f) remain=(%.3f,%.3f)",
                    fault_hits,
                    recovered,
                    (double)x,
                    (double)y,
                    (double)(gx - x),
                    (double)(gy - y));
    TFW_ASSERT_TRUE(fault_hits > 200U);
    TFW_ASSERT_TRUE(recovered > 500U);
    TFW_ASSERT_TRUE(fabsf(gx - x) < 1.4f);
    TFW_ASSERT_TRUE(fabsf(gy - y) < 1.4f);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 负向测试：空命令应触发模型推理输入错误。
 */
static int test_model_expected_error_empty_command(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    float weights[MS_TOTAL_WEIGHT_COUNT];
    float state[STATE_DIM] = {0};
    float act[OUTPUT_DIM] = {0};
    int rc = 0;
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    rc = ms_infer_action(weights, &tokenizer, "", state, act);
    testfw_log_info("model/negative/empty_command: rc=%d (expected=1)", rc);
    TFW_ASSERT_INT_EQ(1, rc);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 负向测试：超长命令应触发 token 缓冲区不足错误。
 */
static int test_model_expected_error_too_many_tokens(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    float weights[MS_TOTAL_WEIGHT_COUNT];
    float state[STATE_DIM] = {0};
    float act[OUTPUT_DIM] = {0};
    int rc = 0;
    const char* cmd = "move right fast move right fast move right fast";
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, ms_init_vocab_tokenizer(&vocab, &tokenizer));
    ms_init_weights(weights, MS_TOTAL_WEIGHT_COUNT);
    rc = ms_infer_action(weights, &tokenizer, cmd, state, act);
    testfw_log_info("model/negative/too_many_tokens: rc=%d (expected=1)", rc);
    TFW_ASSERT_INT_EQ(1, rc);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 负向测试：损坏协议包应在模型前置链路被拒绝。
 */
static int test_model_expected_error_corrupted_protocol_packet(void) {
    char packet[64];
    ProtocolFrame frame;
    char raw[32];
    int ids[8];
    memset(packet, 0, sizeof(packet));
    memset(raw, 0, sizeof(raw));
    memset(ids, 0, sizeof(ids));
    (void)snprintf(packet, sizeof(packet), "TOK|1,2,3");
    TFW_ASSERT_INT_EQ(PROTOCOL_STATUS_FORMAT_ERROR,
                      protocol_decode_packet(packet, &frame, raw, sizeof(raw), ids, 8U));
    testfw_log_info("model/negative/corrupted_protocol: decode rejected as expected");
    return 0;
}

/**
 * @brief 返回“模型专项”测试分组。
 *
 * 覆盖范围：
 * - 泛化：同分布插值、跨区间泛化
 * - OOD：远分布输入稳健性
 * - 对抗扰动：状态扰动、token 扰动
 * - 稳定性：重复推理稳定
 * - 一致性：token 顺序、协议链路、种子复现
 * - 极限：长时序闭环、多目标切换、延迟噪声联合、故障注入恢复
 */
TestCaseGroup testcases_get_model_special_group(void) {
    static const TestCase cases[] = {
        {"model_generalization_interpolation", "模型专项", "验证模型在同分布插值区间的泛化能力", "train=280 eval=220 range=10", "0(PASS)", test_model_generalization_interpolation},
        {"model_generalization_cross_range", "模型专项", "验证模型跨区间泛化能力", "train_range=8 eval_range=14", "0(PASS)", test_model_generalization_cross_range},
        {"model_ood_far_range_bounds", "模型专项", "验证远分布OOD输入下输出边界与误差阈值", "eval_range=20", "0(PASS)", test_model_ood_far_range_bounds},
        {"model_adversarial_state_perturbation", "模型专项", "验证状态对抗扰动下误差增量和翻转率", "eps=0.06 samples=180", "0(PASS)", test_model_adversarial_state_perturbation},
        {"model_adversarial_token_speed_flip", "模型专项", "验证速度token翻转下方向一致性", "fast<->slow samples=180", "0(PASS)", test_model_adversarial_token_speed_flip},
        {"model_stability_repeat_inference", "模型专项", "验证同输入重复推理的确定性", "repeat=1200", "0(PASS)", test_model_stability_repeat_inference},
        {"model_consistency_token_order_invariance", "模型专项", "验证token顺序变化下输出一致性", "reverse token order", "0(PASS)", test_model_consistency_token_order_invariance},
        {"model_consistency_protocol_roundtrip", "模型专项", "验证协议编解码前后模型输出一致", "token packet roundtrip", "0(PASS)", test_model_consistency_protocol_roundtrip},
        {"model_consistency_seed_reproducibility", "模型专项", "验证固定种子训练参数可复现", "same seed same trainset", "0(PASS)", test_model_consistency_seed_reproducibility},
        {"model_extreme_long_horizon_closed_loop", "模型专项", "验证1万帧闭环下长期漂移与收敛稳定", "frames=10000 multi-goal", "0(PASS)", test_model_extreme_long_horizon_closed_loop},
        {"model_extreme_multi_goal_switching", "模型专项", "验证高频目标切换下持续重规划能力", "frames=2400 switch/120", "0(PASS)", test_model_extreme_multi_goal_switching},
        {"model_extreme_delay_noise_joint", "模型专项", "验证观测延迟+噪声联合注入的鲁棒性", "delay=2 noise=0.025", "0(PASS)", test_model_extreme_delay_noise_joint},
        {"model_extreme_fault_injection_recovery", "模型专项", "验证掉帧与错误命令注入后的恢复能力", "fault_rate~16% frames=3200", "0(PASS)", test_model_extreme_fault_injection_recovery},
        {"model_expected_error_empty_command", "模型专项", "验证空命令在模型入口被拒绝", "command=''", "0(PASS)", test_model_expected_error_empty_command},
        {"model_expected_error_too_many_tokens", "模型专项", "验证超长命令触发token缓冲区错误", "9-token command", "0(PASS)", test_model_expected_error_too_many_tokens},
        {"model_expected_error_corrupted_protocol_packet", "模型专项", "验证损坏协议包在前置链路被拒绝", "packet without newline", "0(PASS)", test_model_expected_error_corrupted_protocol_packet}
    };
    TestCaseGroup group;
    group.cases = cases;
    group.count = sizeof(cases) / sizeof(cases[0]);
    return group;
}
