#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/config_user.h"
#include "../include/csv_loader.h"
#include "../include/ops.h"
#include "../include/platform_driver.h"
#include "../include/tensor.h"
#include "../include/tokenizer.h"
#include "../include/weights_io.h"

/**
 * @brief 完整 C99 Demo 的模型参数布局定义。
 *
 * 设计目的：
 * - 使用线性可解释结构完成“数据生成→训练→导出→加载→推理”闭环。
 * - 参数布局固定，方便二进制保存和回读校验。
 */
enum DemoModelLayout {
    DEMO_TOKEN_WEIGHT_COUNT = VOCAB_SIZE * OUTPUT_DIM,
    DEMO_STATE_WEIGHT_COUNT = STATE_DIM * OUTPUT_DIM,
    DEMO_BIAS_COUNT = OUTPUT_DIM,
    DEMO_TOTAL_WEIGHT_COUNT = DEMO_TOKEN_WEIGHT_COUNT + DEMO_STATE_WEIGHT_COUNT + DEMO_BIAS_COUNT
};

/**
 * @brief 训练上下文，统一保存运行时资源，避免资源分散导致释放遗漏。
 */
typedef struct DemoContext {
    Vocabulary vocab;
    Tokenizer tokenizer;
    CsvDataset dataset;
    DriverStub pc_driver;
} DemoContext;

typedef struct Pose2D {
    float x;
    float y;
} Pose2D;

/**
 * @brief 统一写出演示训练数据到 CSV 文件。
 *
 * 背景说明：
 * - 用户要求演示“生成训练数据”步骤，这里显式落盘 CSV。
 * - 每行字段：command + 8 维 state + 4 维 target。
 *
 * @param file_path CSV 输出路径
 * @return int 0 成功，非 0 失败
 */
static int write_demo_training_csv(const char* file_path) {
    static const char* rows[] = {
        "move left,0.8,0.1,0.2,0.0,0.0,0.2,0.0,0.0,1.0,0.0,-0.8,0.0",
        "move right,0.2,0.9,0.2,0.0,0.0,0.2,0.0,0.0,0.0,1.0,0.8,0.0",
        "move left fast,1.0,0.1,0.4,0.0,0.1,0.4,0.0,0.0,1.0,0.0,-1.0,0.5",
        "move right fast,0.1,1.0,0.4,0.0,0.1,0.4,0.0,0.0,0.0,1.0,1.0,0.5",
        "move stop,0.0,0.0,0.1,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0",
        "move left slow,0.7,0.1,0.1,0.0,0.0,0.1,0.0,0.0,1.0,0.0,-0.4,0.0",
        "move right slow,0.1,0.7,0.1,0.0,0.0,0.1,0.0,0.0,0.0,1.0,0.4,0.0",
        "stop,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.1"
    };
    FILE* fp = NULL;
    size_t i = 0U;
    if (file_path == NULL) {
        return -1;
    }
    fp = fopen(file_path, "w");
    if (fp == NULL) {
        return -2;
    }
    for (i = 0U; i < sizeof(rows) / sizeof(rows[0]); ++i) {
        if (fprintf(fp, "%s\n", rows[i]) < 0) {
            fclose(fp);
            return -3;
        }
    }
    fclose(fp);
    return 0;
}

/**
 * @brief 构建演示词表并初始化 Tokenizer。
 *
 * 关键保护点：
 * - 词表 token 与训练数据中的指令词一致，避免推理阶段全部命中 <unk>。
 *
 * @param ctx 训练上下文
 * @return int TokenizerStatus
 */
static int init_vocab_and_tokenizer(DemoContext* ctx) {
    static const char* tokens[] = {
        "<unk>", "move", "left", "right", "stop", "fast", "slow"
    };
    size_t i = 0U;
    int rc = TOKENIZER_STATUS_OK;
    if (ctx == NULL) {
        return TOKENIZER_STATUS_INVALID_ARGUMENT;
    }
    rc = vocab_init(&ctx->vocab, VOCAB_SIZE);
    if (rc != TOKENIZER_STATUS_OK) {
        return rc;
    }
    for (i = 0U; i < sizeof(tokens) / sizeof(tokens[0]); ++i) {
        rc = vocab_add_token(&ctx->vocab, tokens[i], NULL);
        if (rc != TOKENIZER_STATUS_OK) {
            vocab_free(&ctx->vocab);
            return rc;
        }
    }
    rc = tokenizer_init(&ctx->tokenizer, &ctx->vocab, 0);
    if (rc != TOKENIZER_STATUS_OK) {
        vocab_free(&ctx->vocab);
        return rc;
    }
    return TOKENIZER_STATUS_OK;
}

/**
 * @brief 初始化模型参数，提供稳定可复现的初始值。
 *
 * @param weights 模型参数数组
 * @param count   参数数量
 */
static void init_weights(float* weights, size_t count) {
    size_t i = 0U;
    if (weights == NULL) {
        return;
    }
    for (i = 0U; i < count; ++i) {
        /* 使用确定性小值初始化，避免随机数依赖，便于复现实验结果。 */
        weights[i] = ((float)((int)(i % 17U) - 8)) * 0.001f;
    }
}

/**
 * @brief 计算模型 logits（未激活输出）。
 *
 * 算法说明：
 * - 文本特征：token 行向量平均。
 * - 状态特征：state 与线性层相乘。
 * - 输出层：token 分支 + state 分支 + bias。
 *
 * @param weights      参数数组
 * @param token_ids    token id 数组
 * @param token_count  token 数量
 * @param state        8 维状态向量
 * @param out_logits   输出 logits
 */
static void predict_logits(const float* weights,
                           const int* token_ids,
                           size_t token_count,
                           const float* state,
                           float* out_logits) {
    const size_t token_base = 0U;
    const size_t state_base = DEMO_TOKEN_WEIGHT_COUNT;
    const size_t bias_base = DEMO_TOKEN_WEIGHT_COUNT + DEMO_STATE_WEIGHT_COUNT;
    size_t i = 0U;
    size_t j = 0U;
    if (weights == NULL || token_ids == NULL || state == NULL || out_logits == NULL || token_count == 0U) {
        return;
    }

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
 * @brief 将 logits 映射为执行器输出。
 *
 * @param logits 输入 logits
 * @param out_act 输出动作向量
 * @return int 0 成功，非 0 失败
 */
static int activate_output(const float* logits, float* out_act) {
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
 * @brief 执行单样本训练（前向 + MSE + 简化 SGD）。
 *
 * 关键约束：
 * - 为保持 Demo 简洁，梯度使用“输出误差近似”。
 * - 目标是演示训练闭环，不引入复杂自动求导框架。
 *
 * @param weights     参数数组
 * @param token_ids   token id
 * @param token_count token 数量
 * @param state       状态向量
 * @param target      监督目标
 * @param lr          学习率
 * @param out_loss    输出损失
 * @return int 0 成功，非 0 失败
 */
static int train_one_sample(float* weights,
                            const int* token_ids,
                            size_t token_count,
                            const float* state,
                            const float* target,
                            float lr,
                            float* out_loss) {
    const size_t token_base = 0U;
    const size_t state_base = DEMO_TOKEN_WEIGHT_COUNT;
    const size_t bias_base = DEMO_TOKEN_WEIGHT_COUNT + DEMO_STATE_WEIGHT_COUNT;
    float logits[OUTPUT_DIM];
    float pred[OUTPUT_DIM];
    float grad[OUTPUT_DIM];
    size_t i = 0U;
    size_t j = 0U;
    if (weights == NULL || token_ids == NULL || state == NULL || target == NULL || out_loss == NULL || token_count == 0U) {
        return -1;
    }

    predict_logits(weights, token_ids, token_count, state, logits);
    if (activate_output(logits, pred) != 0) {
        return -2;
    }

    *out_loss = 0.0f;
    for (j = 0U; j < OUTPUT_DIM; ++j) {
        float err = pred[j] - target[j];
        grad[j] = err;
        *out_loss += err * err;
    }
    *out_loss /= (float)OUTPUT_DIM;

    for (i = 0U; i < token_count; ++i) {
        size_t id = (token_ids[i] >= 0) ? (size_t)token_ids[i] : 0U;
        if (id >= VOCAB_SIZE) {
            id = 0U;
        }
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            weights[token_base + id * OUTPUT_DIM + j] -= (lr * grad[j]) / (float)token_count;
        }
    }

    for (i = 0U; i < STATE_DIM; ++i) {
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            weights[state_base + i * OUTPUT_DIM + j] -= lr * grad[j] * state[i];
        }
    }

    for (j = 0U; j < OUTPUT_DIM; ++j) {
        weights[bias_base + j] -= lr * grad[j];
    }
    return 0;
}

/**
 * @brief 使用 CSV 数据集执行多轮训练。
 *
 * @param ctx      训练上下文
 * @param weights  参数数组
 * @param epochs   训练轮次
 * @param lr       学习率
 * @return int 0 成功，非 0 失败
 */
static int train_dataset(DemoContext* ctx, float* weights, size_t epochs, float lr) {
    size_t epoch = 0U;
    if (ctx == NULL || weights == NULL || ctx->dataset.samples == NULL || ctx->dataset.count == 0U) {
        return -1;
    }
    for (epoch = 0U; epoch < epochs; ++epoch) {
        size_t i = 0U;
        float loss_sum = 0.0f;
        for (i = 0U; i < ctx->dataset.count; ++i) {
            int token_ids[MAX_SEQ_LEN];
            size_t token_count = 0U;
            float loss = 0.0f;
            int rc = TOKENIZER_STATUS_OK;
            memset(token_ids, 0, sizeof(token_ids));

            rc = tokenizer_encode(&ctx->tokenizer,
                                  ctx->dataset.samples[i].command,
                                  token_ids,
                                  MAX_SEQ_LEN,
                                  &token_count);
            if (rc != TOKENIZER_STATUS_OK || token_count == 0U) {
                return -2;
            }
            if (train_one_sample(weights,
                                 token_ids,
                                 token_count,
                                 ctx->dataset.samples[i].state,
                                 ctx->dataset.samples[i].target,
                                 lr,
                                 &loss) != 0) {
                return -3;
            }
            loss_sum += loss;
        }
        printf("epoch=%zu avg_loss=%.6f\n", epoch + 1U, (double)(loss_sum / (float)ctx->dataset.count));
    }
    return 0;
}

/**
 * @brief 基于已加载权重执行一次推理并发送到驱动桩。
 *
 * @param ctx             训练上下文
 * @param loaded_weights  已加载权重
 * @param command         推理指令文本
 * @param state           推理状态向量
 * @return int 0 成功，非 0 失败
 */
static int run_inference(DemoContext* ctx,
                         const float* loaded_weights,
                         const char* command,
                         const float* state) {
    int token_ids[MAX_SEQ_LEN];
    size_t token_count = 0U;
    float logits[OUTPUT_DIM];
    float act[OUTPUT_DIM];
    int rc = TOKENIZER_STATUS_OK;
    if (ctx == NULL || loaded_weights == NULL || command == NULL || state == NULL) {
        return -1;
    }
    memset(token_ids, 0, sizeof(token_ids));
    memset(logits, 0, sizeof(logits));
    memset(act, 0, sizeof(act));

    rc = tokenizer_encode(&ctx->tokenizer, command, token_ids, MAX_SEQ_LEN, &token_count);
    if (rc != TOKENIZER_STATUS_OK || token_count == 0U) {
        return -2;
    }
    predict_logits(loaded_weights, token_ids, token_count, state, logits);
    if (activate_output(logits, act) != 0) {
        return -3;
    }

    printf("inference command=\"%s\"\n", command);
    printf("actuator=[%.4f, %.4f, %.4f, %.4f]\n",
           (double)act[0], (double)act[1], (double)act[2], (double)act[3]);
    if (driver_stub_apply(&ctx->pc_driver, act, OUTPUT_DIM) != DRIVER_STATUS_OK) {
        return -4;
    }
    return 0;
}

static const char* select_frame_command(const Pose2D* cur, const Pose2D* goal) {
    const float dx = goal->x - cur->x;
    const float dy = goal->y - cur->y;
    if (fabsf(dx) < 0.30f && fabsf(dy) < 0.30f) {
        return "move stop";
    }
    if (dx >= 0.0f) {
        return (dy > 2.0f) ? "move right fast" : "move right slow";
    }
    return (dy > 2.0f) ? "move left fast" : "move left slow";
}

static float clamp_abs_step(float step, float remain) {
    if (fabsf(step) > fabsf(remain)) {
        return remain;
    }
    return step;
}

static int run_external_goal_loop(DemoContext* ctx,
                                  const float* loaded_weights,
                                  const Pose2D* goal,
                                  Pose2D* io_pose,
                                  size_t max_frames) {
    size_t frame = 0U;
    if (ctx == NULL || loaded_weights == NULL || goal == NULL || io_pose == NULL || max_frames == 0U) {
        return -1;
    }
    printf("external loop start: from(%.2f, %.2f) -> goal(%.2f, %.2f)\n",
           (double)io_pose->x,
           (double)io_pose->y,
           (double)goal->x,
           (double)goal->y);
    for (frame = 0U; frame < max_frames; ++frame) {
        const char* command = NULL;
        float state[STATE_DIM];
        int token_ids[MAX_SEQ_LEN];
        size_t token_count = 0U;
        float logits[OUTPUT_DIM];
        float act[OUTPUT_DIM];
        float remain_x = 0.0f;
        float remain_y = 0.0f;
        float step_x = 0.0f;
        float step_y = 0.0f;
        int rc = TOKENIZER_STATUS_OK;

        remain_x = goal->x - io_pose->x;
        remain_y = goal->y - io_pose->y;
        if (fabsf(remain_x) < 0.30f && fabsf(remain_y) < 0.30f) {
            printf("external loop done at frame=%zu pose=(%.3f, %.3f)\n",
                   frame,
                   (double)io_pose->x,
                   (double)io_pose->y);
            return 0;
        }

        command = select_frame_command(io_pose, goal);
        memset(state, 0, sizeof(state));
        memset(token_ids, 0, sizeof(token_ids));
        memset(logits, 0, sizeof(logits));
        memset(act, 0, sizeof(act));

        state[0] = io_pose->x / 15.0f;
        state[1] = io_pose->y / 15.0f;
        state[2] = remain_x / 15.0f;
        state[3] = remain_y / 15.0f;
        state[4] = (fabsf(remain_x) + fabsf(remain_y)) / 30.0f;

        rc = tokenizer_encode(&ctx->tokenizer, command, token_ids, MAX_SEQ_LEN, &token_count);
        if (rc != TOKENIZER_STATUS_OK || token_count == 0U) {
            return -2;
        }
        predict_logits(loaded_weights, token_ids, token_count, state, logits);
        if (activate_output(logits, act) != 0) {
            return -3;
        }

        step_x = act[2] * 0.55f;
        if (fabsf(step_x) < 0.08f && fabsf(remain_x) > 0.5f) {
            step_x = (remain_x > 0.0f) ? 0.10f : -0.10f;
        }
        step_x = clamp_abs_step(step_x, remain_x);

        step_y = 0.0f;
        if (remain_y > 0.0f) {
            step_y = ((act[3] + 1.0f) * 0.5f) * 0.40f;
            if (step_y < 0.05f) {
                step_y = 0.05f;
            }
            step_y = clamp_abs_step(step_y, remain_y);
        } else if (remain_y < 0.0f) {
            step_y = -0.05f;
            step_y = clamp_abs_step(step_y, remain_y);
        }

        io_pose->x += step_x;
        io_pose->y += step_y;

        printf("frame=%03zu cmd=\"%s\" remain=(%.3f, %.3f) act=(%.3f,%.3f,%.3f,%.3f) step=(%.3f, %.3f) pose=(%.3f, %.3f)\n",
               frame + 1U,
               command,
               (double)remain_x,
               (double)remain_y,
               (double)act[0],
               (double)act[1],
               (double)act[2],
               (double)act[3],
               (double)step_x,
               (double)step_y,
               (double)io_pose->x,
               (double)io_pose->y);
    }
    printf("external loop reached max_frames=%zu, final pose=(%.3f, %.3f), remain=(%.3f, %.3f)\n",
           max_frames,
           (double)io_pose->x,
           (double)io_pose->y,
           (double)(goal->x - io_pose->x),
           (double)(goal->y - io_pose->y));
    return 1;
}

/**
 * @brief 释放 Demo 上下文占用的所有资源。
 *
 * @param ctx 训练上下文
 */
static void cleanup_context(DemoContext* ctx) {
    if (ctx == NULL) {
        return;
    }
    csv_free_dataset(&ctx->dataset);
    driver_stub_shutdown(&ctx->pc_driver);
    vocab_free(&ctx->vocab);
}

/**
 * @brief 完整 C99 演示入口。
 *
 * 流程覆盖：
 * 1) 生成训练数据 CSV
 * 2) 加载数据并训练
 * 3) 导出权重（二进制 + C 源码）
 * 4) 加载权重
 * 5) 执行推理
 *
 * @param argc 参数数量
 * @param argv 参数数组
 * @return int 0 成功，非 0 失败
 */
int main(int argc, char** argv) {
    const char* csv_path = "demo_train_data.csv";
    const char* weight_bin_path = "demo_weights.bin";
    const char* weight_c_path = "demo_weights_export.c";
    float weights[DEMO_TOTAL_WEIGHT_COUNT];
    float* loaded_weights = NULL;
    size_t loaded_count = 0U;
    const float infer_state[STATE_DIM] = { 0.95f, 0.12f, 0.35f, 0.0f, 0.05f, 0.20f, 0.0f, 0.0f };
    Pose2D start_pose = { 0.0f, 0.0f };
    Pose2D target_pose = { 15.0f, 15.0f };
    DemoContext ctx;
    int rc = 0;
    (void)argc;
    (void)argv;

    memset(&ctx, 0, sizeof(ctx));
    memset(weights, 0, sizeof(weights));

    rc = write_demo_training_csv(csv_path);
    if (rc != 0) {
        fprintf(stderr, "write_demo_training_csv failed: %d\n", rc);
        return 1;
    }
    printf("generated training data: %s\n", csv_path);

    rc = init_vocab_and_tokenizer(&ctx);
    if (rc != TOKENIZER_STATUS_OK) {
        fprintf(stderr, "init_vocab_and_tokenizer failed: %d\n", rc);
        cleanup_context(&ctx);
        return 2;
    }

    if (driver_stub_init(&ctx.pc_driver, DRIVER_TYPE_PC) != DRIVER_STATUS_OK) {
        fprintf(stderr, "driver_stub_init failed\n");
        cleanup_context(&ctx);
        return 3;
    }

    rc = csv_load_dataset(csv_path, &ctx.dataset);
    if (rc != 0 || ctx.dataset.samples == NULL || ctx.dataset.count == 0U) {
        fprintf(stderr, "csv_load_dataset failed: %d\n", rc);
        cleanup_context(&ctx);
        return 4;
    }
    printf("loaded dataset count=%zu\n", ctx.dataset.count);

    init_weights(weights, DEMO_TOTAL_WEIGHT_COUNT);
    rc = train_dataset(&ctx, weights, 20U, 0.08f);
    if (rc != 0) {
        fprintf(stderr, "train_dataset failed: %d\n", rc);
        cleanup_context(&ctx);
        return 5;
    }

    rc = weights_save_binary(weight_bin_path, weights, DEMO_TOTAL_WEIGHT_COUNT);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        fprintf(stderr, "weights_save_binary failed: %d\n", rc);
        cleanup_context(&ctx);
        return 6;
    }
    rc = weights_export_c_source(weight_c_path, "g_demo_weights", weights, DEMO_TOTAL_WEIGHT_COUNT);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        fprintf(stderr, "weights_export_c_source failed: %d\n", rc);
        cleanup_context(&ctx);
        return 7;
    }
    printf("exported weights: %s , %s\n", weight_bin_path, weight_c_path);

    rc = weights_load_binary(weight_bin_path, &loaded_weights, &loaded_count);
    if (rc != WEIGHTS_IO_STATUS_OK || loaded_weights == NULL || loaded_count != DEMO_TOTAL_WEIGHT_COUNT) {
        fprintf(stderr, "weights_load_binary failed: rc=%d count=%zu\n", rc, loaded_count);
        free(loaded_weights);
        cleanup_context(&ctx);
        return 8;
    }
    printf("reloaded weight count=%zu\n", loaded_count);

    rc = run_inference(&ctx, loaded_weights, "move left fast", infer_state);
    if (rc != 0) {
        fprintf(stderr, "run_inference failed: %d\n", rc);
        free(loaded_weights);
        loaded_weights = NULL;
        cleanup_context(&ctx);
        return 9;
    }

    rc = run_external_goal_loop(&ctx, loaded_weights, &target_pose, &start_pose, 300U);
    if (rc != 0) {
        fprintf(stderr, "run_external_goal_loop failed: %d\n", rc);
        free(loaded_weights);
        loaded_weights = NULL;
        cleanup_context(&ctx);
        return 10;
    }
    free(loaded_weights);
    loaded_weights = NULL;

    cleanup_context(&ctx);
    printf("full c99 demo completed\n");
    return 0;
}
