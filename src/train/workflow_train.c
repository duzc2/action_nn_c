#include <stdio.h>
#include <string.h>

#include "../include/config_user.h"
#include "../include/csv_loader.h"
#include "../include/ops.h"
#include "../include/tensor.h"
#include "../include/tokenizer.h"
#include "../include/weights_io.h"
#include "../infer/include/workflow_infer.h"
#include "./include/workflow_train.h"

enum {
    TOKEN_WEIGHT_COUNT = VOCAB_SIZE * OUTPUT_DIM,
    STATE_WEIGHT_COUNT = STATE_DIM * OUTPUT_DIM,
    BIAS_COUNT = OUTPUT_DIM,
    TOTAL_WEIGHT_COUNT = TOKEN_WEIGHT_COUNT + STATE_WEIGHT_COUNT + BIAS_COUNT
};

/**
 * @brief 初始化权重为小幅随机样式常量序列。
 *
 * 背景：
 * - 该初始化足够打破全零对称性，同时保持数值稳定，便于示例训练快速收敛。
 */
static void init_weights(float* w, size_t n) {
    size_t i = 0U;
    for (i = 0U; i < n; ++i) {
        w[i] = ((float)((int)(i % 11U) - 5)) * 0.001f;
    }
}

/**
 * @brief 应用输出激活函数。
 */
static int activate_output(const float* logits, float* out_act) {
    const int activations[OUTPUT_DIM] = IO_MAPPING_ACTIVATIONS;
    Tensor in_t;
    Tensor out_t;
    size_t shape[1] = { OUTPUT_DIM };
    int rc = tensor_init_view(&in_t, (float*)logits, 1U, shape);
    if (rc != TENSOR_STATUS_OK) {
        return WORKFLOW_STATUS_INTERNAL_ERROR;
    }
    rc = tensor_init_view(&out_t, out_act, 1U, shape);
    if (rc != TENSOR_STATUS_OK) {
        return WORKFLOW_STATUS_INTERNAL_ERROR;
    }
    rc = op_actuator(&in_t, activations, &out_t);
    return (rc == TENSOR_STATUS_OK) ? WORKFLOW_STATUS_OK : WORKFLOW_STATUS_INTERNAL_ERROR;
}

/**
 * @brief 按统一线性模型计算 logits。
 */
static void predict_logits(const float* w,
                           const int* ids,
                           size_t token_count,
                           const float* state,
                           float* logits) {
    size_t i = 0U;
    size_t j = 0U;
    size_t token_base = 0U;
    size_t state_base = TOKEN_WEIGHT_COUNT;
    size_t bias_base = TOKEN_WEIGHT_COUNT + STATE_WEIGHT_COUNT;
    for (j = 0U; j < OUTPUT_DIM; ++j) {
        logits[j] = w[bias_base + j];
    }
    /* token 分量按 token 数做均值，保持不同文本长度下梯度尺度可控。 */
    for (i = 0U; i < token_count; ++i) {
        size_t id = (ids[i] >= 0) ? (size_t)ids[i] : 0U;
        if (id >= VOCAB_SIZE) {
            id = 0U;
        }
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            logits[j] += w[token_base + id * OUTPUT_DIM + j] / (float)token_count;
        }
    }
    for (i = 0U; i < STATE_DIM; ++i) {
        for (j = 0U; j < OUTPUT_DIM; ++j) {
            logits[j] += state[i] * w[state_base + i * OUTPUT_DIM + j];
        }
    }
}

/**
 * @brief 基于内存样本执行训练循环。
 *
 * 关键约束：
 * - 训练误差采用逐维平方误差。
 * - 参数更新顺序固定为 token 权重、state 权重、偏置，保持可复现行为。
 */
static int train_with_samples(const Tokenizer* tokenizer,
                              const WorkflowTrainSample* samples,
                              size_t sample_count,
                              size_t epochs,
                              float learning_rate,
                              float* weights) {
    size_t epoch = 0U;
    int rc = 0;
    for (epoch = 0U; epoch < epochs; ++epoch) {
        size_t i = 0U;
        float loss_sum = 0.0f;
        for (i = 0U; i < sample_count; ++i) {
            const WorkflowTrainSample* s = &samples[i];
            int ids[MAX_SEQ_LEN];
            size_t count = 0U;
            float logits[OUTPUT_DIM];
            float pred[OUTPUT_DIM];
            float grad[OUTPUT_DIM];
            size_t t = 0U;
            size_t j = 0U;
            if (s->state == NULL || s->target == NULL) {
                return WORKFLOW_STATUS_INVALID_ARGUMENT;
            }
            memset(ids, 0, sizeof(ids));
            memset(logits, 0, sizeof(logits));
            memset(pred, 0, sizeof(pred));
            memset(grad, 0, sizeof(grad));
            if (s->command != NULL && s->command[0] != '\0') {
                rc = tokenizer_encode(tokenizer, s->command, ids, MAX_SEQ_LEN, &count);
                if (rc != TOKENIZER_STATUS_OK || count == 0U) {
                    return WORKFLOW_STATUS_DATA_ERROR;
                }
            }
            predict_logits(weights, ids, count, s->state, logits);
            if (activate_output(logits, pred) != WORKFLOW_STATUS_OK) {
                return WORKFLOW_STATUS_INTERNAL_ERROR;
            }
            for (j = 0U; j < OUTPUT_DIM; ++j) {
                float err = pred[j] - s->target[j];
                grad[j] = err;
                loss_sum += err * err;
            }
            /* 关键算法：文本权重梯度按 token 数均分，避免长命令放大更新步长。 */
            for (t = 0U; t < count; ++t) {
                size_t id = (ids[t] >= 0) ? (size_t)ids[t] : 0U;
                if (id >= VOCAB_SIZE) {
                    id = 0U;
                }
                for (j = 0U; j < OUTPUT_DIM; ++j) {
                    weights[id * OUTPUT_DIM + j] -= (learning_rate * grad[j]) / (float)count;
                }
            }
            for (t = 0U; t < STATE_DIM; ++t) {
                size_t base = TOKEN_WEIGHT_COUNT + t * OUTPUT_DIM;
                for (j = 0U; j < OUTPUT_DIM; ++j) {
                    weights[base + j] -= learning_rate * grad[j] * s->state[t];
                }
            }
            for (j = 0U; j < OUTPUT_DIM; ++j) {
                weights[TOKEN_WEIGHT_COUNT + STATE_WEIGHT_COUNT + j] -= learning_rate * grad[j];
            }
        }
        printf("epoch=%zu avg_loss=%.6f\n",
               epoch + 1U,
               (double)(loss_sum / (float)(sample_count * OUTPUT_DIM)));
    }
    return WORKFLOW_STATUS_OK;
}

/**
 * @brief 从 CSV 数据训练并导出二进制/C 源权重。
 */
int workflow_train_from_csv(const WorkflowTrainOptions* options) {
    CsvDataset ds;
    Vocabulary vocab;
    Tokenizer tokenizer;
    WorkflowTrainSample samples[512];
    float weights[TOTAL_WEIGHT_COUNT];
    size_t i = 0U;
    int rc = 0;
    if (options == NULL || options->csv_path == NULL || options->vocab_path == NULL ||
        options->out_weights_bin == NULL || options->out_weights_c == NULL || options->out_symbol == NULL) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    if (options->epochs == 0U || options->learning_rate <= 0.0f) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    memset(&ds, 0, sizeof(ds));
    memset(&vocab, 0, sizeof(vocab));
    memset(&tokenizer, 0, sizeof(tokenizer));
    memset(weights, 0, sizeof(weights));
    rc = csv_load_dataset(options->csv_path, &ds);
    if (rc != 0 || ds.count == 0U) {
        return WORKFLOW_STATUS_IO_ERROR;
    }
    if (ds.count > sizeof(samples) / sizeof(samples[0])) {
        csv_free_dataset(&ds);
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    rc = workflow_prepare_tokenizer(options->vocab_path, &vocab, &tokenizer);
    if (rc != WORKFLOW_STATUS_OK) {
        csv_free_dataset(&ds);
        return rc;
    }
    init_weights(weights, TOTAL_WEIGHT_COUNT);
    for (i = 0U; i < ds.count; ++i) {
        samples[i].command = ds.samples[i].command;
        samples[i].state = ds.samples[i].state;
        samples[i].target = ds.samples[i].target;
    }
    rc = train_with_samples(&tokenizer, samples, ds.count, options->epochs, options->learning_rate, weights);
    if (rc != WORKFLOW_STATUS_OK) {
        vocab_free(&vocab);
        csv_free_dataset(&ds);
        return rc;
    }
    rc = weights_save_binary(options->out_weights_bin, weights, TOTAL_WEIGHT_COUNT);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        vocab_free(&vocab);
        csv_free_dataset(&ds);
        return WORKFLOW_STATUS_IO_ERROR;
    }
    rc = weights_export_c_source(options->out_weights_c, options->out_symbol, weights, TOTAL_WEIGHT_COUNT);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        vocab_free(&vocab);
        csv_free_dataset(&ds);
        return WORKFLOW_STATUS_IO_ERROR;
    }
    vocab_free(&vocab);
    csv_free_dataset(&ds);
    return WORKFLOW_STATUS_OK;
}

/**
 * @brief 从内存样本训练并导出二进制/C 源权重。
 */
int workflow_train_from_memory(const WorkflowTrainMemoryOptions* options) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    float weights[TOTAL_WEIGHT_COUNT];
    int rc = 0;
    if (options == NULL || options->samples == NULL || options->sample_count == 0U || options->vocab_path == NULL ||
        options->out_weights_bin == NULL || options->out_weights_c == NULL || options->out_symbol == NULL) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    if (options->epochs == 0U || options->learning_rate <= 0.0f) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    memset(&vocab, 0, sizeof(vocab));
    memset(&tokenizer, 0, sizeof(tokenizer));
    memset(weights, 0, sizeof(weights));
    rc = workflow_prepare_tokenizer(options->vocab_path, &vocab, &tokenizer);
    if (rc != WORKFLOW_STATUS_OK) {
        return rc;
    }
    init_weights(weights, TOTAL_WEIGHT_COUNT);
    rc = train_with_samples(&tokenizer,
                            options->samples,
                            options->sample_count,
                            options->epochs,
                            options->learning_rate,
                            weights);
    if (rc != WORKFLOW_STATUS_OK) {
        vocab_free(&vocab);
        return rc;
    }
    rc = weights_save_binary(options->out_weights_bin, weights, TOTAL_WEIGHT_COUNT);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        vocab_free(&vocab);
        return WORKFLOW_STATUS_IO_ERROR;
    }
    rc = weights_export_c_source(options->out_weights_c, options->out_symbol, weights, TOTAL_WEIGHT_COUNT);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        vocab_free(&vocab);
        return WORKFLOW_STATUS_IO_ERROR;
    }
    vocab_free(&vocab);
    return WORKFLOW_STATUS_OK;
}
