#ifndef WORKFLOW_TRAIN_H
#define WORKFLOW_TRAIN_H

#include <stddef.h>

#include "../../include/network_spec.h"
#include "../../include/workflow_status.h"

/**
 * @brief 基于 CSV 数据训练的配置。
 */
typedef struct WorkflowTrainOptions {
    const char* csv_path;
    const char* vocab_path;
    const char* out_weights_bin;
    const char* out_weights_c;
    const char* out_symbol;
    const NetworkSpec* network_spec;
    size_t epochs;
    float learning_rate;
} WorkflowTrainOptions;

/**
 * @brief 从 CSV 数据集训练并导出权重。
 */
int workflow_train_from_csv(const WorkflowTrainOptions* options);

/**
 * @brief 内存样本单元。
 */
typedef struct WorkflowTrainSample {
    const char* command;
    const float* state;
    const float* target;
} WorkflowTrainSample;

/**
 * @brief 基于内存样本训练的配置。
 */
typedef struct WorkflowTrainMemoryOptions {
    const WorkflowTrainSample* samples;
    size_t sample_count;
    const char* vocab_path;
    const char* out_weights_bin;
    const char* out_weights_c;
    const char* out_symbol;
    const NetworkSpec* network_spec;
    size_t epochs;
    float learning_rate;
} WorkflowTrainMemoryOptions;

/**
 * @brief 从内存样本训练并导出权重。
 */
int workflow_train_from_memory(const WorkflowTrainMemoryOptions* options);

#endif
