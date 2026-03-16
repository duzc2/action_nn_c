#ifndef WORKFLOW_INFER_H
#define WORKFLOW_INFER_H

#include <stddef.h>

#include "../../include/tokenizer.h"
#include "../../include/workflow_status.h"

/**
 * @brief 推理运行时上下文。
 *
 * 关键约束：
 * - ready==1 才允许执行 workflow_run_step，确保词表和权重都已完成初始化。
 */
typedef struct WorkflowRuntime {
    Vocabulary vocab;
    Tokenizer tokenizer;
    float* weights;
    size_t weight_count;
    int ready;
} WorkflowRuntime;

/**
 * @brief 加载词表并初始化分词器。
 */
int workflow_prepare_tokenizer(const char* vocab_path, Vocabulary* vocab, Tokenizer* tokenizer);

/**
 * @brief 返回当前推理模型期望的总权重数量。
 */
size_t workflow_weights_count(void);

/**
 * @brief 初始化推理运行时（词表 + 二进制权重）。
 */
int workflow_runtime_init(WorkflowRuntime* runtime, const char* vocab_path, const char* weights_bin_path);

/**
 * @brief 释放推理运行时资源。
 */
void workflow_runtime_shutdown(WorkflowRuntime* runtime);

/**
 * @brief 执行一条命令驱动的推理步骤。
 */
int workflow_run_step(WorkflowRuntime* runtime, const char* command, const float* state, float* out_action);

/**
 * @brief 执行无命令（仅状态驱动）的推理步骤。
 */
int workflow_run_step_goal(WorkflowRuntime* runtime, const float* state, float* out_action);

#endif
