#ifndef CSV_LOADER_H
#define CSV_LOADER_H

#include <stddef.h>
#include "config_user.h"

/**
 * @brief 单条 CSV 样本结构。
 *
 * 设计目的：
 * - 统一训练样本在内存中的表达，避免训练流程直接依赖 CSV 行文本。
 * 关键约束：
 * - command 长度固定为 128，防止运行时动态分配导致额外碎片与复杂度。
 * - state 与 target 维度与当前任务约定一致，保持与训练主流程兼容。
 */
typedef struct CsvSample {
    char command[128];
    float state[STATE_DIM];
    float target[OUTPUT_DIM];
} CsvSample;

/**
 * @brief CSV 数据集容器。
 *
 * 设计目的：
 * - 以连续内存承载样本，便于顺序遍历与释放。
 * 关键保护点：
 * - 由 csv_free_dataset 负责释放 samples，调用方无需逐条释放。
 */
typedef struct CsvDataset {
    CsvSample* samples;
    size_t count;
} CsvDataset;

/**
 * @brief 从 CSV 文件加载训练样本。
 * @param file_path CSV 文件路径。
 * @param out_dataset 输出数据集对象。
 * @return 0 表示成功，负数表示失败。
 */
int csv_load_dataset(const char* file_path, CsvDataset* out_dataset);

/**
 * @brief 释放 csv_load_dataset 分配的数据集内存。
 * @param dataset 待释放的数据集对象，可为 NULL。
 */
void csv_free_dataset(CsvDataset* dataset);

#endif
