#ifndef SEVENSEG_SHARED_H
#define SEVENSEG_SHARED_H

#include "../../src/include/config_user.h"
#include "../../src/train/include/workflow_train.h"

enum {
    /* SevenSeg 演示默认样本容量上限。 */
    MAX_SEVENSEG_SAMPLES = 512
};

/**
 * @brief 数字到七段码标准真值表。
 */
extern const int g_sevenseg_truth[10][7];

/**
 * @brief 确保目录存在。
 */
int sevenseg_ensure_dir(const char* path);

/**
 * @brief 写出 SevenSeg 词表文本文件。
 */
int sevenseg_write_vocab(const char* file_path);

/**
 * @brief 构建 SevenSeg 训练样本集。
 */
int sevenseg_build_samples(WorkflowTrainSample* out_samples,
                           char commands[][32],
                           float states[][STATE_DIM],
                           float targets[][OUTPUT_DIM],
                           size_t capacity,
                           size_t* out_count);

/**
 * @brief 校验样本内容与真值表一致。
 */
int sevenseg_verify_samples(const WorkflowTrainSample* samples, size_t sample_count);

/**
 * @brief 在终端渲染七段数码管图形。
 */
void sevenseg_render_cli(int digit, const int seg[7]);

/**
 * @brief 解析可用的 sevenseg 数据目录。
 */
int sevenseg_resolve_data_dir(const char* preferred_dir, char* out_dir, size_t out_cap);

#endif
