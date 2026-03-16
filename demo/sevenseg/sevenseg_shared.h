#ifndef SEVENSEG_SHARED_H
#define SEVENSEG_SHARED_H

#include "../../src/include/config_user.h"
#include "../../src/train/include/workflow_train.h"

enum {
    MAX_SEVENSEG_SAMPLES = 512
};

extern const int g_sevenseg_truth[10][7];

int sevenseg_ensure_dir(const char* path);
int sevenseg_write_vocab(const char* file_path);
int sevenseg_build_samples(WorkflowTrainSample* out_samples,
                           char commands[][32],
                           float states[][STATE_DIM],
                           float targets[][OUTPUT_DIM],
                           size_t capacity,
                           size_t* out_count);
int sevenseg_verify_samples(const WorkflowTrainSample* samples, size_t sample_count);
void sevenseg_render_cli(int digit, const int seg[7]);

#endif
