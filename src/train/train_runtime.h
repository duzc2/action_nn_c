#ifndef TRAIN_RUNTIME_H
#define TRAIN_RUNTIME_H

typedef struct {
    const char* network_type;
    void* context;
} NNTrainRequest;

int nn_train_runtime_step(const NNTrainRequest* request);

#endif
