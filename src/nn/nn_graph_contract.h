#ifndef NN_GRAPH_CONTRACT_H
#define NN_GRAPH_CONTRACT_H

#include "nn_codegen_hooks.h"

typedef struct {
    const char* type_name;
    NNInferCreateFn create;
    NNInferDestroyFn destroy;
    NNInferAutoRunFn auto_run;
    NNInferGraphRunFn graph_run;
    NNInferLoadWeightsFn load_weights;
    NNInferSaveWeightsFn save_weights;
    int supports_graph_mode;
} NNGraphInferContract;

typedef struct {
    const char* type_name;
    NNTrainCreateFn create;
    NNTrainDestroyFn destroy;
    NNTrainStepWithDataFn step_with_data;
    NNTrainStepWithOutputGradientFn step_with_output_gradient;
    NNTrainGetStatsFn get_stats;
    int supports_graph_backprop;
} NNGraphTrainContract;

const NNGraphInferContract* nn_graph_infer_contract_find(const char* type_name);
int nn_graph_infer_contract_supports_graph_mode(const char* type_name);

const NNGraphTrainContract* nn_graph_train_contract_find(const char* type_name);
int nn_graph_train_contract_supports_backprop(const char* type_name);

#endif
