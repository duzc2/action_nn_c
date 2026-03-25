/**
 * @file mlp_train_ops.h
 * @brief MLP training operations interface
 *
 * Implements:
 * - Gradient descent optimization (SGD)
 * - Backpropagation algorithm
 * - Loss computation (MSE)
 * - Checkpoint save/load with validation
 * - Dual-mode execution (auto loop + step-by-step)
 */

#ifndef MLP_TRAIN_OPS_H
#define MLP_TRAIN_OPS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @brief Optimizer type
 */
typedef enum {
    MLP_OPT_SGD = 0,
    MLP_OPT_ADAM = 1
} MlpOptimizerType;

/**
 * @brief Loss function type
 */
typedef enum {
    MLP_LOSS_MSE = 0,
    MLP_LOSS_CROSS_ENTROPY = 1
} MlpLossType;

/**
 * @brief Training configuration
 */
typedef struct {
    float learning_rate;
    float momentum;
    float weight_decay;
    MlpOptimizerType optimizer;
    MlpLossType loss_func;
    size_t batch_size;
    uint32_t seed;
} MlpTrainConfig;

/**
 * @brief Gradient buffer for a single layer
 *
 * Stores gradients for weights and biases during backpropagation.
 */
typedef struct {
    float* weight_grad;
    float* bias_grad;
    float* velocity_w;
    float* velocity_b;
    float beta1;
    float beta2;
    float* m_w;
    float* m_b;
    float* v_w;
    float* v_b;
} MlpLayerGrad;

/**
 * @brief MLP training context
 *
 * Contains inference context, gradient buffers, optimizer state,
 * and training statistics.
 */
typedef struct {
    void* infer_ctx;
    MlpTrainConfig config;
    MlpLayerGrad* grads;
    size_t layer_count;
    float** activations;
    float* input_buffer;
    float* target_buffer;
    float* loss_buffer;
    float loss_history[100];
    size_t loss_history_count;
    size_t total_epochs;
    size_t total_steps;
    uint64_t checkpoint_network_hash;
    uint64_t checkpoint_layout_hash;
} MlpTrainContext;

/**
 * @brief Training step input
 *
 * Input data for single training step.
 */
typedef struct {
    const float* input;
    const float* target;
    size_t batch_size;
} MlpTrainStepInput;

/**
 * @brief Training step output
 *
 * Output data after single training step.
 */
typedef struct {
    float loss;
    float* predictions;
} MlpTrainStepOutput;

/**
 * @brief Checkpoint info header
 *
 * Stored at beginning of checkpoint file.
 */
typedef struct {
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t abi_version;
    uint32_t epoch;
    uint32_t step;
    float best_loss;
} MlpCheckpointHeader;

/**
 * @brief Create MLP training context
 *
 * @param infer_ctx Inference context (borrowed reference)
 * @param config Training configuration. Must not be NULL.
 * @return New training context, or NULL on failure
 */
MlpTrainContext* nn_mlp_train_create(void* infer_ctx, const MlpTrainConfig* config);

/**
 * @brief Free MLP training context
 *
 * @param ctx Training context to free
 */
void nn_mlp_train_destroy(MlpTrainContext* ctx);

/**
 * @brief Run single training step with forward and backward pass
 *
 * Implements:
 * 1. Forward propagation
 * 2. Loss computation
 * 3. Backpropagation
 * 4. Parameter update
 *
 * @param ctx Training context
 * @param input Input batch
 * @param target Target values
 * @return 0 on success, negative on failure
 */
int nn_mlp_train_step_with_data(MlpTrainContext* ctx, const float* input, const float* target);

/**
 * @brief Run training step with structured input/output
 *
 * @param ctx Training context
 * @param step_in Training step input
 * @param step_out Training step output (loss, predictions)
 * @return 0 on success, negative on failure
 */
int nn_mlp_train_step_ex(
    MlpTrainContext* ctx,
    const MlpTrainStepInput* step_in,
    MlpTrainStepOutput* step_out
);

/**
 * @brief Run automatic training loop
 *
 * @param ctx Training context
 * @param epochs Number of epochs to train
 * @param input Training inputs
 * @param target Training targets
 * @param sample_count Number of samples
 * @return 0 on success, negative on failure
 */
int nn_mlp_train_run_auto(
    MlpTrainContext* ctx,
    size_t epochs,
    const float* input,
    const float* target,
    size_t sample_count
);

/**
 * @brief Compute loss for given predictions and targets
 *
 * @param ctx Training context
 * @param predictions Predicted values
 * @param targets Target values
 * @param count Number of samples
 * @return Loss value
 */
float nn_mlp_train_compute_loss(
    MlpTrainContext* ctx,
    const float* predictions,
    const float* targets,
    size_t count
);

/**
 * @brief Save training checkpoint
 *
 * Stores weights, optimizer state, and training progress.
 *
 * @param ctx Training context
 * @param fp File pointer (opened by caller)
 * @param best_loss Best loss achieved so far
 * @return 1 on success, 0 on failure
 */
int nn_mlp_train_save_checkpoint(
    MlpTrainContext* ctx,
    FILE* fp,
    float best_loss
);

/**
 * @brief Load training checkpoint
 *
 * Validates network hash before loading.
 *
 * @param ctx Training context
 * @param fp File pointer (opened by caller)
 * @param current_hash Current network hash
 * @param current_layout_hash Current layout hash
 * @return 1 on success, 0 on failure
 */
int nn_mlp_train_load_checkpoint(
    MlpTrainContext* ctx,
    FILE* fp,
    uint64_t current_hash,
    uint64_t current_layout_hash
);

/**
 * @brief Validate checkpoint compatibility
 *
 * Checks if checkpoint network/layout hash matches current network.
 *
 * @param fp File pointer (opened by caller)
 * @param current_hash Current network hash
 * @param current_layout_hash Current layout hash
 * @return 1 if compatible, 0 if incompatible
 */
int nn_mlp_train_validate_checkpoint(
    FILE* fp,
    uint64_t current_hash,
    uint64_t current_layout_hash
);

/**
 * @brief Get training statistics
 *
 * @param ctx Training context
 * @param out_epochs Output total epochs
 * @param out_steps Output total steps
 * @param out_avg_loss Output average loss
 */
void nn_mlp_train_get_stats(
    MlpTrainContext* ctx,
    size_t* out_epochs,
    size_t* out_steps,
    float* out_avg_loss
);

/**
 * @brief Registry-compatible train step wrapper
 *
 * Adapter for train registry with void* context parameter.
 * Used by NNTrainRegistry to dispatch training calls.
 *
 * @param context Void pointer to MlpTrainContext
 * @return 0 on success, -1 on failure
 */
int nn_mlp_train_step(void* context);

#endif
