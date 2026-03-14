#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/config_user.h"
#include "../include/csv_loader.h"
#include "../include/ops.h"
#include "../include/tensor.h"
#include "../include/tokenizer.h"
#include "../include/weights_io.h"
#include "../include/workflow.h"

enum {
    TOKEN_WEIGHT_COUNT = VOCAB_SIZE * OUTPUT_DIM,
    STATE_WEIGHT_COUNT = STATE_DIM * OUTPUT_DIM,
    BIAS_COUNT = OUTPUT_DIM,
    TOTAL_WEIGHT_COUNT = TOKEN_WEIGHT_COUNT + STATE_WEIGHT_COUNT + BIAS_COUNT
};

static void init_weights(float* w, size_t n) {
    size_t i = 0U;
    for (i = 0U; i < n; ++i) {
        w[i] = ((float)((int)(i % 11U) - 5)) * 0.001f;
    }
}

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

static const char* choose_cmd(float dx) {
    if (fabsf(dx) < 0.3f) {
        return "move stop";
    }
    if (dx > 0.0f) {
        return (fabsf(dx) > 2.0f) ? "move right fast" : "move right";
    }
    return (fabsf(dx) > 2.0f) ? "move left fast" : "move left";
}

int workflow_prepare_tokenizer(const char* vocab_path, Vocabulary* vocab, Tokenizer* tokenizer) {
    int rc = 0;
    if (vocab_path == NULL || vocab == NULL || tokenizer == NULL) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    memset(vocab, 0, sizeof(*vocab));
    memset(tokenizer, 0, sizeof(*tokenizer));
    rc = vocab_load_text(vocab_path, vocab);
    if (rc != TOKENIZER_STATUS_OK) {
        return WORKFLOW_STATUS_IO_ERROR;
    }
    rc = tokenizer_init(tokenizer, vocab, 0);
    if (rc != TOKENIZER_STATUS_OK) {
        vocab_free(vocab);
        return WORKFLOW_STATUS_DATA_ERROR;
    }
    return WORKFLOW_STATUS_OK;
}

size_t workflow_weights_count(void) {
    return TOTAL_WEIGHT_COUNT;
}

int workflow_train_from_csv(const WorkflowTrainOptions* options) {
    CsvDataset ds;
    Vocabulary vocab;
    Tokenizer tokenizer;
    float weights[TOTAL_WEIGHT_COUNT];
    size_t epoch = 0U;
    int rc = 0;
    if (options == NULL || options->csv_path == NULL || options->vocab_path == NULL ||
        options->out_weights_bin == NULL || options->out_weights_c == NULL || options->out_symbol == NULL) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    if (options->epochs == 0U || options->learning_rate <= 0.0f) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    if (STATE_DIM != 8 || OUTPUT_DIM != 4) {
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
    rc = workflow_prepare_tokenizer(options->vocab_path, &vocab, &tokenizer);
    if (rc != WORKFLOW_STATUS_OK) {
        csv_free_dataset(&ds);
        return rc;
    }
    init_weights(weights, TOTAL_WEIGHT_COUNT);
    for (epoch = 0U; epoch < options->epochs; ++epoch) {
        size_t i = 0U;
        float loss_sum = 0.0f;
        for (i = 0U; i < ds.count; ++i) {
            const CsvSample* s = &ds.samples[i];
            int ids[MAX_SEQ_LEN];
            size_t count = 0U;
            float logits[OUTPUT_DIM];
            float pred[OUTPUT_DIM];
            float grad[OUTPUT_DIM];
            size_t t = 0U;
            size_t j = 0U;
            memset(ids, 0, sizeof(ids));
            memset(logits, 0, sizeof(logits));
            memset(pred, 0, sizeof(pred));
            memset(grad, 0, sizeof(grad));
            rc = tokenizer_encode(&tokenizer, s->command, ids, MAX_SEQ_LEN, &count);
            if (rc != TOKENIZER_STATUS_OK || count == 0U) {
                vocab_free(&vocab);
                csv_free_dataset(&ds);
                return WORKFLOW_STATUS_DATA_ERROR;
            }
            predict_logits(weights, ids, count, s->state, logits);
            if (activate_output(logits, pred) != WORKFLOW_STATUS_OK) {
                vocab_free(&vocab);
                csv_free_dataset(&ds);
                return WORKFLOW_STATUS_INTERNAL_ERROR;
            }
            for (j = 0U; j < OUTPUT_DIM; ++j) {
                float err = pred[j] - s->target[j];
                grad[j] = err;
                loss_sum += err * err;
            }
            for (t = 0U; t < count; ++t) {
                size_t id = (ids[t] >= 0) ? (size_t)ids[t] : 0U;
                if (id >= VOCAB_SIZE) {
                    id = 0U;
                }
                for (j = 0U; j < OUTPUT_DIM; ++j) {
                    weights[id * OUTPUT_DIM + j] -= (options->learning_rate * grad[j]) / (float)count;
                }
            }
            for (t = 0U; t < STATE_DIM; ++t) {
                size_t base = TOKEN_WEIGHT_COUNT + t * OUTPUT_DIM;
                for (j = 0U; j < OUTPUT_DIM; ++j) {
                    weights[base + j] -= options->learning_rate * grad[j] * s->state[t];
                }
            }
            for (j = 0U; j < OUTPUT_DIM; ++j) {
                weights[TOKEN_WEIGHT_COUNT + STATE_WEIGHT_COUNT + j] -= options->learning_rate * grad[j];
            }
        }
        printf("epoch=%zu avg_loss=%.6f\n",
               epoch + 1U,
               (double)(loss_sum / (float)(ds.count * OUTPUT_DIM)));
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

int workflow_run_goal_loop(const WorkflowLoopOptions* options) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    DriverStub driver;
    float* weights = NULL;
    size_t weight_count = 0U;
    float x = 0.0f;
    float y = 0.0f;
    size_t frame = 0U;
    int rc = 0;
    if (options == NULL || options->vocab_path == NULL || options->weights_bin_path == NULL || options->max_frames == 0U) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    if (STATE_DIM != 8 || OUTPUT_DIM != 4) {
        return WORKFLOW_STATUS_INVALID_ARGUMENT;
    }
    memset(&vocab, 0, sizeof(vocab));
    memset(&tokenizer, 0, sizeof(tokenizer));
    memset(&driver, 0, sizeof(driver));
    rc = weights_load_binary(options->weights_bin_path, &weights, &weight_count);
    if (rc != WEIGHTS_IO_STATUS_OK || weights == NULL || weight_count != TOTAL_WEIGHT_COUNT) {
        free(weights);
        return WORKFLOW_STATUS_IO_ERROR;
    }
    rc = workflow_prepare_tokenizer(options->vocab_path, &vocab, &tokenizer);
    if (rc != WORKFLOW_STATUS_OK) {
        free(weights);
        return rc;
    }
    rc = driver_stub_init(&driver, options->driver_type);
    if (rc != DRIVER_STATUS_OK) {
        vocab_free(&vocab);
        free(weights);
        return WORKFLOW_STATUS_INTERNAL_ERROR;
    }
    for (frame = 0U; frame < options->max_frames; ++frame) {
        float dx = options->goal_x - x;
        float dy = options->goal_y - y;
        float state[STATE_DIM];
        float act[OUTPUT_DIM];
        int ids[MAX_SEQ_LEN];
        size_t count = 0U;
        float logits[OUTPUT_DIM];
        const char* cmd = choose_cmd(dx);
        float sx = 0.0f;
        float sy = 0.0f;
        memset(state, 0, sizeof(state));
        memset(act, 0, sizeof(act));
        memset(ids, 0, sizeof(ids));
        memset(logits, 0, sizeof(logits));
        if (fabsf(dx) < 0.3f && fabsf(dy) < 0.3f) {
            printf("done frame=%zu pose=(%.3f,%.3f)\n", frame, (double)x, (double)y);
            break;
        }
        state[0] = x / 20.0f;
        state[1] = y / 20.0f;
        state[2] = dx / 20.0f;
        state[3] = dy / 20.0f;
        state[4] = fabsf(dx) / 20.0f;
        state[5] = fabsf(dy) / 20.0f;
        state[6] = (fabsf(dx) + fabsf(dy)) / 40.0f;
        state[7] = 1.0f;
        rc = tokenizer_encode(&tokenizer, cmd, ids, MAX_SEQ_LEN, &count);
        if (rc != TOKENIZER_STATUS_OK || count == 0U) {
            driver_stub_shutdown(&driver);
            vocab_free(&vocab);
            free(weights);
            return WORKFLOW_STATUS_DATA_ERROR;
        }
        predict_logits(weights, ids, count, state, logits);
        if (activate_output(logits, act) != WORKFLOW_STATUS_OK) {
            driver_stub_shutdown(&driver);
            vocab_free(&vocab);
            free(weights);
            return WORKFLOW_STATUS_INTERNAL_ERROR;
        }
        sx = act[2] * 0.5f;
        sy = ((act[3] + 1.0f) * 0.5f) * 0.4f;
        if (fabsf(sx) > fabsf(dx)) {
            sx = dx;
        }
        if (fabsf(sy) > fabsf(dy)) {
            sy = dy;
        }
        x += sx;
        y += sy;
        (void)driver_stub_apply(&driver, act, OUTPUT_DIM);
        printf("frame=%03zu cmd=%s act=(%.3f,%.3f,%.3f,%.3f) pose=(%.3f,%.3f)\n",
               frame + 1U, cmd,
               (double)act[0], (double)act[1], (double)act[2], (double)act[3],
               (double)x, (double)y);
    }
    driver_stub_shutdown(&driver);
    vocab_free(&vocab);
    free(weights);
    return WORKFLOW_STATUS_OK;
}
