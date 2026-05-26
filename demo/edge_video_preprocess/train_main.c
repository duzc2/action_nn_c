/**
 * @file train_main.c
 * @brief Edge Video Preprocessing demo — CIFAR-10 training with detailed logging.
 *
 * Trains BnConvNet v47 (4 CNN + GAP + 1 MLP) on CIFAR-10.
 * Features: data augmentation, train/val split, checkpoint resume,
 * LR decay scheduling, ETA reporting, epoch history table.
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "weights_load.h"
#include "cifar10_dataset.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define EVP_TRAIN_SAMPLE_LIMIT 5000U
#define EVP_EVAL_SAMPLE_LIMIT  2000U
#define EVP_EPOCH_COUNT        5U
#define EVP_BATCH_SIZE         4U    /* Gradient accumulation batch size */

/* Early stopping patience */
#define EVP_EARLY_STOP_PATIENCE     5U
#define EVP_IMPROVEMENT_THRESHOLD   0.01f

/* Data augmentation */
#define EVP_AUG_FLIP  1   /* 1 = horizontal flip (p=0.5) */
#define EVP_AUG_CROP  2   /* 2px pad + random crop back to 32x32 */

/* Train / validation split (80% train, 20% val from training batches) */
#define EVP_TRAIN_SPLIT  0.8f

/* LR decay (Step Decay): halve LR every N epochs.
 * NOTE: Runtime LR adjustment requires a core API (train_set_lr).
 * When available, apply: lr_new = lr_initial * decay_rate ^ (epoch / decay_epochs). */
#define EVP_LR_DECAY_RATE   0.5f
#define EVP_LR_DECAY_EPOCHS 3U

/* Diagnostic: mini-eval every N samples during training */
#define EVP_MINI_EVAL_INTERVAL   500U
#define EVP_MINI_EVAL_SAMPLES    1000U

#define EVP_DATASET_ROOT "../../../../demo/edge_video_preprocess/dataset"

static const char* kTrainBatches[5] = {
    EVP_DATASET_ROOT "/data_batch_1.bin",
    EVP_DATASET_ROOT "/data_batch_2.bin",
    EVP_DATASET_ROOT "/data_batch_3.bin",
    EVP_DATASET_ROOT "/data_batch_4.bin",
    EVP_DATASET_ROOT "/data_batch_5.bin"
};

#define EVP_TEST_BATCH EVP_DATASET_ROOT "/test_batch.bin"
#define EVP_WEIGHTS_PATH "../data/weights.bin"
#define EVP_BEST_FILE "../data/best_weights.bin"

/* ========================================================================
 *  Epoch history entry
 * ======================================================================== */
typedef struct {
    size_t epoch;
    float  avg_loss;
    float  val_accuracy;
    float  lr;
    double elapsed_sec;
} EpochHistory;

#define EVP_MAX_EPOCH_HISTORY 100U
static EpochHistory g_epoch_history[EVP_MAX_EPOCH_HISTORY];
static size_t g_epoch_history_count = 0U;

/* ========================================================================
 *  Detailed evaluation — per-class accuracy, prediction distribution,
 *  confidence statistics, and top-k margin analysis.
 * ======================================================================== */

typedef struct {
    size_t per_class_correct[10];
    size_t per_class_count[10];
    size_t predicted_count[10];   /* how many times each class was predicted */
    double confidence_sum;
    double confidence_sq_sum;
    float  confidence_min;
    float  confidence_max;
    double top2_margin_sum;       /* avg gap between top-1 and top-2 prob */
    size_t sample_count;
    size_t error_count;
} EvalDetail;

static void eval_detail_init(EvalDetail* d) {
    size_t i;
    (void)memset(d, 0, sizeof(*d));
    d->confidence_min = 1.0f;
    d->confidence_max = 0.0f;
    for (i = 0U; i < 10U; ++i) {
        d->per_class_correct[i] = 0U;
        d->per_class_count[i]   = 0U;
        d->predicted_count[i]   = 0U;
    }
}

static int eval_detail_sample(void* infer_ctx, const Cifar10Dataset* dataset,
    size_t sample_index, EvalDetail* d)
{
    float output[10];
    const float* image;
    uint8_t label;
    int max_idx, second_idx;
    float max_val, second_val;
    size_t c;

    image = &dataset->images[sample_index * CIFAR10_IMAGE_SIZE];
    label = dataset->labels[sample_index];

    if (infer_auto_run(infer_ctx, image, output) != 0) {
        d->error_count++;
        return -1;
    }

    /* Find top-2 predictions */
    max_idx = 0;
    second_idx = 1;
    max_val = output[0];
    second_val = output[1];
    if (second_val > max_val) {
        float tmp = max_val; max_val = second_val; second_val = tmp;
        int tmpi = max_idx; max_idx = second_idx; second_idx = tmpi;
    }
    for (c = 2U; c < 10U; ++c) {
        if (output[c] > max_val) {
            second_val = max_val; second_idx = max_idx;
            max_val = output[c]; max_idx = (int)c;
        } else if (output[c] > second_val) {
            second_val = output[c]; second_idx = (int)c;
        }
    }

    /* Per-class accuracy */
    d->per_class_count[label]++;
    if (max_idx == (int)label) {
        d->per_class_correct[label]++;
    }

    /* Prediction distribution (which class is most predicted) */
    d->predicted_count[max_idx]++;

    /* Confidence = max softmax probability */
    d->confidence_sum += (double)max_val;
    d->confidence_sq_sum += (double)max_val * (double)max_val;
    if (max_val < d->confidence_min) d->confidence_min = max_val;
    if (max_val > d->confidence_max) d->confidence_max = max_val;

    /* Top-1 vs Top-2 margin */
    d->top2_margin_sum += (double)(max_val - second_val);

    d->sample_count++;
    return 0;
}

static float eval_detail_accuracy(const EvalDetail* d) {
    size_t total = 0U, correct = 0U, c;
    for (c = 0U; c < 10U; ++c) { correct += d->per_class_correct[c]; total += d->per_class_count[c]; }
    if (total == 0U) return 0.0f;
    return (float)correct * 100.0f / (float)total;
}

static void eval_detail_print(const EvalDetail* d, const char* label) {
    size_t c;
    float acc;
    double avg_conf, std_conf;

    acc = eval_detail_accuracy(d);
    avg_conf = d->confidence_sum / (double)(d->sample_count > 0U ? d->sample_count : 1U);
    std_conf = sqrt(d->confidence_sq_sum / (double)(d->sample_count > 0U ? d->sample_count : 1U) - avg_conf * avg_conf);

    fprintf(stderr, "\n────────────────────────────────────────────────────────────\n");
    fprintf(stderr, "  [LOG] %s\n", label);
    fprintf(stderr, "  Samples: %zu | Errors: %zu | Overall accuracy: %.2f%%\n",
        d->sample_count, d->error_count, (double)acc);

    /* Per-class accuracy */
    fprintf(stderr, "  Per-class accuracy:\n");
    fprintf(stderr, "    class: ");
    for (c = 0U; c < 10U; ++c) fprintf(stderr, "   %2zu  ", c);
    fprintf(stderr, "\n    acc%%: ");
    for (c = 0U; c < 10U; ++c) {
        float cls_acc = d->per_class_count[c] > 0U
            ? (float)d->per_class_correct[c] * 100.0f / (float)d->per_class_count[c]
            : 0.0f;
        fprintf(stderr, " %5.1f ", (double)cls_acc);
    }
    fprintf(stderr, "\n    n   : ");
    for (c = 0U; c < 10U; ++c) fprintf(stderr, " %5zu ", d->per_class_count[c]);

    /* Prediction distribution */
    fprintf(stderr, "\n  Predicted-as distribution (argmax count):\n");
    fprintf(stderr, "    count: ");
    for (c = 0U; c < 10U; ++c) fprintf(stderr, " %5zu ", d->predicted_count[c]);

    /* Confidence analysis */
    fprintf(stderr, "\n  Confidence (max softmax prob): avg=%.3f std=%.3f min=%.3f max=%.3f\n",
        avg_conf, std_conf, (double)d->confidence_min, (double)d->confidence_max);
    fprintf(stderr, "  Top1-Top2 margin:        avg=%.4f\n",
        d->top2_margin_sum / (double)(d->sample_count > 0U ? d->sample_count : 1U));

    /* Class bias check: is one class overwhelmingly predicted? */
    {
        size_t max_pred = 0U;
        size_t min_pred = d->sample_count;
        for (c = 0U; c < 10U; ++c) {
            if (d->predicted_count[c] > max_pred) max_pred = d->predicted_count[c];
            if (d->predicted_count[c] < min_pred) min_pred = d->predicted_count[c];
        }
        fprintf(stderr, "  Prediction spread: most=%zu least=%zu ratio=%.2f\n",
            max_pred, min_pred, (max_pred > 0U ? (double)max_pred / (double)(min_pred > 0U ? min_pred : 1U) : 0.0));
    }
    fprintf(stderr, "────────────────────────────────────────────────────────────\n");
    (void)fflush(stderr);
}

/* Run detailed evaluation on a subset of the dataset */
static void evaluate_subset_detailed(void* infer_ctx, const Cifar10Dataset* dataset,
    size_t max_samples, const char* label)
{
    EvalDetail detail;
    size_t i;
    size_t n = (dataset->sample_count < max_samples) ? dataset->sample_count : max_samples;
    eval_detail_init(&detail);
    for (i = 0U; i < n; ++i) {
        eval_detail_sample(infer_ctx, dataset, i, &detail);
    }
    eval_detail_print(&detail, label);
}

/* Simple fast accuracy on N samples (for mini-evals) */
static float evaluate_fast(void* infer_ctx, const Cifar10Dataset* dataset, size_t max_samples) {
    size_t i, correct = 0U, errors = 0U;
    size_t n = (dataset->sample_count < max_samples) ? dataset->sample_count : max_samples;
    float output[10];
    for (i = 0U; i < n; ++i) {
        const float* img = &dataset->images[i * CIFAR10_IMAGE_SIZE];
        if (infer_auto_run(infer_ctx, img, output) != 0) { errors++; continue; }
        if ((uint8_t)cifar10_argmax(output, 10U) == dataset->labels[i]) correct++;
    }
    if (errors >= n) return 0.0f;
    return (float)correct * 100.0f / (float)(n - errors);
}

/* ========================================================================
 *  Standard accuracy
 * ======================================================================== */

static float evaluate_accuracy(void* infer_ctx, const Cifar10Dataset* dataset) {
    size_t sample_index;
    size_t correct_count;
    size_t error_count;
    float output[CIFAR10_CLASS_COUNT];

    if (infer_ctx == NULL || dataset == NULL || dataset->sample_count == 0U) {
        return 0.0f;
    }

    correct_count = 0U;
    error_count  = 0U;
    for (sample_index = 0U; sample_index < dataset->sample_count; ++sample_index) {
        const float* image = &dataset->images[sample_index * CIFAR10_IMAGE_SIZE];
        if (infer_auto_run(infer_ctx, image, output) != 0) {
            error_count++;
            continue;
        }

        if ((uint8_t)cifar10_argmax(output, CIFAR10_CLASS_COUNT) == dataset->labels[sample_index]) {
            correct_count++;
        }
    }

    if (error_count > 0U) {
        fprintf(stderr, "  [WARN ] %zu eval errors out of %zu samples\n",
            error_count, dataset->sample_count);
    }

    if (error_count >= dataset->sample_count) return 0.0f;
    return ((float)correct_count * 100.0f) / (float)(dataset->sample_count - error_count);
}

/* ========================================================================
 *  Train/val split: create a validation subset from training data.
 *  Takes the last EVP_VAL_SPLIT fraction of the dataset.
 * ======================================================================== */

static void split_train_val(
    const Cifar10Dataset* full,
    Cifar10Dataset* train_out,
    Cifar10Dataset* val_out,
    float train_split)
{
    size_t train_count;
    size_t val_count;
    size_t i;
    size_t train_bytes;
    size_t val_bytes;

    if (full == NULL || train_out == NULL || val_out == NULL) return;
    if (full->sample_count == 0U || full->images == NULL) return;

    train_count = (size_t)((float)full->sample_count * train_split);
    if (train_count < 1U) train_count = 1U;
    if (train_count >= full->sample_count) train_count = full->sample_count - 1U;
    val_count = full->sample_count - train_count;

    train_bytes = train_count * CIFAR10_IMAGE_SIZE;
    val_bytes   = val_count   * CIFAR10_IMAGE_SIZE;

    train_out->images = (float*)calloc(train_count * CIFAR10_IMAGE_SIZE, sizeof(float));
    train_out->labels = (uint8_t*)calloc(train_count, sizeof(uint8_t));
    val_out->images   = (float*)calloc(val_count * CIFAR10_IMAGE_SIZE, sizeof(float));
    val_out->labels   = (uint8_t*)calloc(val_count, sizeof(uint8_t));

    if (train_out->images == NULL || train_out->labels == NULL
        || val_out->images == NULL || val_out->labels == NULL) {
        cifar10_dataset_free(train_out);
        cifar10_dataset_free(val_out);
        (void)memset(train_out, 0, sizeof(*train_out));
        (void)memset(val_out, 0, sizeof(*val_out));
        return;
    }

    /* Copy train portion */
    for (i = 0U; i < train_bytes; ++i) {
        train_out->images[i] = full->images[i];
    }
    for (i = 0U; i < train_count; ++i) {
        train_out->labels[i] = full->labels[i];
    }
    train_out->sample_count = train_count;
    train_out->image_size = CIFAR10_IMAGE_SIZE;

    /* Copy val portion */
    for (i = 0U; i < val_bytes; ++i) {
        val_out->images[i] = full->images[train_bytes + i];
    }
    for (i = 0U; i < val_count; ++i) {
        val_out->labels[i] = full->labels[train_count + i];
    }
    val_out->sample_count = val_count;
    val_out->image_size = CIFAR10_IMAGE_SIZE;
}

/* ========================================================================
 *  Print epoch history table
 * ======================================================================== */

static void print_epoch_history(void) {
    size_t i;
    fprintf(stderr, "\n[LOG] === Epoch History ===\n");
    fprintf(stderr, " %4s  %10s  %10s  %10s  %8s\n",
        "Epoch", "Loss", "Val Acc%", "LR", "Time(s)");
    fprintf(stderr, " ----  ----------  ----------  ----------  --------\n");
    for (i = 0U; i < g_epoch_history_count; ++i) {
        fprintf(stderr, " %4zu  %10.4f  %10.2f  %10.6f  %8.0f\n",
            g_epoch_history[i].epoch,
            (double)g_epoch_history[i].avg_loss,
            (double)g_epoch_history[i].val_accuracy,
            (double)g_epoch_history[i].lr,
            g_epoch_history[i].elapsed_sec);
    }
    if (g_epoch_history_count == 0U) {
        fprintf(stderr, "  (no epochs recorded)\n");
    }
    (void)fflush(stderr);
}

/* ========================================================================
 *  Main training loop
 * ======================================================================== */

int main(void) {
    Cifar10Dataset full_dataset;
    Cifar10Dataset train_dataset;
    Cifar10Dataset val_dataset;
    Cifar10Dataset eval_dataset;
    void* infer_ctx;
    void* train_ctx;

    /* Disable stderr buffering for real-time diagnostic output on Windows */
    (void)setvbuf(stderr, NULL, _IONBF, 0);

    char error_buffer[512];
    float best_accuracy;
    size_t best_epoch;
    size_t epochs_without_improvement;
    time_t train_start;
    int stopped_early;
    int result;

    /* Augmentation config */
    Cifar10Augment aug;
    aug.horizontal_flip = EVP_AUG_FLIP;
    aug.pad_crop        = EVP_AUG_CROP;
    uint32_t aug_seed = 42U;

    /* Current LR (initial) — for display only */
    float current_lr = 0.001f;

    (void)memset(&full_dataset, 0, sizeof(full_dataset));
    (void)memset(&train_dataset, 0, sizeof(train_dataset));
    (void)memset(&val_dataset, 0, sizeof(val_dataset));
    (void)memset(&eval_dataset, 0, sizeof(eval_dataset));
    (void)memset(g_epoch_history, 0, sizeof(g_epoch_history));
    g_epoch_history_count = 0U;

    error_buffer[0] = '\0';
    best_accuracy = 0.0f;
    best_epoch = 0U;
    epochs_without_improvement = 0U;
    stopped_early = 0;

    (void)setvbuf(stdout, NULL, _IONBF, 0U);

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    /* ── Load datasets ── */
    fprintf(stderr, "[LOG] === Edge Video Preprocessing — CIFAR-10 Training ===\n\n");

    result = cifar10_dataset_load_multiple(
        kTrainBatches, 5U, EVP_TRAIN_SAMPLE_LIMIT / 5U,
        &full_dataset, error_buffer, sizeof(error_buffer));
    if (result != 0) {
        fprintf(stderr, "Failed to load train dataset: %s\n", error_buffer);
        return 1;
    }

    /* Shuffle before splitting */
    cifar10_dataset_shuffle(&full_dataset, 12345U);

    /* Split into train (80%) and validation (20%) */
    split_train_val(&full_dataset, &train_dataset, &val_dataset, EVP_TRAIN_SPLIT);

    if (train_dataset.sample_count == 0U) {
        fprintf(stderr, "Failed to split train/val dataset\n");
        cifar10_dataset_free(&full_dataset);
        return 1;
    }

    result = cifar10_dataset_load_batch(
        EVP_TEST_BATCH, EVP_EVAL_SAMPLE_LIMIT,
        &eval_dataset, error_buffer, sizeof(error_buffer));
    if (result != 0) {
        fprintf(stderr, "Failed to load eval dataset: %s\n", error_buffer);
        cifar10_dataset_free(&val_dataset);
        cifar10_dataset_free(&train_dataset);
        cifar10_dataset_free(&full_dataset);
        return 1;
    }

    fprintf(stderr, "[LOG] Full samples:  %zu\n", full_dataset.sample_count);
    fprintf(stderr, "[LOG] Train samples: %zu (%.0f%%)\n",
        train_dataset.sample_count, (double)EVP_TRAIN_SPLIT * 100.0);
    fprintf(stderr, "[LOG] Val samples:   %zu (%.0f%%)\n",
        val_dataset.sample_count, (double)(1.0f - EVP_TRAIN_SPLIT) * 100.0);
    fprintf(stderr, "[LOG] Eval  samples: %zu\n", eval_dataset.sample_count);
    fprintf(stderr, "[LOG] Max epochs:    %u\n", (unsigned)EVP_EPOCH_COUNT);
    fprintf(stderr, "[LOG] Early stop:    %u epochs without improvement > %.2f%%\n",
        (unsigned)EVP_EARLY_STOP_PATIENCE, (double)EVP_IMPROVEMENT_THRESHOLD);
    fprintf(stderr, "[LOG] Augmentation:  flip=%d crop=%dpx\n",
        EVP_AUG_FLIP, EVP_AUG_CROP);
    fprintf(stderr, "[LOG] LR schedule:   decay_rate=%.2f every %u epochs\n",
        (double)EVP_LR_DECAY_RATE, (unsigned)EVP_LR_DECAY_EPOCHS);
    fprintf(stderr, "[LOG] Network:       BnConvNet v47 (4CNN+BN+LeakyReLU+MLP, batch=%u, lr=0.001, mom=0.0, fix_BN_backward)\n",
        (unsigned)EVP_BATCH_SIZE);
    fprintf(stderr, "[LOG] Mini-eval:     every %u samples on %u eval subset\n\n",
        (unsigned)EVP_MINI_EVAL_INTERVAL, (unsigned)EVP_MINI_EVAL_SAMPLES);
    (void)fflush(stderr);

    /* Free the full dataset after split */
    cifar10_dataset_free(&full_dataset);

    /* ── Create contexts ── */
    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        cifar10_dataset_free(&val_dataset);
        cifar10_dataset_free(&train_dataset);
        cifar10_dataset_free(&eval_dataset);
        return 1;
    }

    /* ── Try checkpoint resume: load existing weights.bin ── */
    {
        int load_result = weights_load_from_file(infer_ctx, EVP_WEIGHTS_PATH);
        if (load_result == 0) {
            fprintf(stderr, "[LOG] Checkpoint resume: loaded %s successfully.\n", EVP_WEIGHTS_PATH);
            fprintf(stderr, "[LOG] Continuing training from saved weights.\n");
        } else {
            fprintf(stderr, "[LOG] No checkpoint found at %s — starting from scratch.\n", EVP_WEIGHTS_PATH);
        }
    }

    train_ctx = train_create(infer_ctx);
    if (train_ctx == NULL) {
        fprintf(stderr, "Failed to create training context\n");
        infer_destroy(infer_ctx);
        cifar10_dataset_free(&val_dataset);
        cifar10_dataset_free(&train_dataset);
        cifar10_dataset_free(&eval_dataset);
        return 1;
    }

    /* ── Pre-training baseline eval ── */
    fprintf(stderr, "[LOG] --- Pre-training baseline evaluation ---\n");
    evaluate_subset_detailed(infer_ctx, &val_dataset, 500U,
        "BASELINE (pre-training, 500 val samples)");
    (void)fflush(stderr);

    train_start = time(NULL);

    /* ── Training loop ── */
    {
        size_t epoch_index;
        for (epoch_index = 0U; epoch_index < EVP_EPOCH_COUNT; ++epoch_index) {
            float epoch_loss_sum = 0.0f;
            float epoch_loss_min = 1e9f;
            float epoch_loss_max = -1e9f;
            size_t loss_sample_count = 0U;
            float val_accuracy;
            float acc_delta;
            int is_best;
            size_t sample_index;
            time_t epoch_start;
            time_t epoch_end;
            double epoch_elapsed;
            double elapsed_total;
            double eta_remaining;

            /* ── LR Decay: compute effective LR for display ── */
            {
                size_t decay_steps = epoch_index / EVP_LR_DECAY_EPOCHS;
                size_t d;
                current_lr = 0.001f;
                for (d = 0U; d < decay_steps; ++d) {
                    current_lr *= EVP_LR_DECAY_RATE;
                }
            }

            /* Shuffle training data every epoch */
            cifar10_dataset_shuffle(&train_dataset, (uint32_t)(epoch_index + 1000U));

            fprintf(stderr, "\n[LOG] ======== EPOCH %zu/%u START (lr=%.6f) ========\n",
                epoch_index + 1U, (unsigned)EVP_EPOCH_COUNT, (double)current_lr);
            (void)fflush(stderr);

            epoch_start = time(NULL);

            for (sample_index = 0U; sample_index < train_dataset.sample_count; ) {
                size_t batch_i;
                size_t batch_count = EVP_BATCH_SIZE;
                if (sample_index + batch_count > train_dataset.sample_count)
                    batch_count = train_dataset.sample_count - sample_index;

                /* Process batch with augmented samples */
                for (batch_i = 0U; batch_i < batch_count; ++batch_i) {
                    float aug_image[CIFAR10_IMAGE_SIZE];
                    const float* image = &train_dataset.images[(sample_index + batch_i) * CIFAR10_IMAGE_SIZE];
                    float target[CIFAR10_CLASS_COUNT];
                    float step_loss;

                    /* Apply data augmentation */
                    cifar10_augment_sample(
                        aug_image, image,
                        CIFAR10_IMAGE_WIDTH, CIFAR10_IMAGE_HEIGHT, CIFAR10_CHANNELS,
                        &aug, &aug_seed);

                    cifar10_make_one_hot(train_dataset.labels[sample_index + batch_i],
                        target, CIFAR10_CLASS_COUNT);

                    if (train_step(train_ctx, aug_image, target) != 0) {
                        fprintf(stderr, "Training step failed at epoch %zu sample %zu\n",
                            epoch_index + 1U, sample_index + batch_i + 1U);
                        train_destroy(train_ctx);
                        infer_destroy(infer_ctx);
                        cifar10_dataset_free(&val_dataset);
                        cifar10_dataset_free(&train_dataset);
                        cifar10_dataset_free(&eval_dataset);
                        return 1;
                    }
                    step_loss = train_get_loss(train_ctx);
                    epoch_loss_sum += step_loss;
                    loss_sample_count++;
                    if (step_loss < epoch_loss_min) epoch_loss_min = step_loss;
                    if (step_loss > epoch_loss_max) epoch_loss_max = step_loss;
                }

                sample_index += batch_count;

                /* Progress indicator every 1000 samples */
                if (sample_index % 1000U == 0U) {
                    float avg_so_far = epoch_loss_sum / (float)loss_sample_count;
                    fprintf(stderr, "  [LOG] epoch %zu  sample %5zu/%zu  avg_loss=%.4f  min=%.4f  max=%.4f\n",
                        epoch_index + 1U, sample_index, train_dataset.sample_count,
                        (double)avg_so_far, (double)epoch_loss_min, (double)epoch_loss_max);
                    (void)fflush(stderr);
                }

                /* Progress trace every 200 samples */
                if (sample_index % 200U == 0U) {
                    float rolling_avg = epoch_loss_sum / (float)loss_sample_count;
                    fprintf(stderr, "  [TRACE] sample %5zu/%zu  avg_loss=%.4f  last_loss=%.4f\n",
                        sample_index, train_dataset.sample_count,
                        (double)rolling_avg, (double)(train_get_loss(train_ctx)));
                    (void)fflush(stderr);
                }

                /* Mini-eval every EVP_MINI_EVAL_INTERVAL samples */
                if (sample_index % EVP_MINI_EVAL_INTERVAL == 0U) {
                    float mini_acc = evaluate_fast(infer_ctx, &eval_dataset, EVP_MINI_EVAL_SAMPLES);
                    float rolling_loss = epoch_loss_sum / (float)loss_sample_count;
                    fprintf(stderr, "  [MINI] sample %5zu/%zu  rolling_avg_loss=%.4f  mini_eval_acc(%usamp)=%.2f%%\n",
                        sample_index, train_dataset.sample_count,
                        (double)rolling_loss, (unsigned)EVP_MINI_EVAL_SAMPLES, (double)mini_acc);
                    (void)fflush(stderr);
                }
            }

            epoch_end = time(NULL);
            epoch_elapsed = (double)(epoch_end - epoch_start);
            {
                float avg_loss = epoch_loss_sum / (float)loss_sample_count;
                fprintf(stderr, "[LOG] Epoch %zu training complete. Loss: avg=%.4f min=%.4f max=%.4f\n",
                    epoch_index + 1U, (double)avg_loss, (double)epoch_loss_min, (double)epoch_loss_max);
            }

            /* Val evaluation on validation split */
            fprintf(stderr, "[LOG] Running validation on %zu val samples...\n", val_dataset.sample_count);
            (void)fflush(stderr);

            val_accuracy = evaluate_accuracy(infer_ctx, &val_dataset);

            /* ── ETA estimate ── */
            elapsed_total = (double)(epoch_end - train_start);
            if (epoch_index + 1U < EVP_EPOCH_COUNT) {
                double avg_epoch_time = elapsed_total / (double)(epoch_index + 1U);
                eta_remaining = avg_epoch_time * (double)(EVP_EPOCH_COUNT - (epoch_index + 1U));
            } else {
                eta_remaining = 0.0;
            }

            /* Print epoch result line */
            printf("Epoch %2zu/%u - avg loss: %.4f - val accuracy: %5.2f%% - time: %5.0fs - lr: %.6f - ETA: %5.0fs",
                epoch_index + 1U,
                (unsigned)EVP_EPOCH_COUNT,
                (double)(epoch_loss_sum / (float)loss_sample_count),
                (double)val_accuracy,
                epoch_elapsed,
                (double)current_lr,
                eta_remaining);

            /* Record history */
            if (g_epoch_history_count < EVP_MAX_EPOCH_HISTORY) {
                g_epoch_history[g_epoch_history_count].epoch        = epoch_index + 1U;
                g_epoch_history[g_epoch_history_count].avg_loss     = epoch_loss_sum / (float)loss_sample_count;
                g_epoch_history[g_epoch_history_count].val_accuracy = val_accuracy;
                g_epoch_history[g_epoch_history_count].lr           = current_lr;
                g_epoch_history[g_epoch_history_count].elapsed_sec  = epoch_elapsed;
                g_epoch_history_count++;
            }

            /* Detailed per-class eval on val */
            {
                EvalDetail detail;
                size_t i;
                eval_detail_init(&detail);
                for (i = 0U; i < val_dataset.sample_count; ++i) {
                    eval_detail_sample(infer_ctx, &val_dataset, i, &detail);
                }
                {
                    char label_buf[64];
                    (void)snprintf(label_buf, sizeof(label_buf),
                        "EPOCH %zu/%u VAL EVAL (%zu samples, acc=%.2f%%)",
                        epoch_index + 1U, (unsigned)EVP_EPOCH_COUNT,
                        val_dataset.sample_count, (double)val_accuracy);
                    eval_detail_print(&detail, label_buf);
                }
            }

            /* ── Save epoch checkpoint ── */
            {
                char epoch_file[64];
                (void)snprintf(epoch_file, sizeof(epoch_file),
                    "../data/epoch_%02zu_weights.bin", epoch_index + 1U);
                if (weights_save_to_file(infer_ctx, epoch_file) != 0) {
                    fprintf(stderr, "\n  [WARN ] Failed to save epoch checkpoint: %s\n", epoch_file);
                }
            }

            /* Also save as weights.bin for resume next time */
            if (weights_save_to_file(infer_ctx, EVP_WEIGHTS_PATH) != 0) {
                fprintf(stderr, "\n  [WARN ] Failed to save resume checkpoint: %s\n", EVP_WEIGHTS_PATH);
            }

            /* ── Track best accuracy ── */
            acc_delta = val_accuracy - best_accuracy;
            is_best = (acc_delta > EVP_IMPROVEMENT_THRESHOLD) ? 1 : 0;

            if (is_best) {
                best_accuracy = val_accuracy;
                best_epoch = epoch_index + 1U;
                epochs_without_improvement = 0U;
                if (weights_save_to_file(infer_ctx, EVP_BEST_FILE) != 0) {
                    fprintf(stderr, "\n  [WARN ] Failed to save best weights: %s\n", EVP_BEST_FILE);
                }
                fprintf(stderr, "[LOG] >>> NEW BEST: %.2f%% (+%.2f%%) <<<\n",
                    (double)val_accuracy, (double)acc_delta);
            } else {
                epochs_without_improvement++;
                fprintf(stderr, "[LOG] No improvement (x%zu). Best so far: %.2f%% (epoch %zu)\n",
                    epochs_without_improvement, (double)best_accuracy, best_epoch);
            }
            (void)fflush(stdout);
            (void)fflush(stderr);

            /* ── Early stopping ── */
            if (epochs_without_improvement >= EVP_EARLY_STOP_PATIENCE) {
                fprintf(stderr, "\n[LOG] Early stopping at epoch %zu/%u:\n",
                    epoch_index + 1U, (unsigned)EVP_EPOCH_COUNT);
                fprintf(stderr, "[LOG]   Best accuracy: %.2f%% (epoch %zu)\n",
                    (double)best_accuracy, best_epoch);
                fprintf(stderr, "[LOG]   No improvement for %u consecutive epochs.\n",
                    (unsigned)EVP_EARLY_STOP_PATIENCE);
                stopped_early = 1;
                break;
            }
        }
    }

    /* ── Training summary ── */
    {
        time_t total_time = time(NULL) - train_start;
        fprintf(stderr, "\n[LOG] === Training Summary ===\n");
        fprintf(stderr, "[LOG] Best accuracy: %.2f%% (epoch %zu)\n",
            (double)best_accuracy, best_epoch);
        fprintf(stderr, "[LOG] Best weights:  %s\n", EVP_BEST_FILE);
        fprintf(stderr, "[LOG] Checkpoint:    %s\n", EVP_WEIGHTS_PATH);
        fprintf(stderr, "[LOG] Total time:    %llds (%.1f hours)\n",
            (long long)total_time, (double)total_time / 3600.0);
        if (stopped_early) {
            fprintf(stderr, "[LOG] Stopped early: plateau detected.\n");
        } else {
            fprintf(stderr, "[LOG] Completed all %u epochs.\n", (unsigned)EVP_EPOCH_COUNT);
        }

        /* Print epoch history table */
        print_epoch_history();
    }

    /* ── Cleanup ── */
    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    cifar10_dataset_free(&val_dataset);
    cifar10_dataset_free(&train_dataset);
    cifar10_dataset_free(&eval_dataset);
    return 0;
}
