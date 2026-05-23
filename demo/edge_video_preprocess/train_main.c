/**
 * @file train_main.c
 * @brief Edge Video Preprocessing demo — CIFAR-10 training with detailed logging.
 *
 * Trains a SimpleConvNet (7 CNN no-BN + 1 MLP) on CIFAR-10.
 * Extensive logging for diagnosis: per-class accuracy, confidence distribution,
 * loss progression, within-epoch mini-evals, prediction bias tracking.
 */

#include "infer.h"
#include "train.h"
#include "weights_save.h"
#include "cifar10_dataset.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define EVP_TRAIN_SAMPLE_LIMIT 50000U
#define EVP_EVAL_SAMPLE_LIMIT  10000U
#define EVP_EPOCH_COUNT        30U

/* Early stopping patience */
#define EVP_EARLY_STOP_PATIENCE     5U
#define EVP_IMPROVEMENT_THRESHOLD   0.01f

/* Diagnostic: mini-eval every N samples during training */
#define EVP_MINI_EVAL_INTERVAL   10000U
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
 *  Standard accuracy (kept for backward compatibility with original logic)
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
 *  Main training loop with extensive logging
 * ======================================================================== */

int main(void) {
    Cifar10Dataset train_dataset;
    Cifar10Dataset eval_dataset;
    void* infer_ctx;
    void* train_ctx;
    char error_buffer[512];
    const char* best_file = "../data/best_weights.bin";
    size_t epoch_index;
    size_t sample_index;
    float target[CIFAR10_CLASS_COUNT];
    float best_accuracy;
    size_t best_epoch;
    size_t epochs_without_improvement;
    float average_loss;
    time_t epoch_start;
    time_t epoch_end;
    time_t train_start;
    int stopped_early;
    int result;

    (void)memset(&train_dataset, 0, sizeof(train_dataset));
    (void)memset(&eval_dataset, 0, sizeof(eval_dataset));
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
        &train_dataset, error_buffer, sizeof(error_buffer));
    if (result != 0) {
        fprintf(stderr, "Failed to load train dataset: %s\n", error_buffer);
        return 1;
    }

    result = cifar10_dataset_load_batch(
        EVP_TEST_BATCH, EVP_EVAL_SAMPLE_LIMIT,
        &eval_dataset, error_buffer, sizeof(error_buffer));
    if (result != 0) {
        fprintf(stderr, "Failed to load eval dataset: %s\n", error_buffer);
        cifar10_dataset_free(&train_dataset);
        return 1;
    }

    fprintf(stderr, "[LOG] Train samples: %zu\n", train_dataset.sample_count);
    fprintf(stderr, "[LOG] Eval  samples: %zu\n", eval_dataset.sample_count);
    fprintf(stderr, "[LOG] Max epochs:    %u\n", (unsigned)EVP_EPOCH_COUNT);
    fprintf(stderr, "[LOG] Early stop:    %u epochs without improvement > %.2f%%\n",
        (unsigned)EVP_EARLY_STOP_PATIENCE, (double)EVP_IMPROVEMENT_THRESHOLD);
    fprintf(stderr, "[LOG] Network:       SimpleConvNet (7 CNN no-BN + 1 MLP), batch_size=4\n");
    fprintf(stderr, "[LOG] Mini-eval:     every %u samples on %u eval subset\n\n",
        (unsigned)EVP_MINI_EVAL_INTERVAL, (unsigned)EVP_MINI_EVAL_SAMPLES);
    (void)fflush(stderr);

    /* ── Create contexts ── */
    infer_ctx = infer_create();
    if (infer_ctx == NULL) {
        fprintf(stderr, "Failed to create inference context\n");
        cifar10_dataset_free(&eval_dataset);
        cifar10_dataset_free(&train_dataset);
        return 1;
    }

    train_ctx = train_create(infer_ctx);
    if (train_ctx == NULL) {
        fprintf(stderr, "Failed to create training context\n");
        infer_destroy(infer_ctx);
        cifar10_dataset_free(&eval_dataset);
        cifar10_dataset_free(&train_dataset);
        return 1;
    }

    /* ── Pre-training baseline eval ── */
    fprintf(stderr, "[LOG] --- Pre-training baseline evaluation ---\n");
    evaluate_subset_detailed(infer_ctx, &eval_dataset, 2000U, "BASELINE (pre-training, 2000 eval samples)");
    (void)fflush(stderr);

    train_start = time(NULL);

    /* ── Training loop ── */
    for (epoch_index = 0U; epoch_index < EVP_EPOCH_COUNT; ++epoch_index) {
        float epoch_loss_sum = 0.0f;
        float epoch_loss_min = 1e9f;
        float epoch_loss_max = -1e9f;
        size_t loss_sample_count = 0U;
        float eval_accuracy;
        float acc_delta;
        int is_best;

        /* Shuffle training data every epoch */
        cifar10_dataset_shuffle(&train_dataset, (uint32_t)epoch_index);

        fprintf(stderr, "\n[LOG] ======== EPOCH %zu/%u START ========\n",
            epoch_index + 1U, (unsigned)EVP_EPOCH_COUNT);
        (void)fflush(stderr);

        epoch_start = time(NULL);

        for (sample_index = 0U; sample_index < train_dataset.sample_count; ++sample_index) {
            const float* image = &train_dataset.images[sample_index * CIFAR10_IMAGE_SIZE];
            float step_loss;

            cifar10_make_one_hot(train_dataset.labels[sample_index], target, CIFAR10_CLASS_COUNT);

            if (train_step(train_ctx, image, target) != 0) {
                fprintf(stderr, "Training step failed at epoch %zu sample %zu\n",
                    epoch_index + 1U, sample_index + 1U);
                train_destroy(train_ctx);
                infer_destroy(infer_ctx);
                cifar10_dataset_free(&eval_dataset);
                cifar10_dataset_free(&train_dataset);
                return 1;
            }
            step_loss = train_get_loss(train_ctx);
            epoch_loss_sum += step_loss;
            loss_sample_count++;
            if (step_loss < epoch_loss_min) epoch_loss_min = step_loss;
            if (step_loss > epoch_loss_max) epoch_loss_max = step_loss;

            /* Progress indicator every 5000 samples */
            if ((sample_index + 1U) % 5000U == 0U) {
                float avg_so_far = epoch_loss_sum / (float)loss_sample_count;
                fprintf(stderr, "  [LOG] epoch %zu  sample %5zu/%zu  avg_loss=%.4f  min=%.4f  max=%.4f\n",
                    epoch_index + 1U, sample_index + 1U, train_dataset.sample_count,
                    (double)avg_so_far, (double)epoch_loss_min, (double)epoch_loss_max);
                (void)fflush(stderr);
            }

            /* Mini-eval every EVP_MINI_EVAL_INTERVAL samples */
            if ((sample_index + 1U) % EVP_MINI_EVAL_INTERVAL == 0U) {
                float mini_acc = evaluate_fast(infer_ctx, &eval_dataset, EVP_MINI_EVAL_SAMPLES);
                float rolling_loss = epoch_loss_sum / (float)loss_sample_count;
                fprintf(stderr, "  [MINI] sample %5zu/%zu  rolling_avg_loss=%.4f  mini_eval_acc(%usamp)=%.2f%%\n",
                    sample_index + 1U, train_dataset.sample_count,
                    (double)rolling_loss, (unsigned)EVP_MINI_EVAL_SAMPLES, (double)mini_acc);
                (void)fflush(stderr);
            }
        }

        epoch_end = time(NULL);
        average_loss = epoch_loss_sum / (float)train_dataset.sample_count;

        fprintf(stderr, "[LOG] Epoch %zu training complete. Loss: avg=%.4f min=%.4f max=%.4f\n",
            epoch_index + 1U, (double)average_loss, (double)epoch_loss_min, (double)epoch_loss_max);

        /* Full eval on 10000 test samples */
        fprintf(stderr, "[LOG] Running full evaluation on %zu test samples...\n", eval_dataset.sample_count);
        (void)fflush(stderr);

        eval_accuracy = evaluate_accuracy(infer_ctx, &eval_dataset);

        /* Print epoch result line */
        printf("Epoch %2zu/%u - avg loss: %.4f - eval accuracy: %5.2f%% - %llds",
            epoch_index + 1U,
            (unsigned)EVP_EPOCH_COUNT,
            (double)average_loss,
            (double)eval_accuracy,
            (long long)(epoch_end - epoch_start));

        /* Detailed per-class analysis every epoch */
        {
            EvalDetail detail;
            size_t i;
            eval_detail_init(&detail);
            for (i = 0U; i < eval_dataset.sample_count; ++i) {
                eval_detail_sample(infer_ctx, &eval_dataset, i, &detail);
            }
            {
                char label_buf[64];
                (void)snprintf(label_buf, sizeof(label_buf), "EPOCH %zu/%u FULL EVAL (10000 samples, acc=%.2f%%)",
                    epoch_index + 1U, (unsigned)EVP_EPOCH_COUNT, (double)eval_accuracy);
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

        /* ── Track best accuracy ── */
        acc_delta = eval_accuracy - best_accuracy;
        is_best = (acc_delta > EVP_IMPROVEMENT_THRESHOLD) ? 1 : 0;

        if (is_best) {
            best_accuracy = eval_accuracy;
            best_epoch = epoch_index + 1U;
            epochs_without_improvement = 0U;
            if (weights_save_to_file(infer_ctx, best_file) != 0) {
                fprintf(stderr, "\n  [WARN ] Failed to save best weights: %s\n", best_file);
            }
            fprintf(stderr, "[LOG] >>> NEW BEST: %.2f%% (+%.2f%%) <<<\n",
                (double)eval_accuracy, (double)acc_delta);
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

    /* ── Training summary ── */
    {
        time_t total_time = time(NULL) - train_start;
        fprintf(stderr, "\n[LOG] === Training Summary ===\n");
        fprintf(stderr, "[LOG] Best accuracy: %.2f%% (epoch %zu)\n",
            (double)best_accuracy, best_epoch);
        fprintf(stderr, "[LOG] Best weights:  %s\n", best_file);
        fprintf(stderr, "[LOG] Total time:    %llds (%.1f hours)\n",
            (long long)total_time, (double)total_time / 3600.0);
        if (stopped_early) {
            fprintf(stderr, "[LOG] Stopped early at epoch %zu/%u (plateau detected).\n",
                epoch_index + 1U, (unsigned)EVP_EPOCH_COUNT);
        } else {
            fprintf(stderr, "[LOG] Completed all %u epochs.\n", (unsigned)EVP_EPOCH_COUNT);
        }
    }

    /* ── Cleanup ── */
    train_destroy(train_ctx);
    infer_destroy(infer_ctx);
    cifar10_dataset_free(&eval_dataset);
    cifar10_dataset_free(&train_dataset);
    return 0;
}
