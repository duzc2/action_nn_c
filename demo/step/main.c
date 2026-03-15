#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "../../src/include/config_user.h"
#include "../../src/include/weights_io.h"
#include "../../src/infer/include/workflow_infer.h"
#include "../../src/train/include/workflow_train.h"

typedef struct Pose2D {
    float x;
    float y;
} Pose2D;

static int create_dir_if_missing(const char* path) {
#if defined(_WIN32)
    int rc = _mkdir(path);
#else
    int rc = mkdir(path, 0755);
#endif
    if (rc == 0 || errno == EEXIST) {
        return 0;
    }
    return -1;
}

static int ensure_step_data_dir(void) {
    if (create_dir_if_missing("demo") != 0) {
        return -1;
    }
    if (create_dir_if_missing("demo/step") != 0) {
        return -2;
    }
    if (create_dir_if_missing("demo/step/data") != 0) {
        return -3;
    }
    return 0;
}

static int write_rows_to_text(const char* file_path, const char* const* rows, size_t row_count) {
    FILE* fp = NULL;
    size_t i = 0U;
    if (file_path == NULL || rows == NULL || row_count == 0U) {
        return -1;
    }
    fp = fopen(file_path, "w");
    if (fp == NULL) {
        return -2;
    }
    for (i = 0U; i < row_count; ++i) {
        if (fprintf(fp, "%s\n", rows[i]) < 0) {
            fclose(fp);
            return -3;
        }
    }
    fclose(fp);
    return 0;
}

static int write_vocab(const char* file_path) {
    static const char* tokens[] = {
        "<unk>", "move", "left", "right", "stop", "fast", "slow"
    };
    return write_rows_to_text(file_path, tokens, sizeof(tokens) / sizeof(tokens[0]));
}

static const float kStepStates[][STATE_DIM] = {
    { 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }
};

static const float kStepTargets[][OUTPUT_DIM] = {
    { 1.0f, 0.0f, -0.8f, 0.0f },
    { 0.0f, 1.0f, 0.8f, 0.0f },
    { 1.0f, 0.0f, -1.0f, 0.5f },
    { 0.0f, 1.0f, 1.0f, 0.5f },
    { 1.0f, 0.0f, -0.4f, 0.0f },
    { 0.0f, 1.0f, 0.4f, 0.0f },
    { 0.0f, 0.0f, 0.0f, 0.0f },
    { 0.0f, 0.0f, 0.0f, -0.1f }
};

static const WorkflowTrainSample kStepSamples[] = {
    { "move left", kStepStates[0], kStepTargets[0] },
    { "move right", kStepStates[1], kStepTargets[1] },
    { "move left fast", kStepStates[2], kStepTargets[2] },
    { "move right fast", kStepStates[3], kStepTargets[3] },
    { "move left slow", kStepStates[4], kStepTargets[4] },
    { "move right slow", kStepStates[5], kStepTargets[5] },
    { "move stop", kStepStates[6], kStepTargets[6] },
    { "stop", kStepStates[7], kStepTargets[7] }
};

static float clamp_float(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static int clamp_int(int v, int lo, int hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static void clear_screen_draw_mode(void) {
    printf("\x1b[2J\x1b[H");
}

static void wait_enter_for_next_frame(void) {
    char linebuf[16];
    printf("Press Enter to compute next frame...");
    fflush(stdout);
    if (fgets(linebuf, sizeof(linebuf), stdin) == NULL) {
        clearerr(stdin);
    }
}

static void build_state_step_only(const Pose2D* pose, float* state) {
    memset(state, 0, sizeof(float) * STATE_DIM);
    state[0] = pose->x / 15.0f;
    state[1] = pose->y / 15.0f;
}

static void render_cli_step_frame(const Pose2D* pose,
                                  size_t frame,
                                  const char* command,
                                  const float* act) {
    char canvas[32][96];
    int width = 45;
    int height = 18;
    int x = 0;
    int y = 0;
    int pose_col = 0;
    int pose_row = 0;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            canvas[y][x] = '.';
        }
        canvas[y][width] = '\0';
    }
    pose_col = clamp_int((int)lroundf((pose->x / 15.0f) * (float)(width - 1)), 0, width - 1);
    pose_row = clamp_int((int)lroundf((pose->y / 15.0f) * (float)(height - 1)), 0, height - 1);
    canvas[height - 1 - pose_row][pose_col] = '@';
    printf("\x1b[H");
    printf("CLI Step Mode | frame=%03zu | cmd=\"%s\"\n", frame + 1U, command);
    printf("pose=(%.2f, %.2f) act=(%.3f, %.3f, %.3f, %.3f)\n",
           (double)pose->x, (double)pose->y,
           (double)act[0], (double)act[1], (double)act[2], (double)act[3]);
    printf("+");
    for (x = 0; x < width; ++x) {
        printf("-");
    }
    printf("+\n");
    for (y = 0; y < height; ++y) {
        printf("|%s|\n", canvas[y]);
    }
    printf("+");
    for (x = 0; x < width; ++x) {
        printf("-");
    }
    printf("+\n");
}

static int run_step_draw_mode(WorkflowRuntime* runtime, Pose2D* io_pose, size_t frames) {
    static const char* commands[] = {
        "move left",
        "move right",
        "move left fast",
        "move right fast",
        "move left slow",
        "move right slow",
        "move stop",
        "stop",
        "move left fast now",
        "move right slow now"
    };
    size_t frame = 0U;
    if (runtime == NULL || io_pose == NULL || frames == 0U) {
        return -1;
    }
    srand((unsigned int)time(NULL));
    clear_screen_draw_mode();
    for (frame = 0U; frame < frames; ++frame) {
        const char* command = commands[rand() % (sizeof(commands) / sizeof(commands[0]))];
        float state[STATE_DIM];
        float act[OUTPUT_DIM];
        int rc = 0;
        build_state_step_only(io_pose, state);
        memset(act, 0, sizeof(act));
        rc = workflow_run_step(runtime, command, state, act);
        if (rc != WORKFLOW_STATUS_OK) {
            return -2;
        }
        io_pose->x = clamp_float(io_pose->x + clamp_float(act[2] * 0.70f, -0.60f, 0.60f), 0.0f, 15.0f);
        io_pose->y = clamp_float(io_pose->y + clamp_float(act[3] * 0.40f, -0.40f, 0.40f), 0.0f, 15.0f);
        render_cli_step_frame(io_pose, frame, command, act);
        wait_enter_for_next_frame();
    }
    printf("step draw done: final pose=(%.3f, %.3f)\n", (double)io_pose->x, (double)io_pose->y);
    return 0;
}

int main(void) {
    const char* vocab_path = "demo/step/data/demo_vocab_step.txt";
    const char* weight_bin_path = "demo/step/data/demo_weights_step.bin";
    const char* weight_c_path = "demo/step/data/demo_weights_step_export.c";
    const char* weight_fn_c_path = "demo/step/data/demo_network_step_functions.c";
    const char* symbol = "g_demo_weights_step";
    const int activations[OUTPUT_DIM] = IO_MAPPING_ACTIVATIONS;
    const float infer_state[STATE_DIM] = { 0.50f, 0.50f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float infer_act[OUTPUT_DIM];
    float* loaded_weights = NULL;
    size_t loaded_count = 0U;
    WorkflowRuntime runtime;
    WorkflowTrainMemoryOptions train_options;
    Pose2D draw_pose = { 7.5f, 7.5f };
    int rc = 0;
    memset(&runtime, 0, sizeof(runtime));
    memset(&train_options, 0, sizeof(train_options));
    memset(infer_act, 0, sizeof(infer_act));

    rc = ensure_step_data_dir();
    if (rc != 0) {
        fprintf(stderr, "ensure_step_data_dir failed: %d\n", rc);
        return 1;
    }

    rc = write_vocab(vocab_path);
    if (rc != 0) {
        fprintf(stderr, "write_vocab failed: %d\n", rc);
        return 2;
    }

    train_options.samples = kStepSamples;
    train_options.sample_count = sizeof(kStepSamples) / sizeof(kStepSamples[0]);
    train_options.vocab_path = vocab_path;
    train_options.out_weights_bin = weight_bin_path;
    train_options.out_weights_c = weight_c_path;
    train_options.out_symbol = symbol;
    train_options.epochs = 20U;
    train_options.learning_rate = 0.08f;
    rc = workflow_train_from_memory(&train_options);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_train_from_memory failed: %d\n", rc);
        return 3;
    }

    rc = weights_load_binary(weight_bin_path, &loaded_weights, &loaded_count);
    if (rc != WEIGHTS_IO_STATUS_OK || loaded_weights == NULL || loaded_count != workflow_weights_count()) {
        fprintf(stderr, "weights_load_binary failed: rc=%d count=%zu\n", rc, loaded_count);
        free(loaded_weights);
        return 4;
    }

    rc = weights_export_c_function_network(weight_fn_c_path,
                                           "g_demo_network_step",
                                           loaded_weights,
                                           VOCAB_SIZE,
                                           MAX_SEQ_LEN,
                                           STATE_DIM,
                                           OUTPUT_DIM,
                                           activations);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        fprintf(stderr, "weights_export_c_function_network failed: %d\n", rc);
        free(loaded_weights);
        return 5;
    }

    rc = workflow_runtime_init(&runtime, vocab_path, weight_bin_path);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_runtime_init failed: %d\n", rc);
        free(loaded_weights);
        return 6;
    }

    rc = workflow_run_step(&runtime, "move left fast", infer_state, infer_act);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_run_step failed: %d\n", rc);
        workflow_runtime_shutdown(&runtime);
        free(loaded_weights);
        return 7;
    }

    rc = run_step_draw_mode(&runtime, &draw_pose, 120U);
    if (rc != 0) {
        fprintf(stderr, "run_step_draw_mode failed: %d\n", rc);
        workflow_runtime_shutdown(&runtime);
        free(loaded_weights);
        return 8;
    }

    workflow_runtime_shutdown(&runtime);
    free(loaded_weights);
    printf("step demo completed\n");
    return 0;
}
