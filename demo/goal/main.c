#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>
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

enum { MAX_GOAL_SAMPLES = 256 };
enum {
    GOAL_MIN_STEPS = 30,
    GOAL_MAX_STEPS = 120,
    GOAL_EPISODES = 5
};

/**
 * @brief 创建目录（若已存在则视为成功）。
 */
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

/**
 * @brief 确保 goal 演示数据目录层级存在。
 */
static int ensure_goal_data_dir(void) {
    if (create_dir_if_missing("demo") != 0) {
        return -1;
    }
    if (create_dir_if_missing("demo/goal") != 0) {
        return -2;
    }
    if (create_dir_if_missing("demo/goal/data") != 0) {
        return -3;
    }
    return 0;
}

/**
 * @brief 将字符串数组按行写入文本文件。
 */
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

/**
 * @brief 写出 goal 演示词表。
 */
static int write_vocab(const char* file_path) {
    static const char* tokens[] = {
        "<unk>", "goal",
        "0", "1", "2", "3", "4", "5", "6", "7",
        "8", "9", "10", "11", "12", "13", "14", "15"
    };
    return write_rows_to_text(file_path, tokens, sizeof(tokens) / sizeof(tokens[0]));
}

/**
 * @brief 按剩余距离裁剪步长，避免越过目标点。
 */
static float clamp_abs_step(float step, float remain) {
    if (fabsf(step) > fabsf(remain)) {
        return remain;
    }
    return step;
}

/**
 * @brief 浮点数区间裁剪。
 */
static float clamp_float(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

/**
 * @brief 整数区间裁剪。
 */
static int clamp_int(int v, int lo, int hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

/**
 * @brief 清屏并将光标移动到左上角。
 */
static void clear_screen_draw_mode(void) {
    printf("\x1b[2J\x1b[H");
}

/**
 * @brief 等待用户回车，用于逐帧观察。
 */
static void wait_enter_for_next_frame(void) {
    char linebuf[16];
    printf("Press Enter to compute next frame...");
    fflush(stdout);
    if (fgets(linebuf, sizeof(linebuf), stdin) == NULL) {
        clearerr(stdin);
    }
}

/**
 * @brief 生成 goal 命令文本（如 "goal 15 4"）。
 */
static void format_goal_command(int gx, int gy, char* out_cmd, size_t cap) {
    if (out_cmd == NULL || cap == 0U) {
        return;
    }
    (void)snprintf(out_cmd, cap, "goal %d %d", gx, gy);
}

/**
 * @brief 随机采样目标点与回合最大步数。
 */
static void pick_random_goal(Pose2D* out_goal, int* out_max_steps) {
    if (out_goal == NULL || out_max_steps == NULL) {
        return;
    }
    out_goal->x = (float)(rand() % 16);
    out_goal->y = (float)(rand() % 16);
    *out_max_steps = GOAL_MIN_STEPS + (rand() % (GOAL_MAX_STEPS - GOAL_MIN_STEPS + 1));
}

/**
 * @brief 构建包含位姿与目标信息的状态向量。
 */
static void build_state_with_goal(const Pose2D* pose, const Pose2D* goal, float* state) {
    float remain_x = goal->x - pose->x;
    float remain_y = goal->y - pose->y;
    memset(state, 0, sizeof(float) * STATE_DIM);
    state[0] = pose->x / 15.0f;
    state[1] = pose->y / 15.0f;
    state[2] = remain_x / 15.0f;
    state[3] = remain_y / 15.0f;
    state[4] = (fabsf(remain_x) + fabsf(remain_y)) / 30.0f;
}

/**
 * @brief 渲染 goal 任务单帧终端画面。
 */
static void render_cli_goal_frame(const Pose2D* pose,
                                  const Pose2D* goal,
                                  size_t frame,
                                  size_t remaining_steps,
                                  const char* goal_command,
                                  const float* act,
                                  float remain_x,
                                  float remain_y) {
    char canvas[32][96];
    int width = 45;
    int height = 18;
    int x = 0;
    int y = 0;
    int pose_col = 0;
    int pose_row = 0;
    int goal_col = 0;
    int goal_row = 0;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            canvas[y][x] = '.';
        }
        canvas[y][width] = '\0';
    }
    pose_col = clamp_int((int)lroundf((pose->x / 15.0f) * (float)(width - 1)), 0, width - 1);
    pose_row = clamp_int((int)lroundf((pose->y / 15.0f) * (float)(height - 1)), 0, height - 1);
    goal_col = clamp_int((int)lroundf((goal->x / 15.0f) * (float)(width - 1)), 0, width - 1);
    goal_row = clamp_int((int)lroundf((goal->y / 15.0f) * (float)(height - 1)), 0, height - 1);
    canvas[height - 1 - goal_row][goal_col] = 'G';
    canvas[height - 1 - pose_row][pose_col] = '@';
    printf("\x1b[H");
    printf("CLI Goal Mode | frame=%03zu | steps_left=%zu\n", frame + 1U, remaining_steps);
    printf("command=\"%s\"\n", goal_command);
    printf("pose=(%.2f, %.2f) goal=(%.2f, %.2f) remain=(%.2f, %.2f) act=(%.3f, %.3f, %.3f, %.3f)\n",
           (double)pose->x, (double)pose->y, (double)goal->x, (double)goal->y,
           (double)remain_x, (double)remain_y,
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

/**
 * @brief 生成 goal 任务训练样本。
 *
 * 关键算法：
 * - 目标方向由 rx/ry 符号决定；
 * - 连续动作幅值由归一化距离裁剪到 [-1,1]。
 */
static int build_goal_training_samples(WorkflowTrainSample* out_samples,
                                       char commands[][32],
                                       float states[][STATE_DIM],
                                       float targets[][OUTPUT_DIM],
                                       size_t capacity,
                                       size_t* out_count) {
    static const int goals[][2] = {
        { 15, 4 },
        { 12, 12 },
        { 3, 14 },
        { 2, 2 }
    };
    static const float starts[][2] = {
        { 0.0f, 0.0f },
        { 2.0f, 1.0f },
        { 4.0f, 3.0f },
        { 7.0f, 2.0f },
        { 10.0f, 5.0f },
        { 12.0f, 8.0f },
        { 6.0f, 12.0f },
        { 1.0f, 14.0f }
    };
    size_t idx = 0U;
    size_t g = 0U;
    size_t s = 0U;
    if (out_samples == NULL || commands == NULL || states == NULL || targets == NULL || out_count == NULL) {
        return -1;
    }
    for (g = 0U; g < sizeof(goals) / sizeof(goals[0]); ++g) {
        for (s = 0U; s < sizeof(starts) / sizeof(starts[0]); ++s) {
            float px = starts[s][0];
            float py = starts[s][1];
            float rx = (float)goals[g][0] - px;
            float ry = (float)goals[g][1] - py;
            if (idx >= capacity) {
                return -2;
            }
            format_goal_command(goals[g][0], goals[g][1], commands[idx], 32U);
            memset(states[idx], 0, sizeof(float) * STATE_DIM);
            memset(targets[idx], 0, sizeof(float) * OUTPUT_DIM);
            states[idx][0] = px / 15.0f;
            states[idx][1] = py / 15.0f;
            states[idx][2] = rx / 15.0f;
            states[idx][3] = ry / 15.0f;
            states[idx][4] = (fabsf(rx) + fabsf(ry)) / 30.0f;
            if (fabsf(rx) >= 0.01f || fabsf(ry) >= 0.01f) {
                if (rx < 0.0f) {
                    targets[idx][0] = 1.0f;
                    targets[idx][1] = 0.0f;
                } else {
                    targets[idx][0] = 0.0f;
                    targets[idx][1] = 1.0f;
                }
                targets[idx][2] = clamp_float(rx / 15.0f, -1.0f, 1.0f);
                targets[idx][3] = clamp_float(ry / 15.0f, -1.0f, 1.0f);
            }
            out_samples[idx].command = commands[idx];
            out_samples[idx].state = states[idx];
            out_samples[idx].target = targets[idx];
            idx++;
        }
    }
    *out_count = idx;
    return 0;
}

/**
 * @brief 执行单回合目标跟随循环。
 *
 * 关键保护点：
 * - 每帧先检测是否到达，避免额外一步造成超调。
 * - 对输出步长设置最小推进阈值，减轻模型输出过小导致停滞的问题。
 */
static int run_goal_loop(WorkflowRuntime* runtime, const Pose2D* goal, const char* goal_command, Pose2D* io_pose, size_t max_frames) {
    size_t frame = 0U;
    if (runtime == NULL || goal == NULL || goal_command == NULL || io_pose == NULL || max_frames == 0U) {
        return -1;
    }
    clear_screen_draw_mode();
    for (frame = 0U; frame < max_frames; ++frame) {
        float state[STATE_DIM];
        float act[OUTPUT_DIM];
        float remain_x = goal->x - io_pose->x;
        float remain_y = goal->y - io_pose->y;
        float step_x = 0.0f;
        float step_y = 0.0f;
        int rc = 0;
        size_t remaining_steps = max_frames - frame;
        memset(act, 0, sizeof(act));
        if (fabsf(remain_x) < 0.45f && fabsf(remain_y) < 0.45f) {
            render_cli_goal_frame(io_pose, goal, frame, remaining_steps, goal_command, act, remain_x, remain_y);
            printf("goal reached at frame=%zu pose=(%.3f, %.3f)\n",
                   frame, (double)io_pose->x, (double)io_pose->y);
            return 0;
        }
        build_state_with_goal(io_pose, goal, state);
        rc = workflow_run_step(runtime, goal_command, state, act);
        if (rc != WORKFLOW_STATUS_OK) {
            return -2;
        }
        step_x = act[2] * 0.55f;
        if (fabsf(step_x) < 0.08f && fabsf(remain_x) > 0.5f) {
            step_x = (remain_x > 0.0f) ? 0.10f : -0.10f;
        }
        step_x = clamp_abs_step(step_x, remain_x);
        if (remain_y > 0.0f) {
            step_y = ((act[3] + 1.0f) * 0.5f) * 0.40f;
            if (step_y < 0.05f) {
                step_y = 0.05f;
            }
            step_y = clamp_abs_step(step_y, remain_y);
        } else if (remain_y < 0.0f) {
            step_y = clamp_abs_step(-0.05f, remain_y);
        }
        io_pose->x += step_x;
        io_pose->y += step_y;
        render_cli_goal_frame(io_pose, goal, frame, remaining_steps, goal_command, act, remain_x, remain_y);
        wait_enter_for_next_frame();
    }
    printf("goal timeout after max_frames=%zu pose=(%.3f, %.3f)\n",
           max_frames, (double)io_pose->x, (double)io_pose->y);
    return 1;
}

/**
 * @brief Goal 演示主流程：训练 -> 导出 -> 推理 -> 多回合交互可视化。
 */
int main(void) {
    const char* vocab_path = "demo/goal/data/demo_vocab_goal.txt";
    const char* weight_bin_path = "demo/goal/data/demo_weights_goal.bin";
    const char* weight_c_path = "demo/goal/data/demo_weights_goal_export.c";
    const char* weight_fn_c_path = "demo/goal/data/demo_network_goal_functions.c";
    const char* symbol = "g_demo_weights_goal";
    const int activations[OUTPUT_DIM] = IO_MAPPING_ACTIVATIONS;
    const float infer_state[STATE_DIM] = { 0.95f, 0.12f, 0.35f, 0.0f, 0.05f, 0.20f, 0.0f, 0.0f };
    float infer_act[OUTPUT_DIM];
    float* loaded_weights = NULL;
    size_t loaded_count = 0U;
    WorkflowRuntime runtime;
    WorkflowTrainMemoryOptions train_options;
    WorkflowTrainSample train_samples[MAX_GOAL_SAMPLES];
    char train_commands[MAX_GOAL_SAMPLES][32];
    float train_states[MAX_GOAL_SAMPLES][STATE_DIM];
    float train_targets[MAX_GOAL_SAMPLES][OUTPUT_DIM];
    char goal_command[32];
    size_t train_count = 0U;
    Pose2D start_pose = { 0.0f, 0.0f };
    Pose2D target_pose;
    int max_steps = 0;
    size_t episode = 0U;
    int rc = 0;
    memset(&runtime, 0, sizeof(runtime));
    memset(&train_options, 0, sizeof(train_options));
    memset(infer_act, 0, sizeof(infer_act));

    rc = ensure_goal_data_dir();
    if (rc != 0) {
        fprintf(stderr, "ensure_goal_data_dir failed: %d\n", rc);
        return 1;
    }

    rc = write_vocab(vocab_path);
    if (rc != 0) {
        fprintf(stderr, "write_vocab failed: %d\n", rc);
        return 2;
    }

    rc = build_goal_training_samples(train_samples,
                                     train_commands,
                                     train_states,
                                     train_targets,
                                     MAX_GOAL_SAMPLES,
                                     &train_count);
    if (rc != 0 || train_count == 0U) {
        fprintf(stderr, "build_goal_training_samples failed: %d\n", rc);
        return 3;
    }
    target_pose.x = 15.0f;
    target_pose.y = 4.0f;
    format_goal_command((int)lroundf(target_pose.x), (int)lroundf(target_pose.y), goal_command, sizeof(goal_command));
    train_options.samples = train_samples;
    train_options.sample_count = train_count;
    train_options.vocab_path = vocab_path;
    train_options.out_weights_bin = weight_bin_path;
    train_options.out_weights_c = weight_c_path;
    train_options.out_symbol = symbol;
    train_options.epochs = 60U;
    train_options.learning_rate = 0.08f;
    rc = workflow_train_from_memory(&train_options);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_train_from_memory failed: %d\n", rc);
        return 4;
    }

    rc = weights_load_binary(weight_bin_path, &loaded_weights, &loaded_count);
    if (rc != WEIGHTS_IO_STATUS_OK || loaded_weights == NULL || loaded_count != workflow_weights_count()) {
        fprintf(stderr, "weights_load_binary failed: rc=%d count=%zu\n", rc, loaded_count);
        free(loaded_weights);
        return 5;
    }

    rc = weights_export_c_function_network(weight_fn_c_path,
                                           "g_demo_network_goal",
                                           loaded_weights,
                                           VOCAB_SIZE,
                                           MAX_SEQ_LEN,
                                           STATE_DIM,
                                           OUTPUT_DIM,
                                           activations);
    if (rc != WEIGHTS_IO_STATUS_OK) {
        fprintf(stderr, "weights_export_c_function_network failed: %d\n", rc);
        free(loaded_weights);
        return 6;
    }

    rc = workflow_runtime_init(&runtime, vocab_path, weight_bin_path);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_runtime_init failed: %d\n", rc);
        free(loaded_weights);
        return 7;
    }

    rc = workflow_run_step(&runtime, goal_command, infer_state, infer_act);
    if (rc != WORKFLOW_STATUS_OK) {
        fprintf(stderr, "workflow_run_step failed: %d\n", rc);
        workflow_runtime_shutdown(&runtime);
        free(loaded_weights);
        return 8;
    }

    srand((unsigned int)time(NULL));
    for (episode = 0U; episode < GOAL_EPISODES; ++episode) {
        pick_random_goal(&target_pose, &max_steps);
        format_goal_command((int)lroundf(target_pose.x), (int)lroundf(target_pose.y), goal_command, sizeof(goal_command));
        printf("episode=%zu target=(%.0f,%.0f) max_steps=%d\n",
               episode + 1U, (double)target_pose.x, (double)target_pose.y, max_steps);
        rc = run_goal_loop(&runtime, &target_pose, goal_command, &start_pose, (size_t)max_steps);
        if (rc < 0) {
            fprintf(stderr, "run_goal_loop failed: %d\n", rc);
            workflow_runtime_shutdown(&runtime);
            free(loaded_weights);
            return 9;
        }
    }

    workflow_runtime_shutdown(&runtime);
    free(loaded_weights);
    printf("goal demo completed\n");
    return 0;
}
