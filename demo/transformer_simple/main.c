#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "../../src/include/config_user.h"
#include "../../src/include/network_spec.h"
#include "../../src/include/weights_io.h"
#include "../../src/infer/include/workflow_infer.h"
#include "../../src/train/include/workflow_train.h"

/*
 * 本文件是一个“可读性优先”的 Transformer 示例：
 * 1) 在代码内构建网络图拓扑；
 * 2) 构造最小训练样本；
 * 3) 训练并导出权重；
 * 4) 再加载权重执行推理；
 * 5) 按“每个输出节点对应一个控制通道”打印验证结果。
 *
 * 这个示例不是做文本分类，也不是只输出一个类别。
 * 输出向量每个维度都代表一个通道：
 * - THROTTLE：油门开关
 * - BRAKE：刹车开关
 * - TURN：转向单轴（-1 左 / 0 中 / +1 右）
 * - AUX：预留通道
 */

/* 创建目录，若目录已存在则视为成功。 */
static int ensure_dir(const char* path) {
    int rc = 0;
    /* 输入路径为空时直接失败。 */
    if (path == NULL || path[0] == '\0') {
        return -1;
    }
#if defined(_WIN32)
    /* Windows 使用 _mkdir。 */
    rc = _mkdir(path);
#else
    /* POSIX 使用 mkdir 并给默认权限。 */
    rc = mkdir(path, 0755);
#endif
    /* 新建成功或“已存在”都算成功。 */
    if (rc == 0 || errno == EEXIST) {
        return 0;
    }
    /* 其他错误返回失败。 */
    return -1;
}

/* 拼接目录和文件名，生成目标路径。 */
static void build_path(char* out_path, size_t cap, const char* dir, const char* file_name) {
    (void)snprintf(out_path, cap, "%s/%s", dir, file_name);
}

/*
 * 写入词表。
 * 这里的 token 只是命令标签，不表示英文 NLP 任务。
 */
static int write_vocab(const char* file_path) {
    static const char* tokens[] = {
        "go_left",
        "go_right",
        "stop",
        "idle"
    };
    FILE* fp = NULL;
    size_t i = 0U;
    /* 输出路径必须有效。 */
    if (file_path == NULL) {
        return -1;
    }
    /* 以写模式创建词表文本。 */
    fp = fopen(file_path, "w");
    if (fp == NULL) {
        return -2;
    }
    /* 每个 token 一行。 */
    for (i = 0U; i < sizeof(tokens) / sizeof(tokens[0]); ++i) {
        if (fprintf(fp, "%s\n", tokens[i]) < 0) {
            fclose(fp);
            return -3;
        }
    }
    fclose(fp);
    return 0;
}

/*
 * 构建示例网络规格：
 * INPUT -> LINEAR -> TRANSFORMER_BLOCK -> TRANSFORMER_BLOCK
 *
 * 注意：这是“拓扑图配置”流程，不依赖旧接口。
 */
static int build_transformer_spec(NetworkSpec* spec) {
    /* 节点定义：id、节点类型、selector_offset、selector_size。 */
    NetworkGraphNode nodes[] = {
        {0, NETWORK_NODE_INPUT, 0, (MAX_SEQ_LEN + STATE_DIM)},
        {1, NETWORK_NODE_LINEAR, 0, OUTPUT_DIM},
        {2, NETWORK_NODE_TRANSFORMER_BLOCK, 0, OUTPUT_DIM},
        {3, NETWORK_NODE_TRANSFORMER_BLOCK, 0, OUTPUT_DIM}
    };
    /* 边定义：from -> to。 */
    NetworkGraphEdge edges[] = {
        {0, 1},
        {1, 2},
        {2, 3}
    };
    NetworkGraph graph;
    int rc = 0;
    memset(&graph, 0, sizeof(graph));
    /* 先构建并校验图。 */
    rc = network_graph_build(&graph,
                             nodes,
                             sizeof(nodes) / sizeof(nodes[0]),
                             edges,
                             sizeof(edges) / sizeof(edges[0]),
                             0,
                             3);
    if (rc != 0) {
        return rc;
    }
    /* 再从图推导执行用的 NetworkSpec。 */
    return network_spec_build_from_graph(spec, &graph);
}

/* 写入单条目标向量：最多写前 4 维，其他维清零。 */
static void fill_target(float* target, float a, float b, float c, float d) {
    size_t i = 0U;
    /* 先清零，避免残留旧值影响训练。 */
    for (i = 0U; i < OUTPUT_DIM; ++i) {
        target[i] = 0.0f;
    }
    /* 按通道顺序写入。 */
    if (OUTPUT_DIM > 0U) {
        target[0] = a;
    }
    if (OUTPUT_DIM > 1U) {
        target[1] = b;
    }
    if (OUTPUT_DIM > 2U) {
        target[2] = c;
    }
    if (OUTPUT_DIM > 3U) {
        target[3] = d;
    }
}

/* 输出通道的人类可读名称。 */
static const char* channel_name(size_t idx) {
    static const char* names[] = { "THROTTLE", "BRAKE", "TURN", "AUX" };
    if (idx < (sizeof(names) / sizeof(names[0]))) {
        return names[idx];
    }
    return "unknown";
}

/*
 * TURN 连续值离散化：
 * <= -0.33 判为左转（-1）
 * >= +0.33 判为右转（+1）
 * 中间区域判为居中（0）
 */
static int turn_class(float v) {
    if (v <= -0.33f) {
        return -1;
    }
    if (v >= 0.33f) {
        return 1;
    }
    return 0;
}

int main(void) {
    /* 输出文件目录。 */
    const char* out_dir = "demo/transformer_simple/data";
    /* 训练/推理依赖文件路径。 */
    char vocab_path[260];
    char bin_path[260];
    char c_path[260];
    /* 训练与推理上下文。 */
    WorkflowTrainMemoryOptions train_options;
    WorkflowRuntime runtime;
    /* 8 条样本（每个场景 2 条）。 */
    WorkflowTrainSample samples[8];
    char commands[8][16];
    float states[8][STATE_DIM];
    float targets[8][OUTPUT_DIM];
    /* 推理输出向量。 */
    float action[OUTPUT_DIM];
    /* 网络规格（由图拓扑生成）。 */
    NetworkSpec spec;
    /* 权重完整性校验所需缓冲。 */
    float* loaded = NULL;
    size_t loaded_count = 0U;
    size_t i = 0U;
    int rc = 0;

    /* 所有结构先清零，防止未初始化字段。 */
    memset(&train_options, 0, sizeof(train_options));
    memset(&runtime, 0, sizeof(runtime));
    memset(&spec, 0, sizeof(spec));
    memset(action, 0, sizeof(action));

    /* 准备输出目录。 */
    if (ensure_dir("demo") != 0 || ensure_dir("demo/transformer_simple") != 0 || ensure_dir(out_dir) != 0) {
        return 1;
    }
    /* 生成各文件路径。 */
    build_path(vocab_path, sizeof(vocab_path), out_dir, "vocab.txt");
    build_path(bin_path, sizeof(bin_path), out_dir, "weights.bin");
    build_path(c_path, sizeof(c_path), out_dir, "weights_export.c");

    /* 写词表。 */
    rc = write_vocab(vocab_path);
    if (rc != 0) {
        return 2;
    }

    /* 基于拓扑图构建 spec。 */
    rc = build_transformer_spec(&spec);
    if (rc != 0) {
        return 3;
    }

    /*
     * 构造 4 类控制命令（每类 2 条）：
     * - go_left
     * - go_right
     * - stop
     * - idle
     */
    (void)snprintf(commands[0], sizeof(commands[0]), "%s", "go_left");
    (void)snprintf(commands[1], sizeof(commands[1]), "%s", "go_left");
    (void)snprintf(commands[2], sizeof(commands[2]), "%s", "go_right");
    (void)snprintf(commands[3], sizeof(commands[3]), "%s", "go_right");
    (void)snprintf(commands[4], sizeof(commands[4]), "%s", "stop");
    (void)snprintf(commands[5], sizeof(commands[5]), "%s", "stop");
    (void)snprintf(commands[6], sizeof(commands[6]), "%s", "idle");
    (void)snprintf(commands[7], sizeof(commands[7]), "%s", "idle");

    /* 先把全部状态维度清零。 */
    for (i = 0U; i < 8U; ++i) {
        size_t j = 0U;
        for (j = 0U; j < STATE_DIM; ++j) {
            states[i][j] = 0.0f;
        }
    }

    /*
     * 填充状态特征。
     * 这里不是物理精确建模，只是最小可分离的示例数据。
     */
    states[0][0] = 0.9f;
    states[0][1] = 0.8f;
    states[1][0] = 0.7f;
    states[1][1] = 0.6f;
    states[2][0] = 0.9f;
    states[2][2] = 0.8f;
    states[3][0] = 0.7f;
    states[3][2] = 0.6f;
    states[4][3] = 1.0f;
    states[5][3] = 0.9f;
    states[6][0] = 0.1f;
    states[7][0] = 0.0f;

    /*
     * 填充目标向量，通道语义：
     * [THROTTLE, BRAKE, TURN, AUX]
     *
     * go_left  => [1,0,-1,0]
     * go_right => [1,0,+1,0]
     * stop     => [0,1, 0,0]
     * idle     => [0,0, 0,0]
     */
    fill_target(targets[0], 1.0f, 0.0f, -1.0f, 0.0f);
    fill_target(targets[1], 1.0f, 0.0f, -1.0f, 0.0f);
    fill_target(targets[2], 1.0f, 0.0f, 1.0f, 0.0f);
    fill_target(targets[3], 1.0f, 0.0f, 1.0f, 0.0f);
    fill_target(targets[4], 0.0f, 1.0f, 0.0f, 0.0f);
    fill_target(targets[5], 0.0f, 1.0f, 0.0f, 0.0f);
    fill_target(targets[6], 0.0f, 0.0f, 0.0f, 0.0f);
    fill_target(targets[7], 0.0f, 0.0f, 0.0f, 0.0f);

    /* 组装 WorkflowTrainSample 指针。 */
    for (i = 0U; i < 8U; ++i) {
        samples[i].command = commands[i];
        samples[i].state = states[i];
        samples[i].target = targets[i];
    }

    /* 配置训练参数。 */
    train_options.samples = samples;
    train_options.sample_count = 8U;
    train_options.vocab_path = vocab_path;
    train_options.out_weights_bin = bin_path;
    train_options.out_weights_c = c_path;
    train_options.out_symbol = "g_transformer_simple_weights";
    train_options.network_spec = &spec;
    train_options.epochs = 80U;
    train_options.learning_rate = 0.06f;

    /* 打印 demo 元信息，帮助首次用户理解意图。 */
    printf("=== Transformer Multi-Output Demo ===\n");
    printf("Scenario: command+state -> control vector [THROTTLE, BRAKE, TURN, AUX]\n");
    printf("Network: INPUT -> LINEAR -> TRANSFORMER_BLOCK -> TRANSFORMER_BLOCK\n");
    printf("Goal: one node per channel; TURN is a signed axis {-1:left, 0:center, 1:right}\n");
    printf("Channels: [THROTTLE BRAKE TURN AUX]\n");
    printf("-----------------------------------------------\n");
    printf("Training log note: epoch/avg_loss is mean error per epoch; lower is better\n");
    printf("training start: epochs=%u, sample_count=%u\n", (unsigned)train_options.epochs, (unsigned)train_options.sample_count);

    /* 执行训练。 */
    rc = workflow_train_from_memory(&train_options);
    if (rc != WORKFLOW_STATUS_OK) {
        return 4;
    }

    /* 读取并校验权重数量，确保训练输出与 spec 一致。 */
    rc = weights_load_binary(bin_path, &loaded, &loaded_count);
    if (rc != WEIGHTS_IO_STATUS_OK || loaded == NULL || loaded_count != workflow_weights_count(&spec)) {
        free(loaded);
        return 5;
    }
    /* 校验后释放临时权重缓冲。 */
    free(loaded);
    loaded = NULL;

    /* 初始化推理运行时。 */
    rc = workflow_runtime_init(&runtime, vocab_path, bin_path, &spec);
    if (rc != WORKFLOW_STATUS_OK) {
        return 6;
    }

    /* 开始推理验证。 */
    printf("-----------------------------------------------\n");
    printf("Inference validation: run 1 sample for each control scenario\n");
    printf("Columns: channel-level expected/predicted, no argmax selection\n");
    {
        const char* eval_names[] = { "go_left", "go_right", "stop", "idle" };
        const char* eval_cmds[] = { "go_left", "go_right", "stop", "idle" };
        const float* eval_states[] = { states[0], states[2], states[4], states[6] };
        const float* eval_targets[] = { targets[0], targets[2], targets[4], targets[6] };
        size_t scenario_ok_count = 0U;
        size_t channel_ok_count = 0U;
        size_t total_channels = 0U;
        size_t k = 0U;

        /* 每个场景跑一条验证样本。 */
        for (k = 0U; k < 4U; ++k) {
            size_t j = 0U;
            int scenario_ok = 1;

            /* 运行一步推理。 */
            rc = workflow_run_step(&runtime, eval_cmds[k], eval_states[k], action);
            if (rc != WORKFLOW_STATUS_OK) {
                workflow_runtime_shutdown(&runtime);
                return 7;
            }

            /* 打印当前场景名称。 */
            printf("scenario[%u] %s\n", (unsigned)k, eval_names[k]);

            /* 逐输出节点比较 expected / predicted。 */
            for (j = 0U; j < OUTPUT_DIM; ++j) {
                /* TURN 是三值语义（-1/0/+1），使用专门判定。 */
                if (j == 2U) {
                    int expected_turn = turn_class(eval_targets[k][j]);
                    int predicted_turn = turn_class(action[j]);
                    if (expected_turn == predicted_turn) {
                        channel_ok_count += 1U;
                    } else {
                        scenario_ok = 0;
                    }
                    total_channels += 1U;
                    printf("  %-11s expected=%d predicted=%d raw=%.3f\n",
                           channel_name(j),
                           expected_turn,
                           predicted_turn,
                           action[j]);
                } else {
                    /* 其余通道按开关语义判定（阈值 0.5）。 */
                    int expected_on = (eval_targets[k][j] >= 0.5f) ? 1 : 0;
                    int predicted_on = (action[j] >= 0.5f) ? 1 : 0;
                    if (expected_on == predicted_on) {
                        channel_ok_count += 1U;
                    } else {
                        scenario_ok = 0;
                    }
                    total_channels += 1U;
                    printf("  %-11s expected=%d predicted=%d raw=%.3f\n",
                           channel_name(j),
                           expected_on,
                           predicted_on,
                           action[j]);
                }
            }
            /* 当前场景所有通道都正确才算场景通过。 */
            if (scenario_ok) {
                scenario_ok_count += 1U;
            }
        }

        /* 汇总两类指标：按场景、按通道。 */
        printf("summary: scenario_pass=%u/%u, channel_pass=%u/%u\n",
               (unsigned)scenario_ok_count,
               4U,
               (unsigned)channel_ok_count,
               (unsigned)total_channels);
    }

    /* 结论与结束。 */
    printf("-----------------------------------------------\n");
    printf("Demo conclusion: transformer is used as feature mixer, then each output node drives one actuator channel directly\n");
    printf("transformer_simple_demo ok\n");

    /* 释放推理资源。 */
    workflow_runtime_shutdown(&runtime);
    return 0;
}
