/**
 * @file generate_main.c
 * @brief Edge Video Preprocessing demo — network code generator.
 *
 * CIFAR-10: 32x32x3 RGB, 10 classes, 50K train / 10K test.
 *
 * Architecture — SimpleConvNet (7 CNN + 1 MLP), ~340K params:
 *
 *   Conv1: 3×3,  3→32,   s=1, ReLU6    → 30×30×32   (28800)
 *   Conv2: 3×3,  32→32,  s=1, ReLU6    → 28×28×32   (25088)
 *   Conv3: 3×3,  32→64,  s=2, ReLU6    → 13×13×64   (10816) [downsample]
 *   Conv4: 3×3,  64→64,  s=1, ReLU6    → 11×11×64   (7744)
 *   Conv5: 3×3,  64→128, s=2, ReLU6    →  5×5×128   (3200)  [downsample]
 *   Conv6: 3×3, 128→128, s=1, ReLU6    →  3×3×128   (1152)
 *   GAP:   POOL_AVG, ReLU6 → 128 scalars
 *   MLP:   128 → [128] → 10, ADAM+CE
 *
 * Based on VGG-style proven CIFAR-10 architectures. No Batch Norm — too many
 * dead ReLU neurons when BN centers activations at zero. Batch_size=1 with
 * momentum-driven per-sample updates.
 * All CNN layers: SGD+momentum=0.9, lr=0.0002, wd=1e-4, bias_wd=1e-3, dropout=0.3, bs=1.
 */
#include "profiler.h"
#include "network_def.h"
#include "types/cnn/cnn_config.h"
#include "types/mlp/mlp_config.h"
#include "../demo_runtime_paths.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Dimension constants ── */
#define EVP_INPUT_SIZE          3072U   /* 32 x 32 x 3 */
#define GAP_FEATURES            128U    /* Final conv projection output */
#define MLP_H1                  128U
#define CLASSES                 10U

/* ========================================================================
 *  CNN layer helpers
 * ======================================================================== */

static void cnn_set_config(CnnConfig* c,
    size_t h, size_t w, size_t ch, size_t k, size_t filters,
    size_t feat, CnnPoolingMode pool, CnnActivationType act,
    CnnActivationType output_act,
    CnnConvMode conv, size_t stride, int use_bn, uint32_t seed)
{
    (void)memset(c, 0, sizeof(*c));
    c->total_input_size   = h * w * ch;
    c->sequence_length    = 1U;
    c->frame_width        = w;
    c->frame_height       = h;
    c->channel_count      = ch;
    c->kernel_size        = k;
    c->filter_count       = filters;
    c->feature_size       = feat;
    c->pooling_activation = act;
    c->output_activation  = output_act;
    c->pooling_mode       = pool;
    c->conv_mode          = conv;
    c->stride             = (stride > 0U) ? stride : 1U;
    c->use_batch_norm     = use_bn;
    c->bn_momentum        = 0.9f;
    c->bn_epsilon         = 1e-5f;
    c->seed               = seed;
}

static void cnn_set_train(CnnTrainConfig* c, float lr, float momentum, uint32_t seed) {
    (void)memset(c, 0, sizeof(*c));
    c->learning_rate = lr;
    c->momentum      = momentum;
    c->weight_decay       = 1e-4f;
    c->bias_weight_decay  = 1e-3f;
    c->dropout_rate       = 0.3f;
    c->batch_size         = 1U;
    c->seed               = seed;
}

/**
 * @brief Create a CNN leaf, add to parent, and wire from the previous leaf.
 *
 * Returns the new leaf on success, NULL on failure.
 * If prev_name is NULL, wiring is skipped (used for the first leaf).
 */
static NNSubnetDef* add_and_wire(
    NN_NetworkDef* net, NNSubnetDef* parent,
    const char* name, const char* prev_name,
    size_t input_size, size_t output_size, size_t prev_out_size,
    size_t h, size_t w, size_t ch, size_t k, size_t filters,
    size_t feat_size, CnnPoolingMode pool,
    CnnActivationType act, CnnActivationType output_act,
    CnnConvMode conv, size_t stride, int use_bn,
    uint32_t seed, float lr, float momentum)
{
    NNSubnetDef* leaf;
    CnnConfig infer_cfg;
    CnnTrainConfig train_cfg;
    size_t hs[1];

    hs[0] = (pool == CNN_POOL_NONE) ? output_size : feat_size;

    leaf = nn_subnet_def_create(name, "cnn", input_size, output_size);
    if (!leaf) return NULL;
    if (nn_subnet_def_set_hidden_layers(leaf, 1U, hs) != 0) {
        nn_subnet_def_free(leaf); return NULL;
    }

    cnn_set_config(&infer_cfg, h, w, ch, k, filters, feat_size, pool, act, output_act, conv, stride, use_bn, seed);
    cnn_set_train(&train_cfg, lr, momentum, seed);

    if (nn_subnet_def_set_infer_type_config(leaf, &infer_cfg, sizeof(infer_cfg),
            "types/cnn/cnn_config.h", "CnnConfig") != 0) {
        nn_subnet_def_free(leaf); return NULL;
    }
    if (nn_subnet_def_set_train_type_config(leaf, &train_cfg, sizeof(train_cfg),
            "types/cnn/cnn_config.h", "CnnTrainConfig") != 0) {
        nn_subnet_def_free(leaf); return NULL;
    }

    if (nn_subnet_def_add_subnet(parent, leaf) != 0) {
        nn_subnet_def_free(leaf); return NULL;
    }

    if (prev_name != NULL) {
        size_t i;
        for (i = 0U; i < prev_out_size; ++i) {
            NNConnectionDef* conn = nn_connection_def_create(prev_name, "out", i, name, "in", i);
            if (!conn) { nn_subnet_def_free(leaf); return NULL; }
            if (nn_network_def_add_connection(net, conn) != 0) {
                nn_connection_def_free(conn);
                nn_subnet_def_free(leaf);
                return NULL;
            }
        }
    }
    return leaf;
}

/* ========================================================================
 *  MLP (classification head)
 * ======================================================================== */

static NNSubnetDef* create_mlp_leaf(size_t input_size) {
    NNSubnetDef* s;
    size_t hs[1] = { MLP_H1 };
    MlpConfig* infer_cfg;
    MlpTrainConfig train_cfg;

    s = nn_subnet_def_create("mlp_head", "mlp", input_size, CLASSES);
    if (!s) return NULL;
    if (nn_subnet_def_set_hidden_layers(s, 1U, hs) != 0) { nn_subnet_def_free(s); return NULL; }

    infer_cfg = mlp_config_create(1U);
    if (!infer_cfg) { nn_subnet_def_free(s); return NULL; }
    if (mlp_config_init(infer_cfg, input_size, 1U, hs,
            CLASSES, MLP_ACT_RELU, MLP_ACT_SOFTMAX) != 0) {
        free(infer_cfg); nn_subnet_def_free(s); return NULL;
    }
    (void)memset(&train_cfg, 0, sizeof(train_cfg));
    train_cfg.learning_rate = 0.001f;
    train_cfg.momentum      = 0.9f;
    train_cfg.weight_decay  = 1e-4f;
    train_cfg.optimizer     = MLP_OPT_ADAM;
    train_cfg.loss_func     = MLP_LOSS_CROSS_ENTROPY;
    train_cfg.batch_size    = 1U;
    train_cfg.seed          = 99U;
    if (nn_subnet_def_set_infer_type_config(s, infer_cfg,
            mlp_config_size_for_hidden_layers(infer_cfg->hidden_layer_count),
            "types/mlp/mlp_config.h", "MlpConfig") != 0) {
        free(infer_cfg); nn_subnet_def_free(s); return NULL;
    }
    free(infer_cfg);
    if (nn_subnet_def_set_train_type_config(s, &train_cfg, sizeof(train_cfg),
            "types/mlp/mlp_config.h", "MlpTrainConfig") != 0) { nn_subnet_def_free(s); return NULL; }
    return s;
}

/* ========================================================================
 *  wire helper for MLP connections
 * ======================================================================== */

static int wire_1to1(NN_NetworkDef* net,
    const char* src, const char* tgt, size_t count)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        NNConnectionDef* conn = nn_connection_def_create(src, "out", i, tgt, "in", i);
        if (!conn) return -1;
        if (nn_network_def_add_connection(net, conn) != 0) {
            nn_connection_def_free(conn);
            return -1;
        }
    }
    return 0;
}

/* ========================================================================
 *  Build SimpleConvNet for CIFAR-10 (7 CNN + 1 MLP), ~340K params
 *
 *  Standard VGG-style 3×3 convolutions with stride=2 for downsampling.
 *  All CNN layers: ReLU6, SGD+momentum=0.9, lr=0.0002, wd=1e-4, bias_wd=1e-3, dropout=0.3, bs=1.
 *  GAP: Global average pooling → MLP classifier.
 *
 *  Architecture:
 *    Conv1: 3×3,  3→32,   s=1 → 30×30×32    (28800)
 *    Conv2: 3×3,  32→32,  s=1 → 28×28×32    (25088)
 *    Conv3: 3×3,  32→64,  s=2 → 13×13×64    (10816)
 *    Conv4: 3×3,  64→64,  s=1 → 11×11×64    (7744)
 *    Conv5: 3×3,  64→128, s=2 →  5×5×128    (3200)
 *    Conv6: 3×3, 128→128, s=1 →  3×3×128    (1152)
 *    GAP:   POOL_AVG, BN+ReLU6 → 128 scalars
 *    MLP:   128 → [128] → 10, ADAM+CE, lr=0.001
 *
 *  Total: 8 leaves (7 CNN + 1 MLP), ~340K params.
 * ======================================================================== */

static NN_NetworkDef* create_network(void) {
    NN_NetworkDef* net;
    NNSubnetDef *agent = NULL, *perception = NULL, *head = NULL;
    NNSubnetDef *leaf = NULL;
    const char* prev_name = NULL;
    size_t prev_out;
    uint32_t seed = 41U;

    /* Single-use typedef to shorten the call signature */
    typedef NNSubnetDef* (*awl_t)(NN_NetworkDef*, NNSubnetDef*,
        const char*, const char*, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t, size_t,
        size_t, CnnPoolingMode, CnnActivationType, CnnActivationType,
        CnnConvMode, size_t, int, uint32_t, float, float);
    awl_t A = (awl_t)add_and_wire;

    net = nn_network_def_create("edge_video_preprocess");
    if (!net) return NULL;

    agent      = nn_subnet_def_create("agent",      NULL, 0U, 0U);
    perception = nn_subnet_def_create("perception", NULL, 0U, 0U);
    head       = nn_subnet_def_create("head",       NULL, 0U, 0U);
    if (!agent || !perception || !head) goto fail;

    /* ── Conv1: 3×3, 3→32, s=1, ReLU6 → 30×30×32 ── */
    {
        size_t out = 30U * 30U * 32U;
        leaf = A(net, perception, "conv1", NULL, EVP_INPUT_SIZE, out, 0,
            32U, 32U, 3U, 3U, 32U, 0U, CNN_POOL_NONE, CNN_ACT_RELU6,
            CNN_ACT_RELU6, CNN_CONV_STANDARD, 1U, 0, seed++, 0.0002f, 0.9f);
        if (!leaf) goto fail;
        prev_name = "conv1";
        prev_out  = out;
    }

    /* ── Conv2: 3×3, 32→32, s=1, ReLU6 → 28×28×32 ── */
    {
        size_t out = 28U * 28U * 32U;
        leaf = A(net, perception, "conv2", prev_name, prev_out, out, prev_out,
            30U, 30U, 32U, 3U, 32U, 0U, CNN_POOL_NONE, CNN_ACT_RELU6,
            CNN_ACT_RELU6, CNN_CONV_STANDARD, 1U, 0, seed++, 0.0002f, 0.9f);
        if (!leaf) goto fail;
        prev_name = "conv2";
        prev_out  = out;
    }

    /* ── Conv3: 3×3, 32→64, s=2, ReLU6 → 13×13×64 ── */
    {
        size_t out = 13U * 13U * 64U;
        leaf = A(net, perception, "conv3", prev_name, prev_out, out, prev_out,
            28U, 28U, 32U, 3U, 64U, 0U, CNN_POOL_NONE, CNN_ACT_RELU6,
            CNN_ACT_RELU6, CNN_CONV_STANDARD, 2U, 0, seed++, 0.0002f, 0.9f);
        if (!leaf) goto fail;
        prev_name = "conv3";
        prev_out  = out;
    }

    /* ── Conv4: 3×3, 64→64, s=1, ReLU6 → 11×11×64 ── */
    {
        size_t out = 11U * 11U * 64U;
        leaf = A(net, perception, "conv4", prev_name, prev_out, out, prev_out,
            13U, 13U, 64U, 3U, 64U, 0U, CNN_POOL_NONE, CNN_ACT_RELU6,
            CNN_ACT_RELU6, CNN_CONV_STANDARD, 1U, 0, seed++, 0.0002f, 0.9f);
        if (!leaf) goto fail;
        prev_name = "conv4";
        prev_out  = out;
    }

    /* ── Conv5: 3×3, 64→128, s=2, ReLU6 → 5×5×128 ── */
    {
        size_t out = 5U * 5U * 128U;
        leaf = A(net, perception, "conv5", prev_name, prev_out, out, prev_out,
            11U, 11U, 64U, 3U, 128U, 0U, CNN_POOL_NONE, CNN_ACT_RELU6,
            CNN_ACT_RELU6, CNN_CONV_STANDARD, 2U, 0, seed++, 0.0002f, 0.9f);
        if (!leaf) goto fail;
        prev_name = "conv5";
        prev_out  = out;
    }

    /* ── Conv6: 3×3, 128→128, s=1, ReLU6 → 3×3×128 ── */
    {
        size_t out = 3U * 3U * 128U;
        leaf = A(net, perception, "conv6", prev_name, prev_out, out, prev_out,
            5U, 5U, 128U, 3U, 128U, 0U, CNN_POOL_NONE, CNN_ACT_RELU6,
            CNN_ACT_RELU6, CNN_CONV_STANDARD, 1U, 0, seed++, 0.0002f, 0.9f);
        if (!leaf) goto fail;
        prev_name = "conv6";
        prev_out  = out;
    }

    /* ── GAP: POOL_AVG, ReLU6 → 128 scalars ── */
    {
        size_t feat = GAP_FEATURES;
        leaf = A(net, perception, "gap", prev_name, prev_out, feat, prev_out,
            3U, 3U, 128U, 1U, feat, feat,
            CNN_POOL_AVG, CNN_ACT_RELU6, CNN_ACT_NONE, CNN_CONV_STANDARD, 1U, 0,
            seed++, 0.0002f, 0.9f);
        if (!leaf) goto fail;
        prev_name = "gap";
        prev_out  = feat;
    }

    /* ── MLP: 128 → [128] → 10, ADAM+CE ── */
    {
        NNSubnetDef* mlp_leaf = create_mlp_leaf(prev_out);
        if (!mlp_leaf) goto fail;
        if (nn_subnet_def_add_subnet(head, mlp_leaf) != 0) {
            nn_subnet_def_free(mlp_leaf); goto fail;
        }
        if (wire_1to1(net, prev_name, "mlp_head", prev_out) != 0) goto fail;
    }

    if (nn_subnet_def_add_subnet(agent, perception) != 0) goto fail;
    perception = NULL;
    if (nn_subnet_def_add_subnet(agent, head) != 0) goto fail;
    head = NULL;

    if (nn_network_def_add_subnet(net, agent) != 0) goto fail;
    agent = NULL;

    return net;

fail:
    nn_subnet_def_free(agent); nn_subnet_def_free(perception);
    nn_subnet_def_free(head);
    nn_network_def_free(net);
    return NULL;
}

/* ========================================================================
 *  Main
 * ======================================================================== */

int main(void) {
    NN_NetworkDef* network;
    ProfGenerateRequest request;
    ProfGenerateResult result;
    ProfStatus status;
    char error_buffer[512];
    ProfOutputLayout layout = {
        .tokenizer    = { .c_path = "../data/tokenizer.c",    .h_path = "../data/tokenizer.h"    },
        .network_init = { .c_path = "../data/network_init.c", .h_path = "../data/network_init.h" },
        .weights_load = { .c_path = "../data/weights_load.c", .h_path = "../data/weights_load.h" },
        .train        = { .c_path = "../data/train.c",        .h_path = "../data/train.h"        },
        .weights_save = { .c_path = "../data/weights_save.c", .h_path = "../data/weights_save.h" },
        .infer        = { .c_path = "../data/infer.c",        .h_path = "../data/infer.h"         },
        .metadata_path = "../data/network_metadata.h"
    };

    if (demo_set_working_directory_to_executable() != 0) {
        fprintf(stderr, "Failed to switch working directory to executable directory\n");
        return 1;
    }

    printf("Edge Video Preprocessing — Network Code Generator v20 (SimpleConvNet)\n");
    printf("=========================================================================\n\n");
    printf("Input: 32x32x3 RGB, CIFAR-10 (10 classes)\n\n");
    printf("Architecture: SimpleConvNet (7 CNN + 1 MLP), ~340K params\n");
    printf("  Conv1: 3x3,  3->32,   s=1, ReLU6  -> 30x30x32\n");
    printf("  Conv2: 3x3,  32->32,  s=1, ReLU6  -> 28x28x32\n");
    printf("  Conv3: 3x3,  32->64,  s=2, ReLU6  -> 13x13x64\n");
    printf("  Conv4: 3x3,  64->64,  s=1, ReLU6  -> 11x11x64\n");
    printf("  Conv5: 3x3,  64->128, s=2, ReLU6  ->  5x5x128\n");
    printf("  Conv6: 3x3, 128->128, s=1, ReLU6  ->  3x3x128\n");
    printf("  GAP:   POOL_AVG, %u -> %u, ReLU6\n",
        (unsigned)GAP_FEATURES, (unsigned)GAP_FEATURES);
    printf("  MLP:   %u -> [%u] -> %u, ADAM+CE\n\n",
        (unsigned)GAP_FEATURES, (unsigned)MLP_H1, (unsigned)CLASSES);
    printf("CNN layers: SGD+momentum=0.9, lr=0.0002, wd=1e-4, bias_wd=1e-3, dropout=0.3, ReLU6, bs=1\n");
    printf("Init: He (Kaiming) uniform | SGD per-sample updates (batch_size=1)\n");
    printf("Downsampling: stride=2 in Conv3 and Conv5 | VGG-style design\n\n");

    network = create_network();
    if (!network) {
        fprintf(stderr, "Failed to create network definition\n");
        return 1;
    }

    printf("\nNetwork: %s\n", network->network_name);
    printf("Subnets: %zu, Connections: %zu\n",
        (size_t)network->subnet_count, (size_t)network->connection_count);

    (void)memset(&request, 0, sizeof(request));
    request.network_def     = (const void*)network;
    request.error.buffer    = error_buffer;
    request.error.capacity  = sizeof(error_buffer);
    request.output_layout   = layout;

    printf("\nGenerating code to ../data/ ...\n");
    status = profiler_generate_v2(&request, &result);
    if (status != PROF_STATUS_OK) {
        fprintf(stderr, "Code generation failed: %s\n", error_buffer);
        nn_network_def_free(network);
        return 1;
    }

    printf("\nCode generation successful!\n");
    printf("Network Hash: 0x%016llx\n", (unsigned long long)result.network_hash);
    printf("Metadata: %s\n", result.metadata_written_path);

    nn_network_def_free(network);
    return 0;
}
