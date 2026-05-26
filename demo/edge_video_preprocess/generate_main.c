/**
 * @file generate_main.c
 * @brief Edge Video Preprocessing demo — network code generator.
 *
 * CIFAR-10: 32x32x3 RGB, 10 classes, 50K train / 10K test.
 *
 * Architecture — BnConvNet v47 (batch_size=4, lr=0.001, mom=0.0), 4 CNN + BN + LeakyReLU + MLP(LeakyReLU):
 *
 *   Conv1: 3x3,   3->32,   s=1, BN, LeakyReLU  -> 30x30x32
 *   Conv2: 3x3,  32->64,   s=1, BN, LeakyReLU  -> 28x28x64
 *   Conv3: 3x3,  64->128,  s=1, BN, LeakyReLU  -> 26x26x128
 *   Conv4: 3x3, 128->256,  s=1, BN, LeakyReLU  -> 24x24x256
 *   GAP:   POOL_AVG over 24x24 -> 256 scalars (no activation)
 *   MLP:   256 -> [256(LeakyReLU)] -> 10, ADAM+CE
 *
 * Key design decisions:
 *   - bn_spatial_var fix applied (v34 verification: stable 43K+ steps)
 *   - LeakyReLU (alpha=0.01) on ALL layers (CNN + MLP) to prevent dead neuron
 *   - All stride=1 — proven better than stride-2 for this framework
 *   - **batch_size=4 + lr=0.001 + mom=0.0** — lr lowered to 0.001 to compensate
 *     for 4x summed gradient; mom=0 to avoid momentum-induced NaN as seen in v45
 *   - **v47 fix**: BN backward x_hat bug (line 453 uses running stats, not gamma/beta);
 *     BN running statistics now updated for POOL_AVG/POOL_MAX/POOL_DUAL paths
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
#define GAP_FEATURES            256U
#define MLP_H1                  256U
#define CLASSES                 10U

/* ── CNN learning rate ── */
#define CNN_LR                  0.001f  /* v47: 0.005/4 to compensate for 4x sum-gradients */
#define CNN_MOMENTUM            0.0f    /* v47: mom=0 to avoid momentum-induced NaN */
#define CNN_BATCH_SIZE          4U      /* v47: batch_size>1 to reduce gradient noise */

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

static void cnn_set_train(CnnTrainConfig* c, float lr, float momentum, uint32_t seed, int debug_layer_idx) {
    (void)memset(c, 0, sizeof(*c));
    c->learning_rate      = lr;
    c->momentum           = momentum;
    c->weight_decay       = 1e-4f;
    c->bias_weight_decay  = 1e-3f;
    c->dropout_rate       = 0.0f;
    c->batch_size         = CNN_BATCH_SIZE;
    c->seed               = seed;
    c->debug_level        = 3;
    c->debug_layer_index  = debug_layer_idx;
}

static NNSubnetDef* add_and_wire(
    NN_NetworkDef* net, NNSubnetDef* parent,
    const char* name, const char* prev_name,
    size_t input_size, size_t output_size, size_t prev_out_size,
    size_t h, size_t w, size_t ch, size_t k, size_t filters,
    size_t feat_size, CnnPoolingMode pool,
    CnnActivationType act, CnnActivationType output_act,
    CnnConvMode conv, size_t stride, int use_bn,
    uint32_t seed, float lr, float momentum, int debug_layer_idx)
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

    cnn_set_config(&infer_cfg, h, w, ch, k, filters, feat_size, pool,
                   act, output_act, conv, stride, use_bn, seed);
    cnn_set_train(&train_cfg, lr, momentum, seed, debug_layer_idx);

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
            NNConnectionDef* conn = nn_connection_def_create(
                prev_name, "out", i, name, "in", i);
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
    if (nn_subnet_def_set_hidden_layers(s, 1U, hs) != 0) {
        nn_subnet_def_free(s); return NULL;
    }

    infer_cfg = mlp_config_create(1U);
    if (!infer_cfg) { nn_subnet_def_free(s); return NULL; }
    if (mlp_config_init(infer_cfg, input_size, 1U, hs,
            CLASSES, MLP_ACT_LEAKY_RELU, MLP_ACT_SOFTMAX) != 0) {
        free(infer_cfg); nn_subnet_def_free(s); return NULL;
    }
    (void)memset(&train_cfg, 0, sizeof(train_cfg));
    train_cfg.learning_rate = 0.001f;
    train_cfg.momentum      = 0.9f;
    train_cfg.weight_decay  = 1e-4f;
    train_cfg.optimizer     = MLP_OPT_ADAM;
    train_cfg.loss_func     = MLP_LOSS_CROSS_ENTROPY;
    train_cfg.batch_size    = CNN_BATCH_SIZE;
    train_cfg.seed          = 99U;
    if (nn_subnet_def_set_infer_type_config(s, infer_cfg,
            mlp_config_size_for_hidden_layers(infer_cfg->hidden_layer_count),
            "types/mlp/mlp_config.h", "MlpConfig") != 0) {
        free(infer_cfg); nn_subnet_def_free(s); return NULL;
    }
    free(infer_cfg);
    if (nn_subnet_def_set_train_type_config(s, &train_cfg, sizeof(train_cfg),
            "types/mlp/mlp_config.h", "MlpTrainConfig") != 0) {
        nn_subnet_def_free(s); return NULL;
    }
    return s;
}

static int wire_1to1(NN_NetworkDef* net,
    const char* src, const char* tgt, size_t count)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        NNConnectionDef* conn = nn_connection_def_create(
            src, "out", i, tgt, "in", i);
        if (!conn) return -1;
        if (nn_network_def_add_connection(net, conn) != 0) {
            nn_connection_def_free(conn);
            return -1;
        }
    }
    return 0;
}

/* ========================================================================
 *  Build BnConvNet v47 for CIFAR-10 (4 CNN + 1 MLP, batch_size=4, lr=0.001, mom=0.0)
 *
 *  Architecture (all layers use BN + LeakyReLU except GAP):
 *    Conv1: 3x3,   3->32,   s=1 -> 30x30x32
 *    Conv2: 3x3,  32->64,   s=1 -> 28x28x64
 *    Conv3: 3x3,  64->128,  s=1 -> 26x26x128
 *    Conv4: 3x3, 128->256,  s=1 -> 24x24x256
 *    GAP:   1x1, 256->256,  POOL_AVG -> 256 scalars
 *    MLP:   256 -> [256(LeakyReLU)] -> 10, ADAM+CE, lr=0.001
 * ======================================================================== */

static NN_NetworkDef* create_network(void) {
    NN_NetworkDef* net;
    NNSubnetDef *agent = NULL, *perception = NULL, *head = NULL;
    NNSubnetDef *leaf = NULL;
    const char* prev_name = NULL;
    size_t prev_out;
    uint32_t seed = 42U;

    typedef NNSubnetDef* (*awl_t)(NN_NetworkDef*, NNSubnetDef*,
        const char*, const char*, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t, size_t,
        size_t, CnnPoolingMode, CnnActivationType, CnnActivationType,
        CnnConvMode, size_t, int, uint32_t, float, float, int);
    awl_t A = (awl_t)add_and_wire;

    net = nn_network_def_create("edge_video_preprocess");
    if (!net) return NULL;

    agent      = nn_subnet_def_create("agent",      NULL, 0U, 0U);
    perception = nn_subnet_def_create("perception", NULL, 0U, 0U);
    head       = nn_subnet_def_create("head",       NULL, 0U, 0U);
    if (!agent || !perception || !head) goto fail;

    /* ── conv1: 3x3, 3->32, s=1, BN, LeakyReLU -> 30x30x32 ── */
    {
        size_t out = 30U * 30U * 32U;
        leaf = A(net, perception, "conv1", NULL, EVP_INPUT_SIZE, out, 0,
            32U, 32U, 3U, 3U, 32U, 0U, CNN_POOL_NONE, CNN_ACT_LEAKY_RELU,
            CNN_ACT_LEAKY_RELU, CNN_CONV_STANDARD, 1U, 1, seed++, CNN_LR, CNN_MOMENTUM, 0);
        if (!leaf) goto fail;
        prev_name = "conv1";
        prev_out  = out;
    }

    /* ── Conv2: 3x3, 32->64, s=1, BN, LeakyReLU -> 28x28x64 ── */
    {
        size_t out = 28U * 28U * 64U;
        leaf = A(net, perception, "conv2", prev_name, prev_out, out, prev_out,
            30U, 30U, 32U, 3U, 64U, 0U, CNN_POOL_NONE, CNN_ACT_LEAKY_RELU,
            CNN_ACT_LEAKY_RELU, CNN_CONV_STANDARD, 1U, 1, seed++, CNN_LR, CNN_MOMENTUM, 1);
        if (!leaf) goto fail;
        prev_name = "conv2";
        prev_out  = out;
    }

    /* ── Conv3: 3x3, 64->128, s=1, BN, LeakyReLU -> 26x26x128 ── */
    {
        size_t out = 26U * 26U * 128U;
        leaf = A(net, perception, "conv3", prev_name, prev_out, out, prev_out,
            28U, 28U, 64U, 3U, 128U, 0U, CNN_POOL_NONE, CNN_ACT_LEAKY_RELU,
            CNN_ACT_LEAKY_RELU, CNN_CONV_STANDARD, 1U, 1, seed++, CNN_LR, CNN_MOMENTUM, 2);
        if (!leaf) goto fail;
        prev_name = "conv3";
        prev_out  = out;
    }

    /* ── Conv4: 3x3, 128->256, s=1, BN, LeakyReLU -> 24x24x256 ── */
    {
        size_t out = 24U * 24U * 256U;
        leaf = A(net, perception, "conv4", prev_name, prev_out, out, prev_out,
            26U, 26U, 128U, 3U, 256U, 0U, CNN_POOL_NONE, CNN_ACT_LEAKY_RELU,
            CNN_ACT_LEAKY_RELU, CNN_CONV_STANDARD, 1U, 1, seed++, CNN_LR, CNN_MOMENTUM, 3);
        if (!leaf) goto fail;
        prev_name = "conv4";
        prev_out  = out;
    }

    /* ── GAP: 1x1 conv + POOL_AVG over 24x24 -> 256 scalars ── */
    {
        size_t feat = GAP_FEATURES;
        leaf = A(net, perception, "gap", prev_name, prev_out, feat, prev_out,
            24U, 24U, 256U, 1U, feat, feat,
            CNN_POOL_AVG, CNN_ACT_NONE, CNN_ACT_NONE,
            CNN_CONV_STANDARD, 1U, 1, seed++, CNN_LR, CNN_MOMENTUM, 4);
        if (!leaf) goto fail;
        prev_name = "gap";
        prev_out  = feat;
    }

    /* ── MLP: 256 -> [256] -> 10, ADAM+CE ── */
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

    printf("Edge Video Preprocessing — Network Code Generator v47 (BnConvNet, batch_size=4, lr=0.001, mom=0.0)\n");
    printf("===================================================================================================\n\n");
    printf("Input: 32x32x3 RGB, CIFAR-10 (10 classes)\n\n");
    printf("Architecture: BnConvNet v47 (4 CNN + BN + LeakyReLU + 1 MLP LeakyReLU)\n");
    printf("  Conv1: 3x3,   3->32,   s=1, BN, LeakyReLU(0.01)  -> 30x30x32\n");
    printf("  Conv2: 3x3,  32->64,   s=1, BN, LeakyReLU(0.01)  -> 28x28x64\n");
    printf("  Conv3: 3x3,  64->128,  s=1, BN, LeakyReLU(0.01)  -> 26x26x128\n");
    printf("  Conv4: 3x3, 128->256,  s=1, BN, LeakyReLU(0.01)  -> 24x24x256\n");
    printf("  GAP:   POOL_AVG over 24x24 -> %u scalars (no activation)\n", (unsigned)GAP_FEATURES);
    printf("  MLP:   %u -> [%u(LeakyReLU)] -> %u, ADAM+CE\n\n",
        (unsigned)GAP_FEATURES, (unsigned)MLP_H1, (unsigned)CLASSES);
    printf("CNN layers: SGD+momentum(%.1f), lr=%.4f, wd=1e-4, LeakyReLU, BN on all\n",
        (double)CNN_MOMENTUM, (double)CNN_LR);
    printf("**Batch_size=%u + lr=%.4f + mom=%.1f** — lr lowered to compensate for 4x sum-gradients\n\n",
        (unsigned)CNN_BATCH_SIZE, (double)CNN_LR, (double)CNN_MOMENTUM);

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
