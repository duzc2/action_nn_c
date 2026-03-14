#include "../include/test_cases.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/include/config_user.h"
#include "../../src/include/weights_io.h"

size_t g_demo_weights_count(void);
const float* g_demo_weights_data(void);
int g_demo_weights_copy(float* out, size_t out_capacity);

size_t g_demo_network_vocab_size(void);
size_t g_demo_network_state_dim(void);
size_t g_demo_network_output_dim(void);
size_t g_demo_network_token_slots(void);
void g_demo_network_forward(const float* token_onehot, const float* state, float* out);

static float test_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static int test_generated_weights_module_consistency(void) {
    size_t i = 0U;
    size_t count = g_demo_weights_count();
    const float* data = g_demo_weights_data();
    float* copy = NULL;
    TFW_ASSERT_TRUE(count > 0U);
    TFW_ASSERT_TRUE(data != NULL);
    copy = (float*)malloc(sizeof(float) * count);
    TFW_ASSERT_TRUE(copy != NULL);
    TFW_ASSERT_INT_EQ(0, g_demo_weights_copy(copy, count));
    for (i = 0U; i < count; ++i) {
        TFW_ASSERT_FLOAT_NEAR(data[i], copy[i], 1e-7f);
    }
    free(copy);
    return 0;
}

static int test_generated_bin_parse_and_forward_match(void) {
    size_t i = 0U;
    size_t j = 0U;
    size_t t = 0U;
    size_t vocab = g_demo_network_vocab_size();
    size_t state_dim = g_demo_network_state_dim();
    size_t out_dim = g_demo_network_output_dim();
    size_t slots = g_demo_network_token_slots();
    size_t count = 0U;
    float* weights = NULL;
    float* onehot = NULL;
    float* state = NULL;
    float out_gen[OUTPUT_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out_ref[OUTPUT_DIM] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int activations[OUTPUT_DIM] = IO_MAPPING_ACTIVATIONS;
    TFW_ASSERT_SIZE_EQ(VOCAB_SIZE, vocab);
    TFW_ASSERT_SIZE_EQ(STATE_DIM, state_dim);
    TFW_ASSERT_SIZE_EQ(OUTPUT_DIM, out_dim);
    TFW_ASSERT_TRUE(slots > 0U);
    TFW_ASSERT_INT_EQ(WEIGHTS_IO_STATUS_OK, weights_load_binary("demo_weights.bin", &weights, &count));
    TFW_ASSERT_TRUE(weights != NULL);
    TFW_ASSERT_SIZE_EQ(g_demo_weights_count(), count);

    onehot = (float*)calloc(slots * vocab, sizeof(float));
    state = (float*)calloc(state_dim, sizeof(float));
    TFW_ASSERT_TRUE(onehot != NULL);
    TFW_ASSERT_TRUE(state != NULL);

    onehot[0U * vocab + 2U] = 1.0f;
    onehot[1U * vocab + 3U] = 1.0f;
    onehot[2U * vocab + 5U] = 1.0f;
    state[0] = 0.15f;
    state[1] = -0.2f;
    state[2] = 0.4f;
    state[3] = 0.1f;
    state[4] = -0.05f;
    state[5] = 0.22f;
    state[6] = 0.3f;
    state[7] = 1.0f;

    g_demo_network_forward(onehot, state, out_gen);

    {
        const size_t token_weight_count = vocab * out_dim;
        const size_t state_weight_count = state_dim * out_dim;
        for (j = 0U; j < out_dim; ++j) {
            float z = weights[token_weight_count + state_weight_count + j];
            for (t = 0U; t < slots; ++t) {
                for (i = 0U; i < vocab; ++i) {
                    z += onehot[t * vocab + i] * (weights[i * out_dim + j] / (float)slots);
                }
            }
            for (i = 0U; i < state_dim; ++i) {
                z += state[i] * weights[token_weight_count + i * out_dim + j];
            }
            out_ref[j] = (activations[j] == 0) ? test_sigmoid(z) : tanhf(z);
        }
    }

    for (j = 0U; j < out_dim; ++j) {
        TFW_ASSERT_TRUE(isfinite(out_gen[j]));
        TFW_ASSERT_FLOAT_NEAR(out_ref[j], out_gen[j], 1e-5f);
    }

    free(state);
    free(onehot);
    free(weights);
    return 0;
}

TestCaseGroup testcases_get_generated_artifacts_group(void) {
    static const TestCase cases[] = {
        {"generated_weights_module_consistency", "集成", "验证导出C数据模块函数可调用且数据一致", "count/data/copy", "0(PASS)", test_generated_weights_module_consistency},
        {"generated_bin_parse_and_forward_match", "集成", "验证bin可解析且函数网络前向结果与权重计算一致", "demo_weights.bin + demo_network_functions.c", "0(PASS)", test_generated_bin_parse_and_forward_match}
    };
    TestCaseGroup group;
    group.cases = cases;
    group.count = sizeof(cases) / sizeof(cases[0]);
    return group;
}
