#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "../include/test_cases.h"

#include <string.h>

#include "../../src/include/ops.h"
#include "../../src/include/tensor.h"
#include "../../src/include/tokenizer.h"

/**
 * @brief 单元测试：验证 tensor 计算元素数量与 shape 比较。
 *
 * @return int 0=通过，非0=失败
 */
static int test_unit_tensor_numel_and_shape(void) {
    size_t shape[2] = {2U, 3U};
    size_t numel = 0U;
    float data_a[6];
    float data_b[6];
    Tensor a;
    Tensor b;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_calc_numel(2U, shape, &numel));
    TFW_ASSERT_SIZE_EQ(6U, numel);
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&a, data_a, 2U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&b, data_b, 2U, shape));
    TFW_ASSERT_TRUE(tensor_same_shape(&a, &b));
    return 0;
}

/**
 * @brief 单元测试：验证 tensor_fill 可覆盖全部元素。
 *
 * @return int 0=通过，非0=失败
 */
static int test_unit_tensor_fill(void) {
    size_t i = 0U;
    size_t shape[2] = {2U, 2U};
    float data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    Tensor t;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&t, data, 2U, shape));
    tensor_fill(&t, -3.5f);
    for (i = 0U; i < 4U; ++i) {
        TFW_ASSERT_FLOAT_NEAR(-3.5f, data[i], 1e-6f);
    }
    return 0;
}

/**
 * @brief 单元测试：验证内存池分配与 reset 逻辑。
 *
 * @return int 0=通过，非0=失败
 */
static int test_unit_tensor_pool_alloc_reset(void) {
    float pool_buf[8] = {0};
    TensorPool pool;
    Tensor t1;
    Tensor t2;
    Tensor t3;
    size_t shape_a[1] = {3U};
    size_t shape_b[1] = {6U};
    size_t shape_c[1] = {5U};
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_pool_init(&pool, pool_buf, 8U));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_pool_alloc(&pool, &t1, 1U, shape_a));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_POOL_EXHAUSTED, tensor_pool_alloc(&pool, &t2, 1U, shape_b));
    tensor_pool_reset(&pool);
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_pool_alloc(&pool, &t3, 1U, shape_c));
    return 0;
}

/**
 * @brief 单元测试：2D 矩阵乘法结果正确。
 *
 * @return int 0=通过，非0=失败
 */
static int test_unit_matmul_basic(void) {
    size_t a_shape[2] = {2U, 2U};
    size_t b_shape[2] = {2U, 2U};
    size_t out_shape[2] = {2U, 2U};
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float out_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    Tensor a;
    Tensor b;
    Tensor out;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&a, a_data, 2U, a_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&b, b_data, 2U, b_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out, out_data, 2U, out_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, op_matmul_2d(&a, &b, &out));
    TFW_ASSERT_FLOAT_NEAR(19.0f, out_data[0], 1e-5f);
    TFW_ASSERT_FLOAT_NEAR(22.0f, out_data[1], 1e-5f);
    TFW_ASSERT_FLOAT_NEAR(43.0f, out_data[2], 1e-5f);
    TFW_ASSERT_FLOAT_NEAR(50.0f, out_data[3], 1e-5f);
    return 0;
}

/**
 * @brief 单元测试：GELU 在采样点满足单调趋势。
 *
 * @return int 0=通过，非0=失败
 */
static int test_unit_gelu_monotonic_samples(void) {
    size_t shape[1] = {3U};
    float in_data[3] = {-1.0f, 0.0f, 1.0f};
    float out_data[3] = {0.0f, 0.0f, 0.0f};
    Tensor in;
    Tensor out;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&in, in_data, 1U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out, out_data, 1U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, op_gelu(&in, &out));
    TFW_ASSERT_TRUE(out_data[0] < out_data[1]);
    TFW_ASSERT_TRUE(out_data[1] < out_data[2]);
    TFW_ASSERT_FLOAT_NEAR(0.0f, out_data[1], 1e-6f);
    return 0;
}

/**
 * @brief 正确性测试：softmax 每行和应接近 1。
 *
 * @return int 0=通过，非0=失败
 */
static int test_correctness_softmax_sum_is_one(void) {
    size_t shape[2] = {2U, 3U};
    float in_data[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};
    float out_data[6] = {0};
    Tensor in;
    Tensor out;
    float row0 = 0.0f;
    float row1 = 0.0f;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&in, in_data, 2U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out, out_data, 2U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, op_softmax_last_dim(&in, &out));
    row0 = out_data[0] + out_data[1] + out_data[2];
    row1 = out_data[3] + out_data[4] + out_data[5];
    TFW_ASSERT_FLOAT_NEAR(1.0f, row0, 1e-5f);
    TFW_ASSERT_FLOAT_NEAR(1.0f, row1, 1e-5f);
    return 0;
}

/**
 * @brief 正确性测试：softmax 对常数平移不变。
 *
 * @return int 0=通过，非0=失败
 */
static int test_correctness_softmax_shift_invariance(void) {
    size_t i = 0U;
    size_t shape[1] = {4U};
    float in_a_data[4] = {0.2f, 1.1f, -2.3f, 4.0f};
    float in_b_data[4] = {1000.2f, 1001.1f, 997.7f, 1004.0f};
    float out_a_data[4] = {0};
    float out_b_data[4] = {0};
    Tensor in_a;
    Tensor in_b;
    Tensor out_a;
    Tensor out_b;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&in_a, in_a_data, 1U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&in_b, in_b_data, 1U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out_a, out_a_data, 1U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out_b, out_b_data, 1U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, op_softmax_last_dim(&in_a, &out_a));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, op_softmax_last_dim(&in_b, &out_b));
    for (i = 0U; i < 4U; ++i) {
        TFW_ASSERT_FLOAT_NEAR(out_a_data[i], out_b_data[i], 1e-5f);
    }
    return 0;
}

/**
 * @brief 正确性测试：RMSNorm 在单位权重下输出均方根接近 1。
 *
 * @return int 0=通过，非0=失败
 */
static int test_correctness_rmsnorm_unit_weight_rms_one(void) {
    size_t shape[2] = {2U, 2U};
    size_t w_shape[1] = {2U};
    float in_data[4] = {2.0f, 4.0f, -3.0f, 1.0f};
    float w_data[2] = {1.0f, 1.0f};
    float out_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    Tensor in;
    Tensor weight;
    Tensor out;
    float rms_row0 = 0.0f;
    float rms_row1 = 0.0f;
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&in, in_data, 2U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&weight, w_data, 1U, w_shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, tensor_init_view(&out, out_data, 2U, shape));
    TFW_ASSERT_INT_EQ(TENSOR_STATUS_OK, op_rmsnorm_last_dim(&in, &weight, 1e-6f, &out));
    rms_row0 = (out_data[0] * out_data[0] + out_data[1] * out_data[1]) / 2.0f;
    rms_row1 = (out_data[2] * out_data[2] + out_data[3] * out_data[3]) / 2.0f;
    TFW_ASSERT_FLOAT_NEAR(1.0f, rms_row0, 1e-4f);
    TFW_ASSERT_FLOAT_NEAR(1.0f, rms_row1, 1e-4f);
    return 0;
}

/**
 * @brief 正确性测试：Tokenizer 已登录词编码并可还原。
 *
 * @return int 0=通过，非0=失败
 */
static int test_correctness_tokenizer_known_tokens_roundtrip(void) {
    Vocabulary vocab;
    Tokenizer tokenizer;
    int ids[8] = {0};
    size_t count = 0U;
    char decoded[64];
    memset(&vocab, 0, sizeof(vocab));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_init(&vocab, 8U));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "<unk>", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "go", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, vocab_add_token(&vocab, "right", NULL));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tokenizer_init(&tokenizer, &vocab, 0));
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tokenizer_encode(&tokenizer, "go right", ids, 8U, &count));
    TFW_ASSERT_SIZE_EQ(2U, count);
    TFW_ASSERT_INT_EQ(1, ids[0]);
    TFW_ASSERT_INT_EQ(2, ids[1]);
    TFW_ASSERT_INT_EQ(TOKENIZER_STATUS_OK, tokenizer_decode(&vocab, ids, count, decoded, sizeof(decoded)));
    TFW_ASSERT_TRUE(strcmp(decoded, "go right") == 0);
    vocab_free(&vocab);
    return 0;
}

/**
 * @brief 返回“单元 + 正确性”测试分组。
 *
 * @return TestCaseGroup 分组对象
 */
TestCaseGroup testcases_get_unit_correctness_group(void) {
    static const TestCase cases[] = {
        {"tensor_numel_and_shape", "单元", "验证张量元素计数与形状比较逻辑正确", "shape=[2,3]", "0(PASS)", test_unit_tensor_numel_and_shape},
        {"tensor_fill", "单元", "验证tensor_fill可覆盖全部元素", "2x2张量填充-3.5", "0(PASS)", test_unit_tensor_fill},
        {"tensor_pool_alloc_reset", "单元", "验证内存池分配上限与reset后复用", "pool=8,float alloc=3/5", "0(PASS)", test_unit_tensor_pool_alloc_reset},
        {"matmul_basic", "单元", "验证2x2矩阵乘法结果正确性", "A/B固定矩阵", "0(PASS)", test_unit_matmul_basic},
        {"gelu_monotonic_samples", "单元", "验证GELU在采样点的单调趋势", "x={-1,0,1}", "0(PASS)", test_unit_gelu_monotonic_samples},
        {"softmax_sum_is_one", "正确性", "验证softmax每行归一化求和为1", "输入2x3矩阵", "0(PASS)", test_correctness_softmax_sum_is_one},
        {"softmax_shift_invariance", "正确性", "验证softmax对常数平移不变", "向量+1000偏移", "0(PASS)", test_correctness_softmax_shift_invariance},
        {"rmsnorm_unit_weight_rms_one", "正确性", "验证RMSNorm单位权重时输出均方根接近1", "2行2列输入", "0(PASS)", test_correctness_rmsnorm_unit_weight_rms_one},
        {"tokenizer_known_tokens_roundtrip", "正确性", "验证已登录词编码解码一致", "go right样本", "0(PASS)", test_correctness_tokenizer_known_tokens_roundtrip}
    };
    TestCaseGroup group;
    group.cases = cases;
    group.count = sizeof(cases) / sizeof(cases[0]);
    return group;
}
