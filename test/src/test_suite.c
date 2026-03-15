#include "../include/test_cases.h"

#include <string.h>

/**
 * @brief 将分组中的测试用例追加到统一数组。
 *
 * 设计目的：
 * - 维持 testfw_run_all 的简单接口（一次接收连续数组）。
 * - 允许用例在多个文件按职责拆分定义，降低单文件复杂度。
 *
 * @param dst        目标数组
 * @param dst_cap    目标数组容量
 * @param offset     当前已写入数量
 * @param group      待追加分组
 * @return size_t    新的已写入数量；若失败返回原 offset
 */
static size_t append_case_group(TestCase* dst,
                                size_t dst_cap,
                                size_t offset,
                                TestCaseGroup group) {
    if (dst == NULL || offset > dst_cap || group.cases == NULL) {
        return offset;
    }
    if (group.count > dst_cap - offset) {
        return offset;
    }
    memcpy(&dst[offset], group.cases, sizeof(TestCase) * group.count);
    return offset + group.count;
}

/**
 * @brief 程序入口：注册并执行扩展后的全量测试。
 *
 * 分类覆盖：
 * - 单元测试
 * - 正确性测试
 * - 错误测试
 * - 边界测试
 * - 压力测试
 * - 集成测试
 *
 * @return int 进程返回码
 */
int main(void) {
    TestCase all_cases[96];
    size_t count = 0U;
    count = append_case_group(all_cases, 96U, count, testcases_get_unit_correctness_group());
    count = append_case_group(all_cases, 96U, count, testcases_get_error_boundary_group());
    count = append_case_group(all_cases, 96U, count, testcases_get_stress_integration_group());
    count = append_case_group(all_cases, 96U, count, testcases_get_model_special_group());
    return testfw_run_all(all_cases, count);
}
