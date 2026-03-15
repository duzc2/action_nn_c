#ifndef TEST_CASES_H
#define TEST_CASES_H

#include <stddef.h>

#include "test_framework.h"

/**
 * @brief 测试用例分组描述。
 *
 * 设计目的：
 * - 将测试执行器与用例定义解耦，便于按责任拆分多个源文件。
 * - 通过只读数组暴露用例，避免调用方误改测试元数据。
 */
typedef struct TestCaseGroup {
    const TestCase* cases;
    size_t count;
} TestCaseGroup;

/**
 * @brief 获取单元 + 正确性测试分组。
 *
 * @return TestCaseGroup 分组对象
 */
TestCaseGroup testcases_get_unit_correctness_group(void);

/**
 * @brief 获取错误 + 边界测试分组。
 *
 * @return TestCaseGroup 分组对象
 */
TestCaseGroup testcases_get_error_boundary_group(void);

/**
 * @brief 获取压力 + 集成测试分组。
 *
 * @return TestCaseGroup 分组对象
 */
TestCaseGroup testcases_get_stress_integration_group(void);

/**
 * @brief 获取模型专项测试分组（泛化/OOD/对抗扰动/稳定性/一致性）。
 *
 * @return TestCaseGroup 分组对象
 */
TestCaseGroup testcases_get_model_special_group(void);

#endif /* TEST_CASES_H */
