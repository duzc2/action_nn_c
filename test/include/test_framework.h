#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <stddef.h>

/**
 * @brief 单个测试用例函数签名。
 *
 * 返回约定：
 * - 返回 0 表示通过。
 * - 返回非 0 表示失败。
 */
typedef int (*TestCaseFn)(void);

/**
 * @brief 测试用例元数据。
 *
 * 设计目的：
 * - 将“名称/分类/函数入口”解耦，便于按分类扩展测试而不改动执行器核心。
 */
typedef struct TestCase {
    const char* name;
    const char* category;
    const char* purpose;
    const char* params;
    const char* expected;
    TestCaseFn fn;
} TestCase;

/**
 * @brief 运行测试集合并输出进度条与实时日志。
 *
 * @param cases      测试数组
 * @param case_count 测试数量
 * @return int       0=全部通过，非0=存在失败
 */
int testfw_run_all(const TestCase* cases, size_t case_count);

int testfw_init_logging(void);
void testfw_shutdown_logging(void);
void testfw_log_case_meta(const TestCase* tc, size_t index, size_t total);
void testfw_log_case_actual(const char* fmt, ...);

/**
 * @brief 输出实时日志（INFO 级别）。
 *
 * @param fmt printf 风格格式串
 * @param ... 可变参数
 */
void testfw_log_info(const char* fmt, ...);

/**
 * @brief 输出实时日志（ERROR 级别）。
 *
 * @param fmt printf 风格格式串
 * @param ... 可变参数
 */
void testfw_log_error(const char* fmt, ...);

/**
 * @brief 测试断言：整数相等。
 *
 * @param expected 期望值
 * @param actual   实际值
 * @param expr     断言表达式文本
 * @param file     文件名
 * @param line     行号
 * @return int     1=断言成功，0=断言失败
 */
int testfw_assert_int_eq(int expected, int actual, const char* expr, const char* file, int line);

/**
 * @brief 测试断言：size_t 相等。
 *
 * @param expected 期望值
 * @param actual   实际值
 * @param expr     断言表达式文本
 * @param file     文件名
 * @param line     行号
 * @return int     1=断言成功，0=断言失败
 */
int testfw_assert_size_eq(size_t expected, size_t actual, const char* expr, const char* file, int line);

/**
 * @brief 测试断言：浮点近似相等。
 *
 * @param expected 期望值
 * @param actual   实际值
 * @param eps      误差阈值
 * @param expr     断言表达式文本
 * @param file     文件名
 * @param line     行号
 * @return int     1=断言成功，0=断言失败
 */
int testfw_assert_float_near(float expected,
                             float actual,
                             float eps,
                             const char* expr,
                             const char* file,
                             int line);

/**
 * @brief 测试断言：条件为真。
 *
 * @param condition 条件值
 * @param expr      断言表达式文本
 * @param file      文件名
 * @param line      行号
 * @return int      1=断言成功，0=断言失败
 */
int testfw_assert_true(int condition, const char* expr, const char* file, int line);

/**
 * @brief 便捷宏：整数相等断言。
 */
#define TFW_ASSERT_INT_EQ(expected, actual) \
    do { \
        if (!testfw_assert_int_eq((expected), (actual), #expected " == " #actual, __FILE__, __LINE__)) { \
            return 1; \
        } \
    } while (0)

/**
 * @brief 便捷宏：size_t 相等断言。
 */
#define TFW_ASSERT_SIZE_EQ(expected, actual) \
    do { \
        if (!testfw_assert_size_eq((expected), (actual), #expected " == " #actual, __FILE__, __LINE__)) { \
            return 1; \
        } \
    } while (0)

/**
 * @brief 便捷宏：浮点近似断言。
 */
#define TFW_ASSERT_FLOAT_NEAR(expected, actual, eps) \
    do { \
        if (!testfw_assert_float_near((expected), (actual), (eps), #expected " ~= " #actual, __FILE__, __LINE__)) { \
            return 1; \
        } \
    } while (0)

/**
 * @brief 便捷宏：布尔断言。
 */
#define TFW_ASSERT_TRUE(condition) \
    do { \
        if (!testfw_assert_true((condition), #condition, __FILE__, __LINE__)) { \
            return 1; \
        } \
    } while (0)

#endif /* TEST_FRAMEWORK_H */
