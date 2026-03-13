#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "../include/test_framework.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

/**
 * @brief 全局统计信息，用于统一打印最终汇总。
 */
typedef struct TestSummary {
    size_t passed;
    size_t failed;
} TestSummary;

static FILE* g_test_log_fp = NULL;

static void testfw_try_create_log_dir(void) {
#if defined(_WIN32)
    (void)_mkdir("test/logs");
#else
    (void)mkdir("test", 0755);
    (void)mkdir("test/logs", 0755);
#endif
}

int testfw_init_logging(void) {
    time_t now = 0;
    struct tm* local_tm = NULL;
    char ts[32];
    char path[256];
    testfw_try_create_log_dir();
    now = time(NULL);
    local_tm = localtime(&now);
    if (local_tm != NULL) {
        (void)strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", local_tm);
    } else {
        (void)snprintf(ts, sizeof(ts), "00000000_000000");
    }
    (void)snprintf(path, sizeof(path), "test/logs/test_run_%s.log", ts);
    g_test_log_fp = fopen(path, "w");
    if (g_test_log_fp == NULL) {
        return 1;
    }
    fprintf(g_test_log_fp, "测试日志文件: %s\n", path);
    fflush(g_test_log_fp);
    return 0;
}

void testfw_shutdown_logging(void) {
    if (g_test_log_fp != NULL) {
        fflush(g_test_log_fp);
        fclose(g_test_log_fp);
        g_test_log_fp = NULL;
    }
}

/**
 * @brief 输出带时间戳的实时日志。
 *
 * @param level 日志级别文本
 * @param fmt   printf 风格格式串
 * @param args  可变参数列表
 */
static void testfw_vlog(const char* level, const char* fmt, va_list args) {
    time_t now = 0;
    struct tm* local_tm = NULL;
    char time_buf[32];
    va_list args_copy;
    now = time(NULL);
    local_tm = localtime(&now);
    if (local_tm != NULL) {
        (void)strftime(time_buf, sizeof(time_buf), "%H:%M:%S", local_tm);
    } else {
        (void)snprintf(time_buf, sizeof(time_buf), "00:00:00");
    }
    printf("[实时日志][%s][%s] ", time_buf, level);
    va_copy(args_copy, args);
    vprintf(fmt, args_copy);
    va_end(args_copy);
    printf("\n");
    fflush(stdout);

    if (g_test_log_fp != NULL) {
        fprintf(g_test_log_fp, "[实时日志][%s][%s] ", time_buf, level);
        va_copy(args_copy, args);
        vfprintf(g_test_log_fp, fmt, args_copy);
        va_end(args_copy);
        fprintf(g_test_log_fp, "\n");
        fflush(g_test_log_fp);
    }
}

/**
 * @brief 构造进度条字符串，便于测试执行过程可视化。
 *
 * @param done       已完成数量
 * @param total      总数量
 * @param out_bar    输出缓冲区
 * @param bar_length 进度条宽度
 */
static void testfw_build_progress_bar(size_t done, size_t total, char* out_bar, size_t bar_length) {
    size_t filled = 0U;
    size_t i = 0U;
    if (out_bar == NULL || bar_length < 3U || total == 0U) {
        return;
    }
    filled = (done * (bar_length - 2U)) / total;
    out_bar[0] = '[';
    for (i = 1U; i < bar_length - 1U; ++i) {
        out_bar[i] = (i <= filled) ? '#' : '.';
    }
    out_bar[bar_length - 1U] = ']';
    out_bar[bar_length] = '\0';
}

/**
 * @brief 计算毫秒耗时。
 *
 * @param begin 起始时钟
 * @param end   结束时钟
 * @return double 毫秒
 */
static double testfw_elapsed_ms(clock_t begin, clock_t end) {
    return ((double)(end - begin) * 1000.0) / (double)CLOCKS_PER_SEC;
}

/**
 * @brief 运行测试集合并输出进度条与实时日志。
 *
 * @param cases      测试数组
 * @param case_count 测试数量
 * @return int       0=全部通过，非0=存在失败
 */
int testfw_run_all(const TestCase* cases, size_t case_count) {
    size_t i = 0U;
    TestSummary summary;
    summary.passed = 0U;
    summary.failed = 0U;

    if (testfw_init_logging() != 0) {
        printf("无法创建测试日志文件，测试中止。\n");
        fflush(stdout);
        return 1;
    }

    if (cases == NULL || case_count == 0U) {
        testfw_log_error("未提供测试用例，无法执行。");
        testfw_shutdown_logging();
        return 1;
    }

    printf("========================================\n");
    printf(" C99 全量测试套件开始执行（共 %zu 项）\n", case_count);
    printf("========================================\n");
    fflush(stdout);
    if (g_test_log_fp != NULL) {
        fprintf(g_test_log_fp, "========================================\n");
        fprintf(g_test_log_fp, " C99 全量测试套件开始执行（共 %zu 项）\n", case_count);
        fprintf(g_test_log_fp, "========================================\n");
        fflush(g_test_log_fp);
    }

    for (i = 0U; i < case_count; ++i) {
        clock_t begin = 0;
        clock_t end = 0;
        double elapsed = 0.0;
        int rc = 0;
        char bar[34];
        int percent = (int)(((i + 1U) * 100U) / case_count);

        printf("\n[执行] (%zu/%zu) [%s] %s\n",
               i + 1U,
               case_count,
               cases[i].category,
               cases[i].name);
        fflush(stdout);
        testfw_log_case_meta(&cases[i], i + 1U, case_count);

        begin = clock();
        rc = cases[i].fn();
        end = clock();
        elapsed = testfw_elapsed_ms(begin, end);

        testfw_build_progress_bar(i + 1U, case_count, bar, 33U);
        if (rc == 0) {
            summary.passed += 1U;
            printf("[结果] PASS %s %3d%% 用时 %.3f ms\n", bar, percent, elapsed);
            testfw_log_case_actual("实际返回值=0(PASS), 期望返回值=0, 用时=%.3fms, 进度=%d%%", elapsed, percent);
        } else {
            summary.failed += 1U;
            printf("[结果] FAIL %s %3d%% 用时 %.3f ms\n", bar, percent, elapsed);
            testfw_log_case_actual("实际返回值=%d(FAIL), 期望返回值=0, 用时=%.3fms, 进度=%d%%", rc, elapsed, percent);
        }
        fflush(stdout);
        if (g_test_log_fp != NULL) {
            fprintf(g_test_log_fp, "[进度] %s %d%%\n", bar, percent);
            fflush(g_test_log_fp);
        }
    }

    printf("\n========================================\n");
    printf(" 测试完成：通过 %zu，失败 %zu，总计 %zu\n",
           summary.passed,
           summary.failed,
           case_count);
    printf("========================================\n");
    fflush(stdout);
    if (g_test_log_fp != NULL) {
        fprintf(g_test_log_fp, "========================================\n");
        fprintf(g_test_log_fp, " 测试完成：通过 %zu，失败 %zu，总计 %zu\n",
                summary.passed,
                summary.failed,
                case_count);
        fprintf(g_test_log_fp, "========================================\n");
        fflush(g_test_log_fp);
    }

    testfw_shutdown_logging();

    return (summary.failed == 0U) ? 0 : 1;
}

void testfw_log_case_meta(const TestCase* tc, size_t index, size_t total) {
    if (tc == NULL) {
        return;
    }
    testfw_log_info("测试进度: %zu/%zu", index, total);
    testfw_log_info("测试内容: [%s] %s", tc->category, tc->name);
    testfw_log_info("测试目的: %s", (tc->purpose != NULL) ? tc->purpose : "未提供");
    testfw_log_info("测试参数: %s", (tc->params != NULL) ? tc->params : "未提供");
    testfw_log_info("期望返回值: %s", (tc->expected != NULL) ? tc->expected : "0(PASS)");
}

void testfw_log_case_actual(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    testfw_vlog("ACTUAL", fmt, args);
    va_end(args);
}

/**
 * @brief 输出实时日志（INFO 级别）。
 *
 * @param fmt printf 风格格式串
 * @param ... 可变参数
 */
void testfw_log_info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    testfw_vlog("INFO", fmt, args);
    va_end(args);
}

/**
 * @brief 输出实时日志（ERROR 级别）。
 *
 * @param fmt printf 风格格式串
 * @param ... 可变参数
 */
void testfw_log_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    testfw_vlog("ERROR", fmt, args);
    va_end(args);
}

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
int testfw_assert_int_eq(int expected, int actual, const char* expr, const char* file, int line) {
    if (expected != actual) {
        testfw_log_error("断言失败：%s，期望=%d，实际=%d，位置=%s:%d", expr, expected, actual, file, line);
        return 0;
    }
    return 1;
}

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
int testfw_assert_size_eq(size_t expected, size_t actual, const char* expr, const char* file, int line) {
    if (expected != actual) {
        testfw_log_error("断言失败：%s，期望=%zu，实际=%zu，位置=%s:%d", expr, expected, actual, file, line);
        return 0;
    }
    return 1;
}

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
                             int line) {
    float diff = fabsf(expected - actual);
    if (diff > eps) {
        testfw_log_error("断言失败：%s，期望=%.7f，实际=%.7f，误差=%.7f，阈值=%.7f，位置=%s:%d",
                         expr,
                         (double)expected,
                         (double)actual,
                         (double)diff,
                         (double)eps,
                         file,
                         line);
        return 0;
    }
    return 1;
}

/**
 * @brief 测试断言：条件为真。
 *
 * @param condition 条件值
 * @param expr      断言表达式文本
 * @param file      文件名
 * @param line      行号
 * @return int      1=断言成功，0=断言失败
 */
int testfw_assert_true(int condition, const char* expr, const char* file, int line) {
    if (!condition) {
        testfw_log_error("断言失败：%s，位置=%s:%d", expr, file, line);
        return 0;
    }
    return 1;
}
