#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "../include/test_framework.h"

#include <math.h>
#include <stdlib.h>
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
 * @brief 单个类别统计。
 *
 * 设计目的：
 * - 按“单元/正确性/错误/边界/压力/集成”等分类聚合通过率与耗时。
 * - 避免新增复杂容器，使用固定上限数组保证 C99 可移植性。
 */
typedef struct CategorySummary {
    char category[32];
    size_t total;
    size_t passed;
    size_t failed;
    double elapsed_ms;
} CategorySummary;

/**
 * @brief 测试套件总统计。
 *
 * 统计内容：
 * - 用例通过/失败数
 * - 断言总数与失败数
 * - 总耗时
 * - 分类汇总
 */
typedef struct TestSummary {
    size_t passed;
    size_t failed;
    size_t assertions_total;
    size_t assertions_failed;
    double elapsed_ms;
    CategorySummary categories[32];
    size_t category_count;
} TestSummary;

static FILE* g_test_log_fp = NULL;
static size_t g_case_assert_total = 0U;
static size_t g_case_assert_failed = 0U;

/**
 * @brief 重置单用例断言计数器。
 */
static void testfw_reset_case_assert_stats(void) {
    g_case_assert_total = 0U;
    g_case_assert_failed = 0U;
}

/**
 * @brief 记录一次断言执行结果。
 *
 * @param passed 1=通过，0=失败
 */
static void testfw_record_assert(int passed) {
    g_case_assert_total += 1U;
    if (!passed) {
        g_case_assert_failed += 1U;
    }
}

/**
 * @brief 创建日志目录，确保文件日志可用。
 */
static void testfw_try_create_log_dir(void) {
#if defined(_WIN32)
    (void)_mkdir("test");
    (void)_mkdir("test/logs");
#else
    (void)mkdir("test", 0755);
    (void)mkdir("test/logs", 0755);
#endif
}

/**
 * @brief 初始化日志系统，生成本次测试日志文件。
 *
 * @return int 0=成功，非0=失败
 */
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

/**
 * @brief 关闭日志系统并刷新文件缓冲。
 */
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
 * @brief 在汇总结构中查找/创建分类统计项。
 *
 * @param summary 总统计对象
 * @param category 分类名
 * @return CategorySummary* 分类统计指针；失败返回 NULL
 */
static CategorySummary* testfw_find_or_create_category(TestSummary* summary, const char* category) {
    size_t i = 0U;
    const char* safe_category = (category != NULL) ? category : "未分类";
    if (summary == NULL) {
        return NULL;
    }
    for (i = 0U; i < summary->category_count; ++i) {
        if (strcmp(summary->categories[i].category, safe_category) == 0) {
            return &summary->categories[i];
        }
    }
    if (summary->category_count >= (sizeof(summary->categories) / sizeof(summary->categories[0]))) {
        return NULL;
    }
    {
        CategorySummary* created = &summary->categories[summary->category_count];
        memset(created, 0, sizeof(*created));
        (void)snprintf(created->category, sizeof(created->category), "%s", safe_category);
        summary->category_count += 1U;
        return created;
    }
}

/**
 * @brief 打印套件汇总与分类汇总。
 *
 * @param summary 总统计对象
 * @param case_count 用例总数
 */
static void testfw_print_summary(const TestSummary* summary, size_t case_count) {
    size_t i = 0U;
    double pass_rate = 0.0;
    double assert_pass_rate = 0.0;
    if (summary == NULL || case_count == 0U) {
        return;
    }
    pass_rate = ((double)summary->passed * 100.0) / (double)case_count;
    if (summary->assertions_total > 0U) {
        assert_pass_rate = ((double)(summary->assertions_total - summary->assertions_failed) * 100.0)
                           / (double)summary->assertions_total;
    }
    printf("\n========================================\n");
    printf(" 测试完成：通过 %zu，失败 %zu，总计 %zu\n", summary->passed, summary->failed, case_count);
    printf(" 套件耗时：%.3f ms，用例通过率：%.2f%%\n", summary->elapsed_ms, pass_rate);
    printf(" 断言统计：总计 %zu，失败 %zu，通过率 %.2f%%\n",
           summary->assertions_total,
           summary->assertions_failed,
           assert_pass_rate);
    printf("========================================\n");
    printf(" 分类汇总：\n");
    for (i = 0U; i < summary->category_count; ++i) {
        const CategorySummary* c = &summary->categories[i];
        double cat_rate = (c->total > 0U) ? ((double)c->passed * 100.0) / (double)c->total : 0.0;
        printf(" - [%s] total=%zu pass=%zu fail=%zu pass_rate=%.2f%% elapsed=%.3fms\n",
               c->category,
               c->total,
               c->passed,
               c->failed,
               cat_rate,
               c->elapsed_ms);
    }
    fflush(stdout);
    if (g_test_log_fp != NULL) {
        fprintf(g_test_log_fp, "\n========================================\n");
        fprintf(g_test_log_fp, " 测试完成：通过 %zu，失败 %zu，总计 %zu\n", summary->passed, summary->failed, case_count);
        fprintf(g_test_log_fp, " 套件耗时：%.3f ms，用例通过率：%.2f%%\n", summary->elapsed_ms, pass_rate);
        fprintf(g_test_log_fp, " 断言统计：总计 %zu，失败 %zu，通过率 %.2f%%\n",
                summary->assertions_total,
                summary->assertions_failed,
                assert_pass_rate);
        fprintf(g_test_log_fp, "========================================\n");
        fprintf(g_test_log_fp, " 分类汇总：\n");
        for (i = 0U; i < summary->category_count; ++i) {
            const CategorySummary* c = &summary->categories[i];
            double cat_rate = (c->total > 0U) ? ((double)c->passed * 100.0) / (double)c->total : 0.0;
            fprintf(g_test_log_fp,
                    " - [%s] total=%zu pass=%zu fail=%zu pass_rate=%.2f%% elapsed=%.3fms\n",
                    c->category,
                    c->total,
                    c->passed,
                    c->failed,
                    cat_rate,
                    c->elapsed_ms);
        }
        fflush(g_test_log_fp);
    }
}

/**
 * @brief 运行测试集合并输出进度条与结构化实时日志。
 *
 * @param cases      测试数组
 * @param case_count 测试数量
 * @return int       0=全部通过，非0=存在失败
 */
int testfw_run_all(const TestCase* cases, size_t case_count) {
    size_t i = 0U;
    TestSummary summary;
    clock_t suite_begin = 0;
    clock_t suite_end = 0;
    double cumulative_elapsed = 0.0;
    memset(&summary, 0, sizeof(summary));

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

    suite_begin = clock();
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
        size_t case_assert_total = 0U;
        size_t case_assert_failed = 0U;
        size_t case_assert_passed = 0U;
        int percent = (int)(((i + 1U) * 100U) / case_count);
        CategorySummary* category_summary = NULL;

        printf("\n[执行] (%zu/%zu) [%s] %s\n",
               i + 1U,
               case_count,
               cases[i].category,
               cases[i].name);
        fflush(stdout);
        testfw_log_case_meta(&cases[i], i + 1U, case_count);

        testfw_reset_case_assert_stats();
        begin = clock();
        rc = cases[i].fn();
        end = clock();
        elapsed = testfw_elapsed_ms(begin, end);
        cumulative_elapsed += elapsed;
        case_assert_total = g_case_assert_total;
        case_assert_failed = g_case_assert_failed;
        case_assert_passed = case_assert_total - case_assert_failed;
        summary.assertions_total += case_assert_total;
        summary.assertions_failed += case_assert_failed;
        category_summary = testfw_find_or_create_category(&summary, cases[i].category);
        if (category_summary != NULL) {
            category_summary->total += 1U;
            category_summary->elapsed_ms += elapsed;
        }

        testfw_build_progress_bar(i + 1U, case_count, bar, 33U);
        if (rc == 0) {
            summary.passed += 1U;
            if (category_summary != NULL) {
                category_summary->passed += 1U;
            }
            printf("[结果] PASS %s %3d%% 用时 %.3f ms\n", bar, percent, elapsed);
            testfw_log_case_actual("status=PASS rc=0 expected=0 elapsed_ms=%.3f progress=%d%% assert_total=%zu assert_failed=%zu",
                                   elapsed,
                                   percent,
                                   case_assert_total,
                                   case_assert_failed);
        } else {
            summary.failed += 1U;
            if (category_summary != NULL) {
                category_summary->failed += 1U;
            }
            printf("[结果] FAIL %s %3d%% 用时 %.3f ms\n", bar, percent, elapsed);
            testfw_log_case_actual("status=FAIL rc=%d expected=0 elapsed_ms=%.3f progress=%d%% assert_total=%zu assert_failed=%zu",
                                   rc,
                                   elapsed,
                                   percent,
                                   case_assert_total,
                                   case_assert_failed);
        }
        testfw_log_info("case_index=%zu case_name=%s category=%s elapsed_ms=%.3f cumulative_ms=%.3f assert_passed=%zu assert_failed=%zu",
                        i + 1U,
                        (cases[i].name != NULL) ? cases[i].name : "unknown",
                        (cases[i].category != NULL) ? cases[i].category : "未分类",
                        elapsed,
                        cumulative_elapsed,
                        case_assert_passed,
                        case_assert_failed);
        fflush(stdout);
        if (g_test_log_fp != NULL) {
            fprintf(g_test_log_fp, "[进度] %s %d%%\n", bar, percent);
            fflush(g_test_log_fp);
        }
    }
    suite_end = clock();
    summary.elapsed_ms = testfw_elapsed_ms(suite_begin, suite_end);
    testfw_print_summary(&summary, case_count);

    testfw_shutdown_logging();

    return (summary.failed == 0U) ? 0 : 1;
}

/**
 * @brief 输出用例元数据日志，统一记录“预期信息”。
 *
 * 关键约束：
 * - 仅做日志输出，不改变用例状态，避免影响测试行为。
 * - 对可空字段提供兜底文案，防止日志打印时出现空指针访问。
 *
 * @param tc    测试用例元信息
 * @param index 当前序号（从 1 开始）
 * @param total 总用例数
 */
void testfw_log_case_meta(const TestCase* tc, size_t index, size_t total) {
    if (tc == NULL) {
        return;
    }
    testfw_log_info("case_id=%zu/%zu category=%s name=%s",
                    index,
                    total,
                    (tc->category != NULL) ? tc->category : "未分类",
                    (tc->name != NULL) ? tc->name : "unknown");
    testfw_log_info("purpose=%s", (tc->purpose != NULL) ? tc->purpose : "未提供");
    testfw_log_info("params=%s", (tc->params != NULL) ? tc->params : "未提供");
    testfw_log_info("expected=%s", (tc->expected != NULL) ? tc->expected : "0(PASS)");
}

/**
 * @brief 输出用例实际结果日志（ACTUAL 级别）。
 *
 * 设计目的：
 * - 与 testfw_log_case_meta 形成“期望/实际”双日志，便于问题定位。
 *
 * @param fmt printf 风格格式串
 * @param ... 可变参数
 */
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
        testfw_record_assert(0);
        testfw_log_error("断言失败：%s，期望=%d，实际=%d，位置=%s:%d", expr, expected, actual, file, line);
        return 0;
    }
    testfw_record_assert(1);
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
        testfw_record_assert(0);
        testfw_log_error("断言失败：%s，期望=%zu，实际=%zu，位置=%s:%d", expr, expected, actual, file, line);
        return 0;
    }
    testfw_record_assert(1);
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
        testfw_record_assert(0);
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
    testfw_record_assert(1);
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
        testfw_record_assert(0);
        testfw_log_error("断言失败：%s，位置=%s:%d", expr, file, line);
        return 0;
    }
    testfw_record_assert(1);
    return 1;
}
