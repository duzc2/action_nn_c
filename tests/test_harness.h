#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* MSVC warnings suppressed for test harness macros */
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4127) /* conditional expression is constant */
#endif

static int _tests_run   = 0;
static int _tests_pass  = 0;
static int _tests_fail  = 0;
static const char* _current_test = NULL;

/* 定义测试函数 */
#define TEST(name) \
    static void _test_##name(void)

/* 注册并运行一个测试 */
#define RUN_TEST(name) do { \
    _current_test = #name; \
    _tests_run++; \
    _test_##name(); \
} while(0)

/* 断言宏 */
#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s\n", \
                __FILE__, __LINE__, _current_test, msg); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_EQ_INT(expected, actual, msg) do { \
    int _e = (expected), _a = (actual); \
    if (_e != _a) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected %d, got %d)\n", \
                __FILE__, __LINE__, _current_test, msg, _e, _a); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_EQ_SIZE(expected, actual, msg) do { \
    size_t _e = (expected), _a = (actual); \
    if (_e != _a) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected %zu, got %zu)\n", \
                __FILE__, __LINE__, _current_test, msg, _e, _a); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_EQ_U64(expected, actual, msg) do { \
    uint64_t _e = (expected), _a = (actual); \
    if (_e != _a) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected %llu, got %llu)\n", \
                __FILE__, __LINE__, _current_test, msg, \
                (unsigned long long)_e, (unsigned long long)_a); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_STREQ(expected, actual, msg) do { \
    if (strcmp((expected), (actual)) != 0) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected '%s', got '%s')\n", \
                __FILE__, __LINE__, _current_test, msg, (expected), (actual)); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_NULL(ptr, msg) do { \
    if ((ptr) != NULL) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected NULL, got %p)\n", \
                __FILE__, __LINE__, _current_test, msg, (const void*)(ptr)); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_NOT_NULL(ptr, msg) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected non-NULL)\n", \
                __FILE__, __LINE__, _current_test, msg); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#define ASSERT_FLOAT_EQ(expected, actual, epsilon, msg) do { \
    float _e = (expected), _a = (actual), _eps = (epsilon); \
    float _diff = (_e > _a) ? (_e - _a) : (_a - _e); \
    if (_diff > _eps) { \
        fprintf(stderr, "  FAIL %s:%d: %s -- %s (expected %f, got %f, diff %f)\n", \
                __FILE__, __LINE__, _current_test, msg, _e, _a, _diff); \
        _tests_fail++; return; \
    } \
    _tests_pass++; \
} while(0)

#endif /* TEST_HARNESS_H */
