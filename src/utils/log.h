#ifndef ACTION_C_LOG_H
#define ACTION_C_LOG_H

#include <stdio.h>

#ifndef ACTION_C_LOG_LEVEL
#define ACTION_C_LOG_LEVEL 2
#endif

void _action_c_log(int level, const char* file, int line,
                   const char* fmt, ...);

#if ACTION_C_LOG_LEVEL >= 1
#define LOG_ERROR(fmt, ...) \
    _action_c_log(1, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define LOG_ERROR(fmt, ...) ((void)0)
#endif

#if ACTION_C_LOG_LEVEL >= 2
#define LOG_WARN(fmt, ...) \
    _action_c_log(2, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define LOG_WARN(fmt, ...) ((void)0)
#endif

#if ACTION_C_LOG_LEVEL >= 3
#define LOG_INFO(fmt, ...) \
    _action_c_log(3, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define LOG_INFO(fmt, ...) ((void)0)
#endif

/* Debug assertion: validates invariants in Debug, zero-cost removed in Release */
#ifndef NDEBUG
#define ASSERT(cond, err, msg) do { \
    if (!(cond)) { \
        LOG_ERROR("ASSERT FAILED: %s -- %s", #cond, msg); \
        return (err); \
    } \
} while(0)
#else
#define ASSERT(cond, err, msg) ((void)0)
#endif

#endif
