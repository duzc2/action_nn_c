#include "log.h"
#include <stdarg.h>

void _action_c_log(int level, const char* file, int line,
                   const char* fmt, ...) {
    static const char* level_str[] = {
        "", "[ERROR]", "[WARN]", "[INFO]", "[DEBUG]"
    };
    unsigned int safe_level = (unsigned int)level;
    if (safe_level > 4U) safe_level = 0U;
    fprintf(stderr, "%s %s:%d: ",
            level_str[safe_level], file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}
