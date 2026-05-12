#include "log.h"
#include <stdarg.h>

void _action_c_log(int level, const char* file, int line,
                   const char* fmt, ...) {
    static const char* level_str[] = {
        "", "[ERROR]", "[WARN]", "[INFO]", "[DEBUG]"
    };
    fprintf(stderr, "%s %s:%d: ",
            level_str[level], file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}
