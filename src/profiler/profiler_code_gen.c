/**
 * @file profiler_code_gen.c
 * @brief profiler 代码生成器注册表实现
 *
 * 具体初始化由 CMake 生成的 autogen 文件完成。
 */

#include "profiler_code_gen.h"

#include <string.h>
#include <stdlib.h>

#define MAX_ENTRIES 16

static ProfilerCodeGenEntry g_entries[MAX_ENTRIES];
static size_t g_entry_count = 0;

int profiler_code_gen_register(const char* type_name, ProfilerCodeGenEntry entry) {
    if (g_entry_count >= MAX_ENTRIES) {
        return -1;
    }
    entry.type_name = type_name;
    g_entries[g_entry_count++] = entry;
    return 0;
}

int profiler_code_gen_get(const char* type_name, ProfilerCodeGenEntry* out_entry) {
    for (size_t i = 0; i < g_entry_count; i++) {
        if (strcmp(g_entries[i].type_name, type_name) == 0) {
            *out_entry = g_entries[i];
            return 0;
        }
    }
    return -1;
}
