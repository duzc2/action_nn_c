/**
 * @file profiler_code_gen.h
 * @brief profiler 代码生成器注册接口
 *
 * 每个网络类型注册自己的代码生成函数。
 * profiler 通过注册机制调用，不依赖具体网络类型实现细节。
 *
 * 具体初始化由 CMake 生成的 autogen 文件完成。
 */

#ifndef PROFILER_CODE_GEN_H
#define PROFILER_CODE_GEN_H

#include <stdio.h>

typedef struct {
    const char* type_name;
    int (*generate_c)(const char* name, FILE* fp);
    int (*generate_h)(const char* name, FILE* fp);
} ProfilerCodeGenEntry;

int profiler_code_gen_register(const char* type_name, ProfilerCodeGenEntry entry);
int profiler_code_gen_get(const char* type_name, ProfilerCodeGenEntry* out_entry);
int profiler_code_gen_bootstrap(void);

#endif
