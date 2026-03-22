/**
 * @file profiler.c
 * @brief 代码生成器实现
 *
 * 功能：
 * - 读取用户传入的网络规格（网络名、网络类型）
 * - 验证网络类型是否已注册
 * - 生成网络结构的 C 代码文件（.c 和 .h）
 *
 * 注意：
 * - 生成的文件不包含权重，仅包含网络结构定义
 * - 生成的文件不包含训练数据，训练数据由用户自行准备
 * - 权重由训练步骤生成
 */

#include "profiler.h"
#include "nn_infer_registry.h"
#include "profiler_code_gen.h"
#include "infer_runtime.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

/**
 * @brief 确保目录存在，不存在则创建
 * @param path 目录路径
 * @return 0 成功，-1 失败
 */
static int ensure_dir(const char* path) {
    int rc = MKDIR(path);
    if (rc == 0) {
        return 0;
    }
    if (errno == EEXIST) {
        return 0;
    }
    return -1;
}

/**
 * @brief 写入内容到文件
 * @param path 文件路径
 * @param content 文件内容
 * @return 0 成功，-1 失败
 */
static int write_file(const char* path, const char* content) {
    FILE* fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs(content, fp);
    fclose(fp);
    return 0;
}

/**
 * @brief 默认的 IO 配置
 *
 * 默认配置适用于 move demo：
 * - 输入：x, y, command (3个)
 * - 输出：out_x, out_y (2个)
 */
static void set_default_io_names(ProfIONames* io_names) {
    io_names->input_names = "x,y,cmd";
    io_names->input_count = 3;
    io_names->output_names = "out_x,out_y";
    io_names->output_count = 2;
}

/**
 * @brief 简化的生成接口
 *
 * 使用默认的 IO 配置调用完整接口。
 * 适用于 IO 固定的简单场景。
 *
 * @param request 生成请求，包含网络名、类型、输出目录
 * @return 0 成功，非0 失败
 */
int profiler_generate(const ProfilerGenerateRequest* request) {
    ProfIONames io_names;
    set_default_io_names(&io_names);
    return profiler_generate_with_io(
        request->network_name,
        request->network_type,
        request->output_dir,
        &io_names
    );
}

/**
 * @brief 写入网络实现 C 文件
 *
 * 通过注册机制调用网络类型的代码生成器。
 *
 * @param name 网络名称
 * @param type 网络类型
 * @param fp 文件指针
 * @return 0 成功
 */
static int write_infer_c(const char* name, const char* type, FILE* fp) {
    ProfilerCodeGenEntry entry;
    int rc = profiler_code_gen_get(type, &entry);
    if (rc != 0 || entry.generate_c == NULL) {
        return -10;
    }
    return entry.generate_c(name, fp);
}

/**
 * @brief 写入网络接口头文件
 *
 * 通过注册机制获取网络类型的代码生成器。
 *
 * @param name 网络名称
 * @param type 网络类型
 * @param fp 文件指针
 * @return 0 成功
 */
static int write_header(const char* name, const char* type, FILE* fp) {
    ProfilerCodeGenEntry entry;
    int rc = profiler_code_gen_get(type, &entry);
    if (rc != 0) {
        return -10;
    }
    if (entry.generate_h != NULL) {
        return entry.generate_h(name, fp);
    }
    return 0;
}

/**
 * @brief 完整的生成接口
 *
 * 执行流程：
 * 1. 参数校验
 * 2. 初始化网络注册表
 * 3. 验证网络类型是否已注册
 * 4. 创建输出目录
 * 5. 生成 network_spec.txt（网络规格说明）
 * 6. 生成 xxx.h（接口头文件）
 * 7. 生成 xxx.c（实现代码）
 *
 * @param network_name 网络名称，用于生成文件名和函数名
 * @param network_type 网络类型，用于查找对应的网络实现
 * @param output_dir 输出目录
 * @param io_names 输入输出定义
 * @return 0 成功，-1 参数为空，-2 目录创建失败，-5 类型未注册，-6 注册表初始化失败
 */
int profiler_generate_with_io(
    const char* network_name,
    const char* network_type,
    const char* output_dir,
    const ProfIONames* io_names
) {
    char file_path[512];
    FILE* fp = NULL;
    int rc;

    /* 参数校验：所有参数不能为空 */
    if (network_name == NULL || network_type == NULL || output_dir == NULL || io_names == NULL) {
        return -1;
    }

    /* 初始化网络注册表：确保注册的网络类型可用 */
    rc = nn_infer_registry_bootstrap();
    if (rc != 0) {
        return -6;
    }

    /* 初始化代码生成器注册表 */
    rc = profiler_code_gen_bootstrap();
    if (rc != 0) {
        return -7;
    }

    /* 验证网络类型是否已注册 */
    rc = nn_infer_registry_is_registered(network_type);
    if (!rc) {
        return -5;
    }

    /* 创建输出目录 */
    rc = ensure_dir(output_dir);
    if (rc != 0) {
        return -2;
    }

    /* 生成 network_spec.txt */
    snprintf(file_path, sizeof(file_path), "%s/network_spec.txt", output_dir);
    fp = fopen(file_path, "w");
    if (fp == NULL) {
        return -4;
    }
    fprintf(fp, "network=%s\n", network_name);
    fprintf(fp, "network_type=%s\n", network_type);
    fprintf(fp, "standard=C99\n");
    fprintf(fp, "generated=1\n");
    fclose(fp);

    /* 生成 xxx.h */
    snprintf(file_path, sizeof(file_path), "%s/%s.h", output_dir, network_name);
    fp = fopen(file_path, "w");
    if (fp == NULL) {
        return -11;
    }
    write_header(network_name, network_type, fp);
    fclose(fp);

    /* 生成 xxx.c */
    snprintf(file_path, sizeof(file_path), "%s/%s.c", output_dir, network_name);
    fp = fopen(file_path, "w");
    if (fp == NULL) {
        return -21;
    }
    write_infer_c(network_name, network_type, fp);
    fclose(fp);

    return 0;
}
