/**
 * @file profiler.c
 * @brief 网络代码生成器
 *
 * 功能：
 * - 读取用户传入的网络规格
 * - 验证网络类型是否已注册
 * - 生成网络结构代码和接口
 *
 * 生成的文件：
 * - network.c : 网络结构实现（create/destroy/forward/load/save）
 * - network.h : 网络接口声明
 * - network_spec.txt : 网络描述
 *
 * 注意：
 * - 权重数据（weights.txt）由训练时 save_weights() 导出
 * - 权重常量（weights.c）由训练后导出工具生成
 */

#include "profiler.h"
#include "nn_infer_registry.h"
#include "nn_train_registry.h"
#include "infer_runtime.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

static int ensure_dir(const char* path) {
    int rc = MKDIR(path);
    if (rc == 0) return 0;
    if (errno == EEXIST) return 0;
    return -1;
}

static void capitalize_first(const char* src, char* dest, size_t dest_size) {
    size_t i;
    for (i = 0; i < dest_size - 1 && src[i] != '\0'; i++) {
        if (i == 0) {
            dest[i] = (char)toupper(src[i]);
        } else {
            dest[i] = src[i];
        }
    }
    dest[i] = '\0';
}

int profiler_generate(const ProfilerGenerateRequest* request) {
    return profiler_generate_with_io(
        request->network_name,
        request->network_type,
        request->output_dir,
        NULL
    );
}

int profiler_generate_with_io(
    const char* network_name,
    const char* network_type,
    const char* output_dir,
    const ProfIONames* io_names
) {
    char file_path[512];
    FILE* fp = NULL;
    int rc;

    if (network_name == NULL || network_type == NULL || output_dir == NULL) {
        return -1;
    }

    rc = nn_infer_registry_bootstrap();
    if (rc != 0) return -6;

    rc = nn_train_registry_bootstrap();
    if (rc != 0) return -8;

    rc = nn_infer_registry_is_registered(network_type);
    if (!rc) return -5;

    rc = nn_train_registry_is_registered(network_type);
    if (!rc) return -9;

    rc = ensure_dir(output_dir);
    if (rc != 0) return -2;

    /* 生成 network_spec.txt */
    snprintf(file_path, sizeof(file_path), "%s/network_spec.txt", output_dir);
    fp = fopen(file_path, "w");
    if (fp == NULL) return -4;
    fprintf(fp, "network=%s\n", network_name);
    fprintf(fp, "type=%s\n", network_type);
    fprintf(fp, "standard=C99\n");
    fprintf(fp, "generated=1\n");
    fclose(fp);

    /* 生成 network.c - 网络结构代码（包含 load/save 函数） */
    snprintf(file_path, sizeof(file_path), "%s/%s.c", output_dir, network_name);
    fp = fopen(file_path, "w");
    if (fp == NULL) return -11;
    char type_cap[64];
    capitalize_first(network_type, type_cap, sizeof(type_cap));
    fprintf(fp, "#include \"%s.h\"\n", network_name);
    fprintf(fp, "#include \"infer_runtime.h\"\n");
    fprintf(fp, "#include \"%s_infer_ops.h\"\n", network_type);
    fprintf(fp, "#include <stdlib.h>\n");
    fprintf(fp, "#include <string.h>\n");
    fprintf(fp, "#include <stdio.h>\n");
    fprintf(fp, "\n");

    /* 上下文结构体 */
    fprintf(fp, "typedef struct {\n");
    fprintf(fp, "    %sInferContext ctx;\n", type_cap);
    fprintf(fp, "} %sContext;\n\n", network_name);

    /* 创建函数 */
    fprintf(fp, "void* %s_create(void) {\n", network_name);
    fprintf(fp, "    %sContext* c = (%sContext*)malloc(sizeof(%sContext));\n", network_name, network_name, network_name);
    fprintf(fp, "    if (c == NULL) return NULL;\n");
    fprintf(fp, "    memset(&c->ctx, 0, sizeof(c->ctx));\n");
    fprintf(fp, "    return c;\n");
    fprintf(fp, "}\n\n");

    /* 销毁函数 */
    fprintf(fp, "void %s_destroy(void* ctx) {\n", network_name);
    fprintf(fp, "    if (ctx != NULL) free(ctx);\n");
    fprintf(fp, "}\n\n");

    /* 前向函数 */
    fprintf(fp, "int %s_forward(void* ctx) {\n", network_name);
    fprintf(fp, "    if (ctx == NULL) return -1;\n");
    fprintf(fp, "    %sContext* c = (%sContext*)ctx;\n", network_name, network_name);
    fprintf(fp, "    return nn_%s_infer_step(&c->ctx);\n", network_type);
    fprintf(fp, "}\n\n");

    /* 加载权重函数（从文件） */
    fprintf(fp, "int %s_load_weights(void* ctx, const char* path) {\n", network_name);
    fprintf(fp, "    if (ctx == NULL || path == NULL) return -1;\n");
    fprintf(fp, "    %sContext* c = (%sContext*)ctx;\n", network_name, network_name);
    fprintf(fp, "    FILE* fp = fopen(path, \"r\");\n");
    fprintf(fp, "    if (fp == NULL) return -2;\n");
    fprintf(fp, "    int loaded = nn_%s_load_weights(&c->ctx, fp);\n", network_type);
    fprintf(fp, "    fclose(fp);\n");
    fprintf(fp, "    return loaded ? 0 : -3;\n");
    fprintf(fp, "}\n\n");

    /* 保存权重函数（到文件） */
    fprintf(fp, "int %s_save_weights(void* ctx, const char* path) {\n", network_name);
    fprintf(fp, "    if (ctx == NULL || path == NULL) return -1;\n");
    fprintf(fp, "    %sContext* c = (%sContext*)ctx;\n", network_name, network_name);
    fprintf(fp, "    FILE* fp = fopen(path, \"w\");\n");
    fprintf(fp, "    if (fp == NULL) return -2;\n");
    fprintf(fp, "    int saved = nn_%s_save_weights(&c->ctx, fp);\n", network_type);
    fprintf(fp, "    fclose(fp);\n");
    fprintf(fp, "    return saved ? 0 : -3;\n");
    fprintf(fp, "}\n\n");

    fclose(fp);

    /* 生成 network.h */
    snprintf(file_path, sizeof(file_path), "%s/%s.h", output_dir, network_name);
    fp = fopen(file_path, "w");
    if (fp == NULL) return -13;
    fprintf(fp, "#ifndef %s_H\n", network_name);
    fprintf(fp, "#define %s_H\n\n", network_name);
    fprintf(fp, "/* Network: %s, Type: %s */\n", network_name, network_type);
    fprintf(fp, "/* Generated by profiler */\n\n");
    fprintf(fp, "void* %s_create(void);\n", network_name);
    fprintf(fp, "void %s_destroy(void* ctx);\n", network_name);
    fprintf(fp, "int %s_forward(void* ctx);\n", network_name);
    fprintf(fp, "int %s_load_weights(void* ctx, const char* path);\n", network_name);
    fprintf(fp, "int %s_save_weights(void* ctx, const char* path);\n\n", network_name);
    fprintf(fp, "#endif\n");
    fclose(fp);

    return 0;
}
