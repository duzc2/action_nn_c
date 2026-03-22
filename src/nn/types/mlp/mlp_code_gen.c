/**
 * @file mlp_code_gen.c
 * @brief MLP 网络代码生成器
 *
 * 通过注册机制提供给 profiler，不修改 profiler 主流程。
 */

#include "mlp_infer_ops.h"
#include "profiler_code_gen.h"

#include <stdio.h>
#include <string.h>

static int mlp_generate_c(const char* name, FILE* fp) {
    fprintf(fp, "#include \"%s.h\"\n", name);
    fprintf(fp, "#include \"infer_runtime.h\"\n");
    fprintf(fp, "#include \"mlp_infer_ops.h\"\n");
    fprintf(fp, "#include <stdlib.h>\n");
    fprintf(fp, "\n");

    fprintf(fp, "typedef struct {\n");
    fprintf(fp, "    MLPInferContext mlp_ctx;\n");
    fprintf(fp, "} %sContext;\n\n", name);

    fprintf(fp, "void* %s_create(void) {\n", name);
    fprintf(fp, "    %sContext* ctx = (%sContext*)malloc(sizeof(%sContext));\n", name, name, name);
    fprintf(fp, "    if (ctx == NULL) return NULL;\n");
    fprintf(fp, "    ctx->mlp_ctx.input_x = 0;\n");
    fprintf(fp, "    ctx->mlp_ctx.input_y = 0;\n");
    fprintf(fp, "    ctx->mlp_ctx.command = 0;\n");
    fprintf(fp, "    ctx->mlp_ctx.output_x = 0;\n");
    fprintf(fp, "    ctx->mlp_ctx.output_y = 0;\n");
    fprintf(fp, "    return ctx;\n");
    fprintf(fp, "}\n\n");

    fprintf(fp, "void %s_destroy(void* c) {\n", name);
    fprintf(fp, "    if (c != NULL) free(c);\n");
    fprintf(fp, "}\n\n");

    fprintf(fp, "int %s_forward(void* c) {\n", name);
    fprintf(fp, "    %sContext* ctx = (%sContext*)c;\n", name, name, name);
    fprintf(fp, "    if (ctx == NULL) return -1;\n");
    fprintf(fp, "    return nn_mlp_infer_step(&ctx->mlp_ctx);\n");
    fprintf(fp, "}\n\n");

    return 0;
}

static int mlp_generate_h(const char* name, FILE* fp) {
    fprintf(fp, "#ifndef %s_H\n", name);
    fprintf(fp, "#define %s_H\n\n", name);
    fprintf(fp, "void* %s_create(void);\n", name);
    fprintf(fp, "void %s_destroy(void* ctx);\n", name);
    fprintf(fp, "int %s_forward(void* ctx);\n\n", name);
    fprintf(fp, "#endif\n");
    return 0;
}

void profiler_code_gen_init_mlp(void) {
    ProfilerCodeGenEntry entry = {
        "mlp",
        mlp_generate_c,
        mlp_generate_h
    };
    profiler_code_gen_register("mlp", entry);
}
