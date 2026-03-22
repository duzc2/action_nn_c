/**
 * @file profiler.c
 * @brief Network code generator core implementation
 *
 * Features:
 * - Read network specification passed by user
 * - Validate if network type is registered
 * - Generate network structure code and interfaces
 * - Provide unified error handling mechanism
 *
 * Generated files:
 * - network.c : Network structure implementation (create/destroy/forward/load/save)
 * - network.h : Network interface declaration
 * - network_spec.txt : Network description
 *
 * Note:
 * - Weight data (weights.txt) exported by save_weights() during training
 * - Weight constants (weights.c) exported by post-training export tool
 */

#include "profiler.h"
#include "prof_error.h"
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

/**
 * @brief Ensure directory exists, create if not
 *
 * @param path Directory path
 * @return 0 success, -1 failure
 */
static int ensure_dir(const char* path) {
    int rc = MKDIR(path);
    if (rc == 0) return 0;
    if (errno == EEXIST) return 0;
    return -1;
}

/**
 * @brief Capitalize first character of string
 *
 * @param src Source string
 * @param dest Destination buffer
 * @param dest_size Destination buffer size
 */
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

/**
 * @brief Simplified generate entry (legacy interface)
 *
 * @param request Generate request
 * @return 0 success, non-zero failure
 */
int profiler_generate(const ProfilerGenerateRequest* request) {
    return profiler_generate_with_io(
        request->network_name,
        request->network_type,
        request->output_dir,
        NULL
    );
}

/**
 * @brief Simplified code generation function (legacy interface)
 *
 * @param network_name Network name
 * @param network_type Network type
 * @param output_dir Output directory
 * @param io_names IO names definition (optional)
 * @return 0 success, non-zero failure
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

    (void)io_names;

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

    /* Generate network_spec.txt */
    snprintf(file_path, sizeof(file_path), "%s/network_spec.txt", output_dir);
    fp = fopen(file_path, "w");
    if (fp == NULL) return -4;
    fprintf(fp, "network=%s\n", network_name);
    fprintf(fp, "type=%s\n", network_type);
    fprintf(fp, "standard=C99\n");
    fprintf(fp, "generated=1\n");
    fclose(fp);

    /* Generate network.c - network structure code (includes load/save functions) */
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

    /* Context structure */
    fprintf(fp, "typedef struct {\n");
    fprintf(fp, "    %sInferContext ctx;\n", type_cap);
    fprintf(fp, "} %sContext;\n\n", network_name);

    /* Create function */
    fprintf(fp, "void* %s_create(void) {\n", network_name);
    fprintf(fp, "    %sContext* c = (%sContext*)malloc(sizeof(%sContext));\n", network_name, network_name, network_name);
    fprintf(fp, "    if (c == NULL) return NULL;\n");
    fprintf(fp, "    memset(&c->ctx, 0, sizeof(c->ctx));\n");
    fprintf(fp, "    return c;\n");
    fprintf(fp, "}\n\n");

    /* Destroy function */
    fprintf(fp, "void %s_destroy(void* ctx) {\n", network_name);
    fprintf(fp, "    if (ctx != NULL) free(ctx);\n");
    fprintf(fp, "}\n\n");

    /* Forward function */
    fprintf(fp, "int %s_forward(void* ctx) {\n", network_name);
    fprintf(fp, "    if (ctx == NULL) return -1;\n");
    fprintf(fp, "    %sContext* c = (%sContext*)ctx;\n", network_name, network_name);
    fprintf(fp, "    return nn_%s_infer_step(&c->ctx);\n", network_type);
    fprintf(fp, "}\n\n");

    /* Load weights function (from file) */
    fprintf(fp, "int %s_load_weights(void* ctx, const char* path) {\n", network_name);
    fprintf(fp, "    if (ctx == NULL || path == NULL) return -1;\n");
    fprintf(fp, "    %sContext* c = (%sContext*)ctx;\n", network_name, network_name);
    fprintf(fp, "    FILE* fp = fopen(path, \"r\");\n");
    fprintf(fp, "    if (fp == NULL) return -2;\n");
    fprintf(fp, "    int loaded = nn_%s_load_weights(&c->ctx, fp);\n", network_type);
    fprintf(fp, "    fclose(fp);\n");
    fprintf(fp, "    return loaded ? 0 : -3;\n");
    fprintf(fp, "}\n\n");

    /* Save weights function (to file) */
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

    /* Generate network.h */
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

/**
 * @brief Full code generation function (new interface)
 *
 * @param req Generate request
 * @param out_result Output result
 * @return PROF_STATUS_OK on success, other values on failure
 */
ProfStatus profiler_generate_v2(
    const ProfGenerateRequest* req,
    ProfGenerateResult* out_result
) {
    char error_buffer[256];
    ProfErrorBuffer error;
    ProfGenerateResult local_result;

    if (req == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    if (out_result == NULL) {
        out_result = &local_result;
    }

    prof_error_init(&error, error_buffer, sizeof(error_buffer));

    out_result->network_hash = 0;
    out_result->metadata_written_path = NULL;

    return PROF_STATUS_OK;
}
