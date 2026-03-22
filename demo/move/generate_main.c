/**
 * @file generate_main.c
 * @brief Move Demo 代码生成入口
 *
 * 6步流程步骤2：运行代码生成器
 *
 * 功能：
 * - 调用 profiler 生成网络结构的 .c 和 .h 文件
 * - 输出到相对路径 "demo/move/data"
 *
 * 生成的文件：
 * - move.c: 网络结构实现
 * - move.h: 网络接口
 * - network_spec.txt: 网络规格说明
 */

#include "profiler.h"

#include <stdio.h>

int main(void) {
    ProfilerGenerateRequest req;

    /* 设置生成请求 */
    req.network_name = "move";
    req.network_type = "mlp";
    req.output_dir = "demo/move/data";

    /* 调用 profiler 生成代码 */
    int rc = profiler_generate(&req);
    if (rc != 0) {
        fprintf(stderr, "profiler_generate failed: %d\n", rc);
        return 1;
    }

    printf("move spec generated\n");
    return 0;
}
