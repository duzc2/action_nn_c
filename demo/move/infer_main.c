/**
 * @file infer_main.c
 * @brief Move Demo 推理入口
 *
 * 6步流程步骤6：运行推理
 *
 * 用户输入：
 * - 起始坐标 (x, y)
 * - 命令序列：0=上, 1=下, 2=左, 3=右, 4=停止
 *
 * 输出：每一步执行后的坐标
 */

#include "move.h"

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief 简化的上下文结构
 *
 * 与生成的 move.c 中的 moveContext 相同。
 * 直接使用 void* 也可以，这里为了清晰定义为结构体。
 */
typedef struct {
    int input_x;
    int input_y;
    int command;
    int output_x;
    int output_y;
} MoveContext;

/**
 * @brief 主函数
 *
 * 流程：
 * 1. 读取起始坐标
 * 2. 创建网络上下文
 * 3. 循环读取命令并执行推理
 * 4. 销毁网络上下文
 */
int main(void) {
    int x = 0;
    int y = 0;
    int cmd = 0;
    void* net_ctx = NULL;

    printf("input startX startY first, then commands: 0=up 1=down 2=left 3=right 4=stop\n");

    /* 读取起始坐标 */
    if (scanf("%d %d", &x, &y) != 2) {
        return 1;
    }

    /* 创建网络上下文 */
    net_ctx = move_create();
    if (net_ctx == NULL) {
        fprintf(stderr, "create network failed\n");
        return 1;
    }

    /* 循环处理命令 */
    while (scanf("%d", &cmd) == 1) {
        if (cmd == 4) {
            break;
        }

        /* 设置输入 */
        MoveContext* mc = (MoveContext*)net_ctx;
        mc->input_x = x;
        mc->input_y = y;
        mc->command = cmd;

        /* 执行推理 */
        if (move_forward(net_ctx) != 0) {
            printf("forward failed\n");
            continue;
        }

        /* 获取输出 */
        x = mc->output_x;
        y = mc->output_y;
        printf("x=%d y=%d\n", x, y);
    }

    /* 销毁网络上下文 */
    move_destroy(net_ctx);

    printf("final_x=%d final_y=%d\n", x, y);
    return 0;
}
