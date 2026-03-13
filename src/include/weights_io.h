#ifndef WEIGHTS_IO_H
#define WEIGHTS_IO_H

#include <stddef.h>

/**
 * @brief 权重 IO 模块错误码定义。
 */
typedef enum WeightsIoStatus {
    WEIGHTS_IO_STATUS_OK = 0,
    WEIGHTS_IO_STATUS_INVALID_ARGUMENT = -1,
    WEIGHTS_IO_STATUS_IO_ERROR = -2,
    WEIGHTS_IO_STATUS_FORMAT_ERROR = -3,
    WEIGHTS_IO_STATUS_OOM = -4
} WeightsIoStatus;

/**
 * @brief 保存 float 权重数组为二进制文件。
 *
 * @param file_path  输出文件路径
 * @param weights    权重数据指针
 * @param count      权重元素数量
 * @return int       WeightsIoStatus
 */
int weights_save_binary(const char* file_path, const float* weights, size_t count);

/**
 * @brief 从二进制文件加载 float 权重数组。
 *
 * @param file_path   输入文件路径
 * @param out_weights 输出权重指针（调用方使用 free 释放）
 * @param out_count   输出权重数量
 * @return int        WeightsIoStatus
 */
int weights_load_binary(const char* file_path, float** out_weights, size_t* out_count);

/**
 * @brief 将权重导出为可直接编译的 weights.c 源码文件。
 *
 * 导出内容：
 * - const size_t <symbol>_count
 * - const float <symbol>[]
 *
 * @param file_path  输出 C 源文件路径
 * @param symbol     导出符号前缀（必须是合法 C 标识符）
 * @param weights    权重数据指针
 * @param count      权重元素数量
 * @return int       WeightsIoStatus
 */
int weights_export_c_source(const char* file_path,
                            const char* symbol,
                            const float* weights,
                            size_t count);

#endif /* WEIGHTS_IO_H */
