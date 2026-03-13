#ifndef PLATFORM_DRIVER_H
#define PLATFORM_DRIVER_H

#include <stddef.h>

/**
 * @brief 驱动错误码。
 */
typedef enum DriverStatus {
    DRIVER_STATUS_OK = 0,
    DRIVER_STATUS_INVALID_ARGUMENT = -1,
    DRIVER_STATUS_NOT_READY = -2
} DriverStatus;

/**
 * @brief 驱动类型，当前仅实现 PC/ESP32 驱动桩。
 */
typedef enum DriverType {
    DRIVER_TYPE_PC = 0,
    DRIVER_TYPE_ESP32 = 1
} DriverType;

/**
 * @brief 驱动桩实例，保存基本状态与目标平台。
 */
typedef struct DriverStub {
    DriverType type;
    int is_initialized;
} DriverStub;

/**
 * @brief 初始化驱动桩。
 *
 * @param driver 驱动桩对象
 * @param type   驱动类型
 * @return int   DriverStatus
 */
int driver_stub_init(DriverStub* driver, DriverType type);

/**
 * @brief 将动作向量发送到驱动桩。
 *
 * 行为说明：
 * - PC 驱动桩：打印“模拟键鼠输出”日志。
 * - ESP32 驱动桩：打印“GPIO/PWM 输出”日志。
 *
 * @param driver        驱动桩对象
 * @param actuator_vals 执行器输出数组
 * @param count         通道数量
 * @return int          DriverStatus
 */
int driver_stub_apply(const DriverStub* driver, const float* actuator_vals, size_t count);

/**
 * @brief 释放驱动桩。
 *
 * @param driver 驱动桩对象
 */
void driver_stub_shutdown(DriverStub* driver);

#endif /* PLATFORM_DRIVER_H */
