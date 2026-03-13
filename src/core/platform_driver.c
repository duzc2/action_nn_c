#include "../include/platform_driver.h"

#include <stdio.h>

/**
 * @brief 获取驱动类型对应的人类可读名称。
 *
 * @param type 驱动类型
 * @return const char* 名称字符串
 */
static const char* driver_type_name(DriverType type) {
    if (type == DRIVER_TYPE_PC) {
        return "PC";
    }
    if (type == DRIVER_TYPE_ESP32) {
        return "ESP32";
    }
    return "UNKNOWN";
}

/**
 * @brief 初始化驱动桩。
 *
 * @param driver 驱动桩对象
 * @param type   驱动类型
 * @return int   DriverStatus
 */
int driver_stub_init(DriverStub* driver, DriverType type) {
    if (driver == NULL) {
        return DRIVER_STATUS_INVALID_ARGUMENT;
    }
    driver->type = type;
    driver->is_initialized = 1;
    printf("[driver][%s] initialized stub\n", driver_type_name(type));
    return DRIVER_STATUS_OK;
}

/**
 * @brief 将动作向量发送到驱动桩。
 *
 * @param driver        驱动桩对象
 * @param actuator_vals 执行器输出数组
 * @param count         通道数量
 * @return int          DriverStatus
 */
int driver_stub_apply(const DriverStub* driver, const float* actuator_vals, size_t count) {
    size_t i = 0U;
    if (driver == NULL || actuator_vals == NULL || count == 0U) {
        return DRIVER_STATUS_INVALID_ARGUMENT;
    }
    if (driver->is_initialized == 0) {
        return DRIVER_STATUS_NOT_READY;
    }

    if (driver->type == DRIVER_TYPE_PC) {
        printf("[driver][PC] simulated keyboard/mouse:");
    } else if (driver->type == DRIVER_TYPE_ESP32) {
        printf("[driver][ESP32] simulated gpio/pwm:");
    } else {
        return DRIVER_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0U; i < count; ++i) {
        printf(" ch%zu=%.4f", i, (double)actuator_vals[i]);
    }
    printf("\n");
    return DRIVER_STATUS_OK;
}

/**
 * @brief 释放驱动桩。
 *
 * @param driver 驱动桩对象
 */
void driver_stub_shutdown(DriverStub* driver) {
    if (driver == NULL) {
        return;
    }
    if (driver->is_initialized != 0) {
        printf("[driver][%s] shutdown stub\n", driver_type_name(driver->type));
    }
    driver->is_initialized = 0;
}
