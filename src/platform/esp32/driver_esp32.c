#include "../../include/platform_driver.h"
#include "../../include/platform_runtime.h"

int platform_esp32_apply_action(const float* action_values, size_t action_count) {
    DriverStub driver;
    int rc = 0;
    rc = driver_stub_init(&driver, DRIVER_TYPE_ESP32);
    if (rc != DRIVER_STATUS_OK) {
        return rc;
    }
    rc = driver_stub_apply(&driver, action_values, action_count);
    driver_stub_shutdown(&driver);
    return rc;
}
