#ifndef PLATFORM_RUNTIME_H
#define PLATFORM_RUNTIME_H

#include <stddef.h>

int platform_pc_apply_action(const float* action_values, size_t action_count);
int platform_esp32_apply_action(const float* action_values, size_t action_count);

#endif
