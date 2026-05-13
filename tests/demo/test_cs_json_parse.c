#include "test_harness.h"
#include <string.h>
#include <stdio.h>

/* Standalone copy of cs_json_find_key for testing.
   The actual implementation lives in demo/cs/tools/cs_tool_common.c as
   cs_tool_find_key. This test validates the concept. */

static const char* cs_json_find_key(const char* cursor, const char* key) {
    char search[64];
    int n = snprintf(search, sizeof(search), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(search)) return NULL;
    const char* found = strstr(cursor, search);
    if (!found) return NULL;
    return found + strlen(search);
}

TEST(finds_key_in_json) {
    const char* json = "{\"network_type\": \"mlp\", \"layers\": 3}";

    const char* val = cs_json_find_key(json, "network_type");
    ASSERT_NOT_NULL(val, "network_type found");
    ASSERT_TRUE(strstr(val, "mlp") != NULL, "value contains mlp");

    const char* val2 = cs_json_find_key(json, "layers");
    ASSERT_NOT_NULL(val2, "layers found");
    ASSERT_TRUE(strstr(val2, "3") != NULL, "value contains 3");
}

TEST(returns_null_for_missing_key) {
    const char* json = "{\"network_type\": \"mlp\"}";
    const char* val = cs_json_find_key(json, "nonexistent");
    ASSERT_NULL(val, "missing key returns NULL");
}

TEST(handles_empty_input) {
    const char* val = cs_json_find_key("", "key");
    ASSERT_NULL(val, "empty input returns NULL");
}

TEST(handles_partial_key_match) {
    /* key "ten" should not match "tensor" */
    const char* json = "{\"tensor\": 5}";
    const char* val = cs_json_find_key(json, "ten");
    ASSERT_NULL(val, "partial key match not found");
}

int main(void) {
    RUN_TEST(finds_key_in_json);
    RUN_TEST(returns_null_for_missing_key);
    RUN_TEST(handles_empty_input);
    RUN_TEST(handles_partial_key_match);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
