#include "test_harness.h"
#include "nn/nn_weight_header.h"
#include <stdio.h>

TEST(weight_header_size_rounded) {
    /* Ensure struct size is predictable for binary compatibility.
     * magic(4) + abi(4) + nethash(8) + layouthash(8) + reserved[4](16) = 40 */
    size_t sz = sizeof(NNWeightHeader);
    ASSERT_TRUE(sz >= 4 + 4 + 8 + 8 + 16, "header has sufficient size for all fields");
}

TEST(weight_header_write_and_read_roundtrip) {
    NNWeightHeader hdr = {
        .magic = NN_WEIGHT_MAGIC,
        .abi_version = 1,
        .network_hash = 0xABCD1234,
        .layout_hash = 0x5678EF01,
        .reserved = {0, 0, 0, 0}
    };

    FILE* fp = tmpfile();
    ASSERT_NOT_NULL(fp, "tmpfile created");

    int rc = nn_weight_header_write(&hdr, fp);
    ASSERT_EQ_INT(0, rc, "write succeeds");

    rewind(fp);
    NNWeightHeader read_hdr;
    (void)memset(&read_hdr, 0, sizeof(read_hdr));
    rc = nn_weight_header_read(&read_hdr, 1, 0xABCD1234, 0x5678EF01, fp);
    ASSERT_EQ_INT(0, rc, "read succeeds");

    ASSERT_EQ_INT((int)NN_WEIGHT_MAGIC, (int)read_hdr.magic, "magic matches");
    ASSERT_EQ_INT(1, (int)read_hdr.abi_version, "abi matches");
    ASSERT_EQ_U64(0xABCD1234, read_hdr.network_hash, "network hash matches");
    ASSERT_EQ_U64(0x5678EF01, read_hdr.layout_hash, "layout hash matches");

    fclose(fp);
}

TEST(weight_header_rejects_bad_magic) {
    NNWeightHeader hdr = {
        .magic = 0xDEADBEEF,
        .abi_version = 1,
        .network_hash = 0, .layout_hash = 0,
        .reserved = {0}
    };

    FILE* fp = tmpfile();
    ASSERT_NOT_NULL(fp, "tmpfile created");
    (void)nn_weight_header_write(&hdr, fp);
    rewind(fp);

    NNWeightHeader read_hdr;
    (void)memset(&read_hdr, 0, sizeof(read_hdr));
    int rc = nn_weight_header_read(&read_hdr, 1, 0, 0, fp);
    ASSERT_TRUE(rc < 0, "read rejects bad magic");
    fclose(fp);
}

TEST(weight_header_rejects_abi_mismatch) {
    NNWeightHeader hdr = {
        .magic = NN_WEIGHT_MAGIC,
        .abi_version = 2,
        .network_hash = 0, .layout_hash = 0,
        .reserved = {0}
    };

    FILE* fp = tmpfile();
    ASSERT_NOT_NULL(fp, "tmpfile created");
    (void)nn_weight_header_write(&hdr, fp);
    rewind(fp);

    NNWeightHeader read_hdr;
    (void)memset(&read_hdr, 0, sizeof(read_hdr));
    int rc = nn_weight_header_read(&read_hdr, 1, 0, 0, fp);
    ASSERT_TRUE(rc < 0, "read rejects ABI mismatch (file=2, expected=1)");
    fclose(fp);
}

TEST(weight_header_rejects_hash_mismatch) {
    NNWeightHeader hdr = {
        .magic = NN_WEIGHT_MAGIC,
        .abi_version = 1,
        .network_hash = 0x1111111111111111ULL,
        .layout_hash = 0x2222222222222222ULL,
        .reserved = {0}
    };

    FILE* fp = tmpfile();
    ASSERT_NOT_NULL(fp, "tmpfile created");
    (void)nn_weight_header_write(&hdr, fp);
    rewind(fp);

    NNWeightHeader read_hdr;
    (void)memset(&read_hdr, 0, sizeof(read_hdr));
    int rc = nn_weight_header_read(&read_hdr, 1, 0x999, 0x2222222222222222ULL, fp);
    ASSERT_TRUE(rc < 0, "read rejects network hash mismatch");
    fclose(fp);
}

int main(void) {
    RUN_TEST(weight_header_size_rounded);
    RUN_TEST(weight_header_write_and_read_roundtrip);
    RUN_TEST(weight_header_rejects_bad_magic);
    RUN_TEST(weight_header_rejects_abi_mismatch);
    RUN_TEST(weight_header_rejects_hash_mismatch);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
