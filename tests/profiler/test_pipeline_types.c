#include "test_harness.h"
#include "profiler/prof_pipeline.h"

TEST(generated_files_init_all_null) {
    GeneratedFiles f;
    generated_files_init(&f);
    ASSERT_NULL(f.metadata_c, "metadata_c is NULL after init");
    ASSERT_NULL(f.infer_c, "infer_c is NULL after init");
    ASSERT_NULL(f.train_h, "train_h is NULL after init");
    ASSERT_NULL(f.tokenizer_c, "tokenizer_c is NULL after init");
    generated_files_free(&f);
}

TEST(generated_files_free_after_init_is_safe) {
    GeneratedFiles f;
    generated_files_init(&f);
    generated_files_free(&f);
    /* Double-free must be safe (memset(0) after free clears pointers). */
    generated_files_free(&f);
    ASSERT_TRUE(1, "free after init does not crash");
}

TEST(network_hashes_struct_sizes) {
    NetworkHashes h;
    h.network_hash = 0xABCD1234;
    h.layout_hash  = 0x5678EF01;
    h.abi_version  = 1;
    ASSERT_EQ_U64(0xABCD1234, h.network_hash, "network_hash field works");
    ASSERT_EQ_U64(0x5678EF01, h.layout_hash, "layout_hash field works");
    ASSERT_EQ_INT(1, (int)h.abi_version, "abi_version field works");
}

TEST(pipeline_types_composable) {
    /* Verify that NetworkHashes and GeneratedFiles can coexist in one struct. */
    typedef struct {
        NetworkHashes  hashes;
        GeneratedFiles files;
    } PipelineState;
    PipelineState st;
    (void)memset(&st, 0, sizeof(st));
    st.hashes.abi_version = 2;
    generated_files_free(&st.files);
    ASSERT_EQ_INT(2, (int)st.hashes.abi_version, "composable in PipelineState");
}

int main(void) {
    RUN_TEST(generated_files_init_all_null);
    RUN_TEST(generated_files_free_after_init_is_safe);
    RUN_TEST(network_hashes_struct_sizes);
    RUN_TEST(pipeline_types_composable);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
