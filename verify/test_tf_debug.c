#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/nn/types/transformer/transformer_config.h"
#include "../src/nn/types/transformer/transformer_infer_ops.h"

static TransformerInferContext* tf_ctx_alloc(TransformerModelConfig* cfg) {
    TransformerInferContext* ctx = (TransformerInferContext*)calloc(1, sizeof(TransformerInferContext));
    if (!ctx) return NULL;
    ctx->expected_network_hash = 0xDEAD;
    ctx->expected_layout_hash  = 0xBEEF;
    int rc = nn_transformer_init_parameters(ctx, cfg, cfg->model_dim, cfg->model_dim);
    if (rc != 0) { fprintf(stderr, "INIT FAIL rc=%d\n", rc); free(ctx); return NULL; }
    return ctx;
}

int main(void) {
    TransformerModelConfig cfg;
    (void)memset(&cfg, 0, sizeof(cfg));
    cfg.vocab_size = 32; cfg.model_dim = 16; cfg.max_seq_length = 8;
    cfg.max_response_classes = 4; cfg.max_text_length = 64; cfg.seed = 42;
    cfg.use_rms_norm = 0; cfg.use_skip = 0; cfg.use_dropout = 0;

    /* Test 1: Two separate contexts — first passes should match */
    {
        fprintf(stderr, "=== Two separate contexts ===\n");
        TransformerInferContext* ctx1 = tf_ctx_alloc(&cfg);
        TransformerInferContext* ctx2 = tf_ctx_alloc(&cfg);
        if (!ctx1 || !ctx2) { fprintf(stderr, "ALLOC FAIL\n"); return 1; }
        nn_transformer_find_or_add_class(ctx1, "x");
        nn_transformer_find_or_add_class(ctx1, "y");
        nn_transformer_find_or_add_class(ctx2, "x");
        nn_transformer_find_or_add_class(ctx2, "y");
        float p1[4] = {0}, p2[4] = {0};
        nn_transformer_predict_class(ctx1, "stability", p1, 4, NULL);
        nn_transformer_predict_class(ctx2, "stability", p2, 4, NULL);
        fprintf(stderr, "ctx1: [%.10f, %.10f]\n", (double)p1[0], (double)p1[1]);
        fprintf(stderr, "ctx2: [%.10f, %.10f]\n", (double)p2[0], (double)p2[1]);
        fprintf(stderr, "diff: %.6e %s\n\n",
            (double)(p1[0]-p2[0]),
            fabs(p1[0]-p2[0]) > 1e-6f ? "FAIL" : "OK");
        nn_transformer_infer_destroy(ctx1);
        nn_transformer_infer_destroy(ctx2);
    }

    /* Test 2: Same context, second call after some function call in between */
    {
        fprintf(stderr, "=== Same context with intervening calls ===\n");
        TransformerInferContext* ctx = tf_ctx_alloc(&cfg);
        if (!ctx) { fprintf(stderr, "ALLOC FAIL\n"); return 1; }
        nn_transformer_find_or_add_class(ctx, "x");
        nn_transformer_find_or_add_class(ctx, "y");
        float p1[4] = {0}, p2[4] = {0};
        nn_transformer_predict_class(ctx, "stability", p1, 4, NULL);
        /* intervening: call a different function */
        nn_transformer_find_class(ctx, "x");
        nn_transformer_predict_class(ctx, "stability", p2, 4, NULL);
        fprintf(stderr, "pass1: [%.10f, %.10f]\n", (double)p1[0], (double)p1[1]);
        fprintf(stderr, "pass2: [%.10f, %.10f]\n", (double)p2[0], (double)p2[1]);
        fprintf(stderr, "diff: %.6e %s\n\n",
            (double)(p1[0]-p2[0]),
            fabs(p1[0]-p2[0]) > 1e-6f ? "FAIL" : "OK");
        nn_transformer_infer_destroy(ctx);
    }

    /* Test 3: Third call to verify pattern */
    {
        fprintf(stderr, "=== Three calls on same context ===\n");
        TransformerInferContext* ctx = tf_ctx_alloc(&cfg);
        if (!ctx) { fprintf(stderr, "ALLOC FAIL\n"); return 1; }
        nn_transformer_find_or_add_class(ctx, "x");
        nn_transformer_find_or_add_class(ctx, "y");
        float p[3][4] = {{0}};
        nn_transformer_predict_class(ctx, "stability", p[0], 4, NULL);
        nn_transformer_predict_class(ctx, "stability", p[1], 4, NULL);
        nn_transformer_predict_class(ctx, "stability", p[2], 4, NULL);
        fprintf(stderr, "p0: [%.10f, %.10f]\n", (double)p[0][0], (double)p[0][1]);
        fprintf(stderr, "p1: [%.10f, %.10f]\n", (double)p[1][0], (double)p[1][1]);
        fprintf(stderr, "p2: [%.10f, %.10f]\n", (double)p[2][0], (double)p[2][1]);
        fprintf(stderr, "d01: %.6e d02: %.6e d12: %.6e\n",
            (double)(p[0][0]-p[1][0]),
            (double)(p[0][0]-p[2][0]),
            (double)(p[1][0]-p[2][0]));
        nn_transformer_infer_destroy(ctx);
    }

    return 0;
}
