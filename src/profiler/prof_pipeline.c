/**
 * @file prof_pipeline.c
 * @brief Pipeline intermediate type implementations
 */

#include "prof_pipeline.h"

#include <stdlib.h>
#include <string.h>

void generated_files_init(GeneratedFiles* f) {
    if (f == NULL) {
        return;
    }
    (void)memset(f, 0, sizeof(*f));
}

void generated_files_free(GeneratedFiles* f) {
    if (f == NULL) {
        return;
    }
    free(f->metadata_c);
    free(f->metadata_h);
    free(f->tokenizer_c);
    free(f->tokenizer_h);
    free(f->network_init_c);
    free(f->network_init_h);
    free(f->infer_c);
    free(f->infer_h);
    free(f->train_c);
    free(f->train_h);
    free(f->weights_save_c);
    free(f->weights_save_h);
    free(f->weights_load_c);
    free(f->weights_load_h);
    (void)memset(f, 0, sizeof(*f));
}
