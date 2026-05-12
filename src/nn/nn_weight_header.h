/**
 * @file nn_weight_header.h
 * @brief Unified weight-file header with magic, ABI, and hash verification.
 */

#ifndef ACTION_C_NN_WEIGHT_HEADER_H
#define ACTION_C_NN_WEIGHT_HEADER_H

#include <stdint.h>
#include <stdio.h>

#define NN_WEIGHT_MAGIC 0x4E4E5746  /* "NNWF" */

typedef struct {
    uint32_t magic;
    uint32_t abi_version;
    uint64_t network_hash;
    uint64_t layout_hash;
    uint32_t reserved[4];
} NNWeightHeader;

/* Write the standard header to a FILE stream. Returns ACTION_C_OK on success. */
int nn_weight_header_write(const NNWeightHeader* header, FILE* fp);

/* Read and validate the standard header from a FILE stream.
 * Returns ACTION_C_OK on success, <0 on failure. */
int nn_weight_header_read(NNWeightHeader* header,
                          uint32_t expected_abi,
                          uint64_t expected_network_hash,
                          uint64_t expected_layout_hash,
                          FILE* fp);

#endif /* ACTION_C_NN_WEIGHT_HEADER_H */
