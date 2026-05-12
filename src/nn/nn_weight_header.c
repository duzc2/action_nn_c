/**
 * @file nn_weight_header.c
 * @brief Unified weight-file header implementation.
 */

#include "nn_weight_header.h"
#include "../utils/error.h"
#include "../utils/log.h"

int nn_weight_header_write(const NNWeightHeader* header, FILE* fp) {
    if (header == NULL || fp == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }
    if (fwrite(header, sizeof(NNWeightHeader), 1, fp) != 1) {
        return ACTION_C_ERR_IO_FAILED;
    }
    return ACTION_C_OK;
}

int nn_weight_header_read(NNWeightHeader* header,
                          uint32_t expected_abi,
                          uint64_t expected_network_hash,
                          uint64_t expected_layout_hash,
                          FILE* fp) {
    if (header == NULL || fp == NULL) {
        return ACTION_C_ERR_NULL_POINTER;
    }
    if (fread(header, sizeof(NNWeightHeader), 1, fp) != 1) {
        LOG_ERROR("Failed to read weight header");
        return ACTION_C_ERR_IO_FAILED;
    }
    if (header->magic != NN_WEIGHT_MAGIC) {
        LOG_ERROR("Bad magic: 0x%08X, expected 0x%08X",
                  header->magic, NN_WEIGHT_MAGIC);
        return ACTION_C_ERR_VERSION_MISMATCH;
    }
    if (header->abi_version != expected_abi) {
        LOG_ERROR("ABI mismatch: file=%u, runtime=%u",
                  header->abi_version, expected_abi);
        return ACTION_C_ERR_VERSION_MISMATCH;
    }
    if (header->network_hash != expected_network_hash) {
        LOG_ERROR("Network hash mismatch");
        return ACTION_C_ERR_VERSION_MISMATCH;
    }
    if (header->layout_hash != expected_layout_hash) {
        LOG_ERROR("Layout hash mismatch");
        return ACTION_C_ERR_VERSION_MISMATCH;
    }
    return ACTION_C_OK;
}
