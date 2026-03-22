/**
 * @file prof_hash.h
 * @brief Network hash computation module
 *
 * Computes network signature hash for:
 * - Detecting network structure changes
 * - Validating weight file compatibility
 * - Preventing cross-network weight loading
 *
 * Uses FNV-1a hash algorithm for deterministic results.
 */

#ifndef PROF_HASH_H
#define PROF_HASH_H

#include <stdint.h>
#include "network_def.h"

/**
 * @brief FNV-1a hash algorithm implementation
 *
 * @param data Input data
 * @param len Data length
 * @return Hash value
 */
uint64_t prof_fnv1a_hash(const void* data, size_t len);

/**
 * @brief Compute hash for a subnet definition
 *
 * @param subnet Subnet definition
 * @return Hash value
 */
uint64_t prof_subnet_hash(const NNSubnetDef* subnet);

/**
 * @brief Compute hash for the entire network definition
 *
 * @param network Network definition
 * @return Hash value
 */
uint64_t prof_network_hash(const NN_NetworkDef* network);

/**
 * @brief Compute layout hash for parameter layout validation
 *
 * @param network Network definition
 * @return Layout hash value
 */
uint64_t prof_layout_hash(const NN_NetworkDef* network);

#endif
