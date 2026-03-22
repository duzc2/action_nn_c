/**
 * @file prof_validate.h
 * @brief Network validation module
 *
 * Provides validation functions for network definitions:
 * - Input parameter validation
 * - Network structure validation
 * - DAG (Directed Acyclic Graph) cycle detection
 * - Port and connection validation
 *
 * All validation stops on first error (fail-fast).
 * Error descriptions are in English.
 */

#ifndef PROF_VALIDATE_H
#define PROF_VALIDATE_H

#include "profiler_types.h"
#include "network_def.h"

/**
 * @brief Validation result structure
 */
typedef struct {
    ProfStatus status;
    const char* error_message;
    const char* source_subnet;
    const char* source_port;
    size_t source_node;
    const char* target_subnet;
    const char* target_port;
    size_t target_node;
} ProfValidationResult;

/**
 * @brief Validate the generate request parameters
 *
 * Checks:
 * - Request pointer is not NULL
 * - Network definition is not NULL
 * - Output paths are valid and writable
 * - Error buffer is valid
 *
 * @param req Generate request
 * @param error Error buffer for error message
 * @return PROF_STATUS_OK if valid, error code otherwise
 */
ProfStatus prof_validate_request(
    const ProfGenerateRequest* req,
    ProfErrorBuffer* error
);

/**
 * @brief Validate network definition structure
 *
 * Checks:
 * - Network name is not empty
 * - At least one subnet exists
 * - All subnets have unique IDs
 * - All subnet types are registered
 *
 * @param network Network definition
 * @param error Error buffer
 * @return PROF_STATUS_OK if valid, error code otherwise
 */
ProfStatus prof_validate_network_def(
    const NN_NetworkDef* network,
    ProfErrorBuffer* error
);

/**
 * @brief Validate subnet definition
 *
 * Checks:
 * - Subnet ID is not empty
 * - Subnet type is registered
 * - Input/output layer sizes are valid (> 0)
 * - Hidden layer configuration is valid
 *
 * @param subnet Subnet definition
 * @param error Error buffer
 * @return PROF_STATUS_OK if valid, error code otherwise
 */
ProfStatus prof_validate_subnet(
    const NNSubnetDef* subnet,
    ProfErrorBuffer* error
);

/**
 * @brief Validate connection definitions
 *
 * Checks:
 * - Source and target subnet IDs exist
 * - Source and target ports exist
 * - Node indices are within valid range
 * - No duplicate connections
 *
 * @param network Network definition
 * @param error Error buffer
 * @return PROF_STATUS_OK if valid, error code otherwise
 */
ProfStatus prof_validate_connections(
    const NN_NetworkDef* network,
    ProfErrorBuffer* error
);

/**
 * @brief Detect cycles in network topology (DAG check)
 *
 * Uses depth-first search to detect if the network contains cycles.
 * Networks with cycles are invalid (must be DAG).
 *
 * @param network Network definition
 * @param error Error buffer
 * @return PROF_STATUS_OK if no cycles, PROF_STATUS_CYCLE_DETECTED otherwise
 */
ProfStatus prof_validate_dag(
    const NN_NetworkDef* network,
    ProfErrorBuffer* error
);

/**
 * @brief Complete validation of network definition
 *
 * Performs all validation checks in order:
 * 1. Request parameters
 * 2. Network structure
 * 3. Subnets
 * 4. Connections
 * 5. DAG cycle detection
 *
 * Stops on first error.
 *
 * @param req Generate request
 * @param error Error buffer
 * @return PROF_STATUS_OK if all validations pass, error code otherwise
 */
ProfStatus prof_validate_all(
    const ProfGenerateRequest* req,
    ProfErrorBuffer* error
);

#endif
