/**
 * @file network_def.h
 * @brief Network definition structures
 *
 * Defines the structured network description model used by user programs.
 * These structures describe the complete network topology including:
 * - Subnet list (supports nesting)
 * - Connection list
 * - Port type and shape description
 * - Activation function strategy
 *
 * Constraints:
 * - All types must comply with C99 standard
 * - Network types use semantic naming (not numeric codes)
 * - Must support 0 hidden layers configuration
 * - Activation function uses "default + per-node override" mechanism
 */

#ifndef NETWORK_DEF_H
#define NETWORK_DEF_H

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Activation function types
 */
typedef enum {
    NN_ACTIVATION_NONE = 0,
    NN_ACTIVATION_RELU = 1,
    NN_ACTIVATION_SIGMOID = 2,
    NN_ACTIVATION_TANH = 3,
    NN_ACTIVATION_SOFTMAX = 4,
    NN_ACTIVATION_LEAKY_RELU = 5
} NNActivationType;

/**
 * @brief Port direction
 */
typedef enum {
    NN_PORT_INPUT = 0,
    NN_PORT_OUTPUT = 1
} NNPortDirection;

/**
 * @brief Connection merge strategy for multi-source connections
 */
typedef enum {
    NN_MERGE_SUM = 0,
    NN_MERGE_CONCAT = 1,
    NN_MERGE_AVERAGE = 2
} NNMergeStrategy;

/**
 * @brief Single node activation override
 *
 * Allows overriding activation function for a specific node.
 */
typedef struct {
    size_t node_index;
    NNActivationType activation;
} NNNodeActivation;

/**
 * @brief Port definition
 *
 * Defines an input or output port of a subnet.
 */
typedef struct {
    const char* port_name;
    NNPortDirection direction;
    size_t node_count;
    NNActivationType default_activation;
    NNNodeActivation* node_overrides;
    size_t node_override_count;
} NNPortDef;

/**
 * @brief Subnet definition
 *
 * Defines a subnet within the network.
 * Can be nested (subnet containing subnets).
 */
typedef struct {
    const char* subnet_id;
    const char* subnet_type;
    size_t input_layer_size;
    size_t output_layer_size;
    size_t hidden_layer_count;
    size_t* hidden_layer_sizes;
    NNActivationType default_activation;
    NNNodeActivation* node_overrides;
    size_t node_override_count;
    NNPortDef* inputs;
    size_t input_count;
    NNPortDef* outputs;
    size_t output_count;
    struct NNSubnetDef_tag** subnets;
    size_t subnet_count;
} NNSubnetDef;

/**
 * @brief Single connection definition
 *
 * Defines a connection from one subnet output to another subnet input.
 */
typedef struct {
    const char* source_subnet_id;
    const char* source_port_name;
    size_t source_node_index;
    const char* target_subnet_id;
    const char* target_port_name;
    size_t target_node_index;
    NNMergeStrategy merge_strategy;
} NNConnectionDef;

/**
 * @brief Node routing definition
 *
 * Defines where a specific node output should be routed.
 */
typedef struct {
    const char* source_subnet_id;
    const char* source_port_name;
    size_t source_node_index;
    const char* target_subnet_id;
    const char* target_port_name;
    size_t target_node_index;
} NNNodeRouting;

/**
 * @brief Network definition root structure
 *
 * The top-level structure describing a complete network.
 * Contains subnets and connections.
 */
typedef struct {
    const char* network_name;
    const char* network_version;
    NNSubnetDef** subnets;
    size_t subnet_count;
    NNConnectionDef** connections;
    size_t connection_count;
    NNNodeRouting** routings;
    size_t routing_count;
} NN_NetworkDef;

/**
 * @brief Create an empty network definition
 *
 * @return New network definition, must be freed by nn_network_def_free()
 */
NN_NetworkDef* nn_network_def_create(const char* name);

/**
 * @brief Free network definition
 *
 * @param def Network definition to free
 */
void nn_network_def_free(NN_NetworkDef* def);

/**
 * @brief Add a subnet to network definition
 *
 * @param network Network definition
 * @param subnet Subnet to add
 */
void nn_network_def_add_subnet(NN_NetworkDef* network, NNSubnetDef* subnet);

/**
 * @brief Add a connection to network definition
 *
 * @param network Network definition
 * @param connection Connection to add
 */
void nn_network_def_add_connection(NN_NetworkDef* network, NNConnectionDef* connection);

/**
 * @brief Create a subnet definition
 *
 * @param subnet_id Unique subnet identifier
 * @param subnet_type Semantic type (e.g., "mlp", "transformer", "cnn")
 * @param input_size Input layer node count
 * @param output_size Output layer node count
 * @return New subnet definition, must be freed by nn_subnet_def_free()
 */
NNSubnetDef* nn_subnet_def_create(
    const char* subnet_id,
    const char* subnet_type,
    size_t input_size,
    size_t output_size
);

/**
 * @brief Set hidden layer configuration
 *
 * @param subnet Subnet definition
 * @param layer_count Number of hidden layers (can be 0)
 * @param layer_sizes Array of node counts for each hidden layer
 */
void nn_subnet_def_set_hidden_layers(
    NNSubnetDef* subnet,
    size_t layer_count,
    const size_t* layer_sizes
);

/**
 * @brief Free subnet definition
 *
 * @param subnet Subnet to free
 */
void nn_subnet_def_free(NNSubnetDef* subnet);

/**
 * @brief Create a connection definition
 *
 * @param src_subnet Source subnet ID
 * @param src_port Source port name
 * @param src_node Source node index
 * @param tgt_subnet Target subnet ID
 * @param tgt_port Target port name
 * @param tgt_node Target node index
 * @return New connection definition, must be freed by nn_connection_def_free()
 */
NNConnectionDef* nn_connection_def_create(
    const char* src_subnet,
    const char* src_port,
    size_t src_node,
    const char* tgt_subnet,
    const char* tgt_port,
    size_t tgt_node
);

/**
 * @brief Free connection definition
 *
 * @param conn Connection to free
 */
void nn_connection_def_free(NNConnectionDef* conn);

#endif
