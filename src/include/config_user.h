#ifndef CONFIG_USER_H
#define CONFIG_USER_H

// ==========================================
// Network Architecture Configuration
// ==========================================

// Maximum length of input token sequence (e.g., "walk(3,3)")
#define MAX_SEQ_LEN     32

// Dimension of the embedding vector and hidden states
#define EMBED_DIM       64

// Number of Transformer Encoder Layers
#define NUM_LAYERS      4

// Number of Attention Heads per layer
#define NUM_HEADS       4

// Dimension of the Feed-Forward Network (usually 4x EMBED_DIM)
#define FFN_DIM         256

// Vocabulary Size (Maximum number of unique tokens)
#define VOCAB_SIZE      128

// Number of Experts in Mixture-of-Experts (MoE) layer
#define NUM_EXPERTS     4

// Number of active experts per token
#define K_TOP_EXPERTS   2

// ==========================================
// Input / Output Configuration
// ==========================================

// Dimension of the continuous state vector input
// e.g., [angle_1, angle_2, gyro_x, gyro_y]
#define STATE_DIM       8

// Dimension of the output actuator vector
// e.g., [Key_W, Key_A, Mouse_X, Mouse_Y]
#define OUTPUT_DIM      4

// ==========================================
// IO Mapping Definition
// ==========================================
// Define the activation function for each output neuron.
// 0: Sigmoid (Binary Switch), 1: Tanh (Analog Control)
// This array will be used by the Profiler to generate the mapping table.
// User must ensure the array size matches OUTPUT_DIM.
#define IO_MAPPING_ACTIVATIONS { \
    0, /* Out[0]: Key W (Sigmoid) */ \
    0, /* Out[1]: Key A (Sigmoid) */ \
    1, /* Out[2]: Mouse X (Tanh) */ \
    1  /* Out[3]: Mouse Y (Tanh) */ \
}

// Optional: Description strings for logging/debugging
#define IO_MAPPING_NAMES { \
    "Key_W", \
    "Key_A", \
    "Mouse_X", \
    "Mouse_Y" \
}

#endif // CONFIG_USER_H
