#ifndef CONFIG_USER_H
#define CONFIG_USER_H

// ==========================================
// Network Architecture Configuration
// ==========================================

// Maximum length of input token sequence (e.g., "walk(3,3)")
#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN     32
#endif

// Dimension of the embedding vector and hidden states
#ifndef EMBED_DIM
#define EMBED_DIM       64
#endif

// Number of Transformer Encoder Layers
#ifndef NUM_LAYERS
#define NUM_LAYERS      4
#endif

// Number of Attention Heads per layer
#ifndef NUM_HEADS
#define NUM_HEADS       4
#endif

// Dimension of the Feed-Forward Network (usually 4x EMBED_DIM)
#ifndef FFN_DIM
#define FFN_DIM         256
#endif

// Vocabulary Size (Maximum number of unique tokens)
#ifndef VOCAB_SIZE
#define VOCAB_SIZE      128
#endif

// Number of Experts in Mixture-of-Experts (MoE) layer
#ifndef NUM_EXPERTS
#define NUM_EXPERTS     4
#endif

// Number of active experts per token
#ifndef K_TOP_EXPERTS
#define K_TOP_EXPERTS   2
#endif

// ==========================================
// Input / Output Configuration
// ==========================================

// Dimension of the continuous state vector input
// e.g., [angle_1, angle_2, gyro_x, gyro_y]
#ifndef STATE_DIM
#define STATE_DIM       8
#endif

// Dimension of the output actuator vector
// e.g., [Key_W, Key_A, Mouse_X, Mouse_Y]
#ifndef OUTPUT_DIM
#define OUTPUT_DIM      4
#endif

// ==========================================
// IO Mapping Definition
// ==========================================
// Define the activation function for each output neuron.
// 0: Sigmoid (Binary Switch), 1: Tanh (Analog Control)
// This array will be used by the Profiler to generate the mapping table.
// User must ensure the array size matches OUTPUT_DIM.
#ifndef IO_MAPPING_ACTIVATIONS
#if OUTPUT_DIM == 4
#define IO_MAPPING_ACTIVATIONS { \
    0, /* Out[0]: Key W (Sigmoid) */ \
    0, /* Out[1]: Key A (Sigmoid) */ \
    1, /* Out[2]: Mouse X (Tanh) */ \
    1  /* Out[3]: Mouse Y (Tanh) */ \
}
#elif OUTPUT_DIM == 7
#define IO_MAPPING_ACTIVATIONS { \
    0, 0, 0, 0, 0, 0, 0 \
}
#else
#define IO_MAPPING_ACTIVATIONS { \
    0 \
}
#endif
#endif

// Optional: Description strings for logging/debugging
#ifndef IO_MAPPING_NAMES
#if OUTPUT_DIM == 4
#define IO_MAPPING_NAMES { \
    "Key_W", \
    "Key_A", \
    "Mouse_X", \
    "Mouse_Y" \
}
#elif OUTPUT_DIM == 7
#define IO_MAPPING_NAMES { \
    "Seg_A", \
    "Seg_B", \
    "Seg_C", \
    "Seg_D", \
    "Seg_E", \
    "Seg_F", \
    "Seg_G" \
}
#else
#define IO_MAPPING_NAMES { \
    "Out0" \
}
#endif
#endif

#endif // CONFIG_USER_H
