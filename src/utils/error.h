#ifndef ACTION_C_ERROR_H
#define ACTION_C_ERROR_H

typedef enum {
    ACTION_C_OK = 0,

    /* Parameter errors */
    ACTION_C_ERR_NULL_POINTER   = -1,
    ACTION_C_ERR_INVALID_ARG    = -2,
    ACTION_C_ERR_OUT_OF_RANGE   = -3,

    /* Resource errors */
    ACTION_C_ERR_NO_MEMORY      = -10,
    ACTION_C_ERR_IO_FAILED      = -11,

    /* Data errors */
    ACTION_C_ERR_DIM_MISMATCH   = -20,
    ACTION_C_ERR_VERSION_MISMATCH = -21,
    ACTION_C_ERR_CONFIG_INVALID = -22,
    ACTION_C_ERR_CYCLE_DETECTED = -23,

    /* Lookup errors */
    ACTION_C_ERR_NOT_FOUND      = -30,

    /* Internal errors */
    ACTION_C_ERR_INTERNAL       = -99,
} ActionCError;

#endif
