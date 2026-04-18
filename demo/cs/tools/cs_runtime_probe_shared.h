#ifndef ACTION_C_DEMO_CS_RUNTIME_PROBE_SHARED_H
#define ACTION_C_DEMO_CS_RUNTIME_PROBE_SHARED_H

/*
 * Shared definitions between the external state_trace tool and the injected
 * x86 runtime probe DLL that runs inside the 32-bit Counter-Strike process.
 */

#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define CS_RUNTIME_PROBE_MAPPING_PREFIX "Local\\ActionC_CsRuntimeProbe_"
#define CS_RUNTIME_PROBE_STATUS_SIZE 32
#define CS_RUNTIME_PROBE_ERROR_SIZE 128
#define CS_RUNTIME_PROBE_TIMESTAMP_SIZE 64
#define CS_RUNTIME_PROBE_MAPPING_VERSION 1U

typedef struct CsRuntimeProbeSnapshotTag {
    unsigned int mapping_version;
    unsigned int process_id;
    unsigned int sequence;
    int has_pose;
    float pos_x;
    float pos_y;
    float pos_z;
    float yaw;
    float pitch;
    float velocity_x;
    float velocity_y;
    float velocity_z;
    char timestamp[CS_RUNTIME_PROBE_TIMESTAMP_SIZE];
    char status[CS_RUNTIME_PROBE_STATUS_SIZE];
    char last_error[CS_RUNTIME_PROBE_ERROR_SIZE];
} CsRuntimeProbeSnapshot;

#endif
