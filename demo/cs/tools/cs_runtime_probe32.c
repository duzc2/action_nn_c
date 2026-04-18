#include "cs_runtime_probe_shared.h"

/*
 * x86 runtime probe DLL injected into the 32-bit Counter-Strike process.
 *
 * The probe uses an INT3 hook on client.dll!V_CalcRefdef so it can capture the
 * local player's simulated origin, velocity and view angles without needing to
 * reverse-engineer a fragile global pointer chain in the engine.
 */

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

typedef struct CsRefParamsTag {
    float vieworg[3];
    float viewangles[3];
    float forward[3];
    float right[3];
    float up[3];
    float frametime;
    float time;
    int intermission;
    int paused;
    int spectator;
    int onground;
    int waterlevel;
    float simvel[3];
    float simorg[3];
    float viewheight[3];
    float idealpitch;
    float cl_viewangles[3];
    float health;
    float crosshairangle[3];
    float viewsize;
    float punchangle[3];
    int maxclients;
    int viewentity;
    int playernum;
    int max_entities;
    int demoplayback;
    int hardware;
    int smoothing;
    void* cmd;
    void* movevars;
    int viewport[4];
    int next_view;
    int only_client_draw;
} CsRefParams;

static PVOID g_exception_handler_handle = NULL;
static HANDLE g_mapping_handle = NULL;
static CsRuntimeProbeSnapshot* g_shared_snapshot = NULL;
static unsigned char* g_hook_address = NULL;
static unsigned char g_original_byte = 0U;
static LONG g_single_step_pending = 0;

static void cs_probe_write_timestamp(char* buffer, size_t buffer_size) {
    SYSTEMTIME local_time;

    if (buffer == NULL || buffer_size == 0U) {
        return;
    }

    GetLocalTime(&local_time);
    (void)snprintf(buffer,
                   buffer_size,
                   "%04u-%02u-%02uT%02u:%02u:%02u.%03u",
                   (unsigned int)local_time.wYear,
                   (unsigned int)local_time.wMonth,
                   (unsigned int)local_time.wDay,
                   (unsigned int)local_time.wHour,
                   (unsigned int)local_time.wMinute,
                   (unsigned int)local_time.wSecond,
                   (unsigned int)local_time.wMilliseconds);
}

static void cs_probe_publish_status(const char* status_text, const char* error_text) {
    LONG sequence_value;

    if (g_shared_snapshot == NULL) {
        return;
    }

    sequence_value = InterlockedIncrement((volatile LONG*)&g_shared_snapshot->sequence);
    (void)sequence_value;

    g_shared_snapshot->mapping_version = CS_RUNTIME_PROBE_MAPPING_VERSION;
    g_shared_snapshot->process_id = (unsigned int)GetCurrentProcessId();
    if (status_text != NULL) {
        (void)snprintf(g_shared_snapshot->status, sizeof(g_shared_snapshot->status), "%s", status_text);
    }
    if (error_text != NULL) {
        (void)snprintf(g_shared_snapshot->last_error, sizeof(g_shared_snapshot->last_error), "%s", error_text);
    } else {
        g_shared_snapshot->last_error[0] = '\0';
    }

    InterlockedIncrement((volatile LONG*)&g_shared_snapshot->sequence);
}

static void cs_probe_publish_pose(const CsRefParams* params) {
    LONG sequence_value;

    if (g_shared_snapshot == NULL || params == NULL) {
        return;
    }

    sequence_value = InterlockedIncrement((volatile LONG*)&g_shared_snapshot->sequence);
    (void)sequence_value;

    g_shared_snapshot->mapping_version = CS_RUNTIME_PROBE_MAPPING_VERSION;
    g_shared_snapshot->process_id = (unsigned int)GetCurrentProcessId();
    g_shared_snapshot->has_pose = 1;
    g_shared_snapshot->pos_x = params->simorg[0];
    g_shared_snapshot->pos_y = params->simorg[1];
    g_shared_snapshot->pos_z = params->simorg[2];
    g_shared_snapshot->velocity_x = params->simvel[0];
    g_shared_snapshot->velocity_y = params->simvel[1];
    g_shared_snapshot->velocity_z = params->simvel[2];
    g_shared_snapshot->pitch = params->cl_viewangles[0];
    g_shared_snapshot->yaw = params->cl_viewangles[1];
    cs_probe_write_timestamp(g_shared_snapshot->timestamp, sizeof(g_shared_snapshot->timestamp));
    (void)snprintf(g_shared_snapshot->status, sizeof(g_shared_snapshot->status), "%s", "running");
    g_shared_snapshot->last_error[0] = '\0';

    InterlockedIncrement((volatile LONG*)&g_shared_snapshot->sequence);
}

static int cs_probe_patch_byte(unsigned char* address, unsigned char value) {
    DWORD old_protect;

    if (address == NULL) {
        return 0;
    }

    if (!VirtualProtect(address, 1U, PAGE_EXECUTE_READWRITE, &old_protect)) {
        return 0;
    }
    *address = value;
    FlushInstructionCache(GetCurrentProcess(), address, 1U);
    (void)VirtualProtect(address, 1U, old_protect, &old_protect);
    return 1;
}

static LONG CALLBACK cs_probe_exception_handler(PEXCEPTION_POINTERS exception_info) {
    DWORD exception_code;

    if (exception_info == NULL || exception_info->ExceptionRecord == NULL || exception_info->ContextRecord == NULL) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    exception_code = exception_info->ExceptionRecord->ExceptionCode;
    if (exception_code == EXCEPTION_BREAKPOINT && g_hook_address != NULL) {
        ULONG_PTR eip_value;

        eip_value = (ULONG_PTR)exception_info->ContextRecord->Eip;
        if (eip_value == (ULONG_PTR)g_hook_address + 1U ||
            (ULONG_PTR)exception_info->ExceptionRecord->ExceptionAddress == (ULONG_PTR)g_hook_address) {
            const CsRefParams* params;

            params = *(const CsRefParams* const*)(exception_info->ContextRecord->Esp + 4U);
            cs_probe_publish_pose(params);
            if (!cs_probe_patch_byte(g_hook_address, g_original_byte)) {
                cs_probe_publish_status("error", "Failed to restore V_CalcRefdef byte.");
                return EXCEPTION_CONTINUE_SEARCH;
            }

            exception_info->ContextRecord->Eip = (DWORD)(ULONG_PTR)g_hook_address;
            exception_info->ContextRecord->EFlags |= 0x100U;
            InterlockedExchange(&g_single_step_pending, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }

    if (exception_code == EXCEPTION_SINGLE_STEP &&
        g_hook_address != NULL &&
        InterlockedCompareExchange(&g_single_step_pending, 0, 1) == 1) {
        if (!cs_probe_patch_byte(g_hook_address, 0xCCU)) {
            cs_probe_publish_status("error", "Failed to re-arm V_CalcRefdef breakpoint.");
            return EXCEPTION_CONTINUE_SEARCH;
        }
        exception_info->ContextRecord->EFlags &= ~0x100U;
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

static int cs_probe_build_mapping_name(char* buffer, size_t buffer_size) {
    int written_size;

    written_size = snprintf(buffer,
                            buffer_size,
                            "%s%lu",
                            CS_RUNTIME_PROBE_MAPPING_PREFIX,
                            (unsigned long)GetCurrentProcessId());
    return written_size >= 0 && (size_t)written_size < buffer_size;
}

static int cs_probe_open_mapping(void) {
    char mapping_name[128];

    if (!cs_probe_build_mapping_name(mapping_name, sizeof(mapping_name))) {
        return 0;
    }

    g_mapping_handle = CreateFileMappingA(INVALID_HANDLE_VALUE,
                                          NULL,
                                          PAGE_READWRITE,
                                          0U,
                                          (DWORD)sizeof(CsRuntimeProbeSnapshot),
                                          mapping_name);
    if (g_mapping_handle == NULL) {
        return 0;
    }

    g_shared_snapshot = (CsRuntimeProbeSnapshot*)MapViewOfFile(g_mapping_handle,
                                                               FILE_MAP_ALL_ACCESS,
                                                               0U,
                                                               0U,
                                                               sizeof(CsRuntimeProbeSnapshot));
    if (g_shared_snapshot == NULL) {
        CloseHandle(g_mapping_handle);
        g_mapping_handle = NULL;
        return 0;
    }

    ZeroMemory(g_shared_snapshot, sizeof(*g_shared_snapshot));
    g_shared_snapshot->mapping_version = CS_RUNTIME_PROBE_MAPPING_VERSION;
    g_shared_snapshot->process_id = (unsigned int)GetCurrentProcessId();
    (void)snprintf(g_shared_snapshot->status, sizeof(g_shared_snapshot->status), "%s", "initializing");
    return 1;
}

static int cs_probe_install_breakpoint_hook(void) {
    HMODULE client_module;
    FARPROC export_address;

    client_module = GetModuleHandleA("client.dll");
    if (client_module == NULL) {
        cs_probe_publish_status("error", "client.dll is not loaded in the target process.");
        return 0;
    }

    export_address = GetProcAddress(client_module, "V_CalcRefdef");
    if (export_address == NULL) {
        cs_probe_publish_status("error", "Failed to resolve client.dll!V_CalcRefdef.");
        return 0;
    }

    g_hook_address = (unsigned char*)export_address;
    g_original_byte = *g_hook_address;
    if (g_original_byte == 0xCCU) {
        cs_probe_publish_status("running", NULL);
        return 1;
    }

    g_exception_handler_handle = AddVectoredExceptionHandler(1UL, cs_probe_exception_handler);
    if (g_exception_handler_handle == NULL) {
        cs_probe_publish_status("error", "Failed to install vectored exception handler.");
        return 0;
    }

    if (!cs_probe_patch_byte(g_hook_address, 0xCCU)) {
        cs_probe_publish_status("error", "Failed to patch client.dll!V_CalcRefdef.");
        return 0;
    }

    cs_probe_publish_status("running", NULL);
    return 1;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    (void)instance;
    (void)reserved;

    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(instance);
        if (!cs_probe_open_mapping()) {
            return TRUE;
        }
        (void)cs_probe_install_breakpoint_hook();
    } else if (reason == DLL_PROCESS_DETACH) {
        if (g_hook_address != NULL && g_original_byte != 0U) {
            (void)cs_probe_patch_byte(g_hook_address, g_original_byte);
        }
        if (g_exception_handler_handle != NULL) {
            (void)RemoveVectoredExceptionHandler(g_exception_handler_handle);
            g_exception_handler_handle = NULL;
        }
        if (g_shared_snapshot != NULL) {
            cs_probe_publish_status("stopped", NULL);
            (void)UnmapViewOfFile(g_shared_snapshot);
            g_shared_snapshot = NULL;
        }
        if (g_mapping_handle != NULL) {
            CloseHandle(g_mapping_handle);
            g_mapping_handle = NULL;
        }
    }

    return TRUE;
}
