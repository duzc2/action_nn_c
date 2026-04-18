#include "cs_tool_common.h"
#include "cs_runtime_probe_shared.h"

/*
 * Version 1 state_trace tool.
 *
 * Public commands:
 * - start
 * - stop
 * - status
 *
 * The worker reads capture progress from capture_state.json and samples the
 * latest pose published by the injected x86 runtime probe DLL inside the
 * Counter-Strike process.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <tlhelp32.h>
#include <windows.h>
#endif

typedef struct CsStateTraceCommandTag {
    char session_root[CS_TOOL_MAX_PATH];
    char session_id[64];
} CsStateTraceCommand;

static int cs_arg_is_option(const char* value, const char* expected) {
    return value != NULL && expected != NULL && strcmp(value, expected) == 0;
}

static int cs_state_default_session_root(char* out_root, size_t out_size, char* error_buffer, size_t error_buffer_size) {
    char current_directory[CS_TOOL_MAX_PATH];

    if (!cs_tool_get_current_directory(current_directory, sizeof(current_directory), error_buffer, error_buffer_size)) {
        return 0;
    }

    return cs_tool_join_two_paths(out_root, out_size, current_directory, "demo", "cs\\data\\v1_area\\raw");
}

static int cs_state_reload_capture_state(const char* state_path, CsCaptureState* state, char* error_buffer, size_t error_buffer_size) {
    CsCaptureState latest_state;

    memset(&latest_state, 0, sizeof(latest_state));
    if (!cs_tool_read_capture_state(state_path, &latest_state, error_buffer, error_buffer_size)) {
        return 0;
    }

    *state = latest_state;
    return 1;
}

static int cs_state_parse_options(int argc, char** argv, int start_index, CsStateTraceCommand* command, char* error_buffer, size_t error_buffer_size) {
    int index;

    memset(command, 0, sizeof(*command));
    if (!cs_state_default_session_root(command->session_root, sizeof(command->session_root), error_buffer, error_buffer_size)) {
        return 0;
    }

    index = start_index;
    while (index < argc) {
        const char* option_name;
        const char* option_value;

        option_name = argv[index];
        if (index + 1 >= argc) {
            cs_tool_set_error(error_buffer, error_buffer_size, "Missing value after option: %s", option_name);
            return 0;
        }
        option_value = argv[index + 1];

        if (cs_arg_is_option(option_name, "--session-root")) {
            if (!cs_tool_copy_string(command->session_root, sizeof(command->session_root), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Session root is too long.");
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--session-id")) {
            if (!cs_tool_copy_string(command->session_id, sizeof(command->session_id), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Session id is too long.");
                return 0;
            }
        } else {
            cs_tool_set_error(error_buffer, error_buffer_size, "Unsupported option: %s", option_name);
            return 0;
        }

        index += 2;
    }

    return 1;
}

static int cs_state_validate_common(const CsStateTraceCommand* command, char* error_buffer, size_t error_buffer_size) {
    if (command->session_id[0] == '\0') {
        cs_tool_set_error(error_buffer, error_buffer_size, "Missing required option: --session-id");
        return 0;
    }
    return 1;
}

static int cs_state_count_existing_records(const char* trace_path, int* out_count, char* error_buffer, size_t error_buffer_size) {
    char* text;
    size_t text_size;
    size_t index;
    int line_count;

    if (out_count == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "State trace count output is null.");
        return 0;
    }

    *out_count = 0;
    if (!cs_tool_file_exists(trace_path)) {
        return 1;
    }

    text = NULL;
    text_size = 0U;
    if (!cs_tool_read_text_file(trace_path, &text, &text_size, error_buffer, error_buffer_size)) {
        return 0;
    }

    line_count = 0;
    for (index = 0U; index < text_size; ++index) {
        if (text[index] == '\n') {
            line_count++;
        }
    }

    if (text_size > 0U && text[text_size - 1U] != '\n') {
        line_count++;
    }

    free(text);
    *out_count = line_count;
    return 1;
}

#ifdef _WIN32
static int cs_state_find_game_process(DWORD* out_pid, char* error_buffer, size_t error_buffer_size) {
    HWND hwnd;
    DWORD process_id;

    hwnd = cs_tool_find_counter_strike_window();
    if (hwnd == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to find Counter-Strike window.");
        return 0;
    }

    process_id = 0U;
    (void)GetWindowThreadProcessId(hwnd, &process_id);
    if (process_id == 0U) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to query Counter-Strike process id.");
        return 0;
    }

    *out_pid = process_id;
    return 1;
}

static int cs_state_build_probe_mapping_name(DWORD process_id, char* buffer, size_t buffer_size, char* error_buffer, size_t error_buffer_size) {
    int written_size;

    written_size = snprintf(buffer, buffer_size, "%s%lu", CS_RUNTIME_PROBE_MAPPING_PREFIX, (unsigned long)process_id);
    if (written_size < 0 || (size_t)written_size >= buffer_size) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Runtime probe mapping name is too long.");
        return 0;
    }
    return 1;
}

static int cs_state_get_executable_dir(char* out_dir, size_t out_dir_size, char* error_buffer, size_t error_buffer_size) {
    char executable_path[CS_TOOL_MAX_PATH];
    char* last_slash;

    if (!cs_tool_get_executable_path(executable_path, sizeof(executable_path), error_buffer, error_buffer_size)) {
        return 0;
    }

    last_slash = strrchr(executable_path, '\\');
    if (last_slash == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Executable directory is malformed.");
        return 0;
    }
    *last_slash = '\0';

    if (!cs_tool_copy_string(out_dir, out_dir_size, executable_path)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Executable directory is too long.");
        return 0;
    }
    return 1;
}

static int cs_state_get_probe_dll_path(char* out_path, size_t out_path_size, char* error_buffer, size_t error_buffer_size) {
    char executable_dir[CS_TOOL_MAX_PATH];

    if (!cs_state_get_executable_dir(executable_dir, sizeof(executable_dir), error_buffer, error_buffer_size)) {
        return 0;
    }
    if (!cs_tool_join_path(out_path, out_path_size, executable_dir, "cs_runtime_probe32.dll")) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Runtime probe DLL path is too long.");
        return 0;
    }
    if (!cs_tool_file_exists(out_path)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Runtime probe DLL was not built: %s", out_path);
        return 0;
    }
    return 1;
}

static const unsigned char* cs_state_rva_to_ptr32(const unsigned char* buffer,
                                                  size_t buffer_size,
                                                  const IMAGE_NT_HEADERS32* nt_headers,
                                                  DWORD rva) {
    const IMAGE_SECTION_HEADER* sections;
    WORD section_index;

    if (buffer == NULL || nt_headers == NULL) {
        return NULL;
    }

    if (rva < nt_headers->OptionalHeader.SizeOfHeaders && (size_t)rva < buffer_size) {
        return buffer + rva;
    }

    sections = IMAGE_FIRST_SECTION(nt_headers);
    for (section_index = 0U; section_index < nt_headers->FileHeader.NumberOfSections; ++section_index) {
        DWORD section_rva;
        DWORD section_size;
        DWORD section_offset;

        section_rva = sections[section_index].VirtualAddress;
        section_size = sections[section_index].Misc.VirtualSize;
        if (sections[section_index].SizeOfRawData > section_size) {
            section_size = sections[section_index].SizeOfRawData;
        }

        if (rva < section_rva || rva >= section_rva + section_size) {
            continue;
        }

        section_offset = sections[section_index].PointerToRawData + (rva - section_rva);
        if ((size_t)section_offset >= buffer_size) {
            return NULL;
        }
        return buffer + section_offset;
    }

    return NULL;
}

static int cs_state_find_module_in_process(DWORD process_id,
                                           const char* module_name,
                                           unsigned long long* out_base,
                                           char* out_module_path,
                                           size_t out_module_path_size,
                                           char* error_buffer,
                                           size_t error_buffer_size) {
    HANDLE snapshot_handle;
    MODULEENTRY32 module_entry;
    int found;

    snapshot_handle = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, process_id);
    if (snapshot_handle == INVALID_HANDLE_VALUE) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to enumerate process modules.");
        return 0;
    }

    found = 0;
    memset(&module_entry, 0, sizeof(module_entry));
    module_entry.dwSize = sizeof(module_entry);

    if (Module32First(snapshot_handle, &module_entry)) {
        do {
            if (_stricmp(module_entry.szModule, module_name) == 0) {
                *out_base = (unsigned long long)(ULONG_PTR)module_entry.modBaseAddr;
                if (out_module_path != NULL && out_module_path_size > 0U) {
                    (void)cs_tool_copy_string(out_module_path, out_module_path_size, module_entry.szExePath);
                }
                found = 1;
                break;
            }
        } while (Module32Next(snapshot_handle, &module_entry));
    }

    CloseHandle(snapshot_handle);

    if (!found) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to find %s in Counter-Strike process.", module_name);
        return 0;
    }
    return 1;
}

static int cs_state_read_export_rva32(const char* dll_path,
                                      const char* export_name,
                                      DWORD* out_rva,
                                      char* error_buffer,
                                      size_t error_buffer_size) {
    FILE* file_handle;
    long file_size;
    unsigned char* buffer;
    IMAGE_DOS_HEADER* dos_header;
    IMAGE_NT_HEADERS32* nt_headers;
    IMAGE_EXPORT_DIRECTORY* export_directory;
    DWORD export_rva;
    DWORD export_size;
    int found;

    file_handle = fopen(dll_path, "rb");
    if (file_handle == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open DLL for export parsing: %s", dll_path);
        return 0;
    }

    if (fseek(file_handle, 0L, SEEK_END) != 0) {
        fclose(file_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to seek DLL: %s", dll_path);
        return 0;
    }

    file_size = ftell(file_handle);
    if (file_size <= 0L) {
        fclose(file_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to determine DLL size: %s", dll_path);
        return 0;
    }

    if (fseek(file_handle, 0L, SEEK_SET) != 0) {
        fclose(file_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to rewind DLL: %s", dll_path);
        return 0;
    }

    buffer = (unsigned char*)malloc((size_t)file_size);
    if (buffer == NULL) {
        fclose(file_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Out of memory while reading DLL: %s", dll_path);
        return 0;
    }

    if (fread(buffer, 1U, (size_t)file_size, file_handle) != (size_t)file_size) {
        free(buffer);
        fclose(file_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to read DLL: %s", dll_path);
        return 0;
    }
    fclose(file_handle);

    dos_header = (IMAGE_DOS_HEADER*)buffer;
    if (dos_header->e_magic != IMAGE_DOS_SIGNATURE) {
        free(buffer);
        cs_tool_set_error(error_buffer, error_buffer_size, "DLL is not a valid PE file: %s", dll_path);
        return 0;
    }

    nt_headers = (IMAGE_NT_HEADERS32*)(buffer + dos_header->e_lfanew);
    if (nt_headers->Signature != IMAGE_NT_SIGNATURE ||
        nt_headers->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR32_MAGIC) {
        free(buffer);
        cs_tool_set_error(error_buffer, error_buffer_size, "DLL is not a valid 32-bit PE file: %s", dll_path);
        return 0;
    }

    export_rva = nt_headers->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress;
    export_size = nt_headers->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].Size;
    if (export_rva == 0U || export_size == 0U) {
        free(buffer);
        cs_tool_set_error(error_buffer, error_buffer_size, "DLL does not expose an export directory: %s", dll_path);
        return 0;
    }

    export_directory = (IMAGE_EXPORT_DIRECTORY*)cs_state_rva_to_ptr32(buffer,
                                                                      (size_t)file_size,
                                                                      nt_headers,
                                                                      export_rva);
    if (export_directory == NULL) {
        free(buffer);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to map export directory from %s", dll_path);
        return 0;
    }
    found = 0;

    {
        const DWORD* name_rvas;
        const WORD* name_ordinals;
        const DWORD* function_rvas;
        DWORD index;

        name_rvas = (const DWORD*)cs_state_rva_to_ptr32(buffer,
                                                        (size_t)file_size,
                                                        nt_headers,
                                                        export_directory->AddressOfNames);
        name_ordinals = (const WORD*)cs_state_rva_to_ptr32(buffer,
                                                           (size_t)file_size,
                                                           nt_headers,
                                                           export_directory->AddressOfNameOrdinals);
        function_rvas = (const DWORD*)cs_state_rva_to_ptr32(buffer,
                                                            (size_t)file_size,
                                                            nt_headers,
                                                            export_directory->AddressOfFunctions);
        if (name_rvas == NULL || name_ordinals == NULL || function_rvas == NULL) {
            free(buffer);
            cs_tool_set_error(error_buffer, error_buffer_size, "Failed to map export tables from %s", dll_path);
            return 0;
        }

        for (index = 0U; index < export_directory->NumberOfNames; ++index) {
            const char* name_text;
            WORD ordinal;

            name_text = (const char*)cs_state_rva_to_ptr32(buffer,
                                                           (size_t)file_size,
                                                           nt_headers,
                                                           name_rvas[index]);
            if (name_text == NULL) {
                continue;
            }
            if (strcmp(name_text, export_name) == 0) {
                ordinal = name_ordinals[index];
                if ((DWORD)ordinal >= export_directory->NumberOfFunctions) {
                    break;
                }
                *out_rva = function_rvas[ordinal];
                if (*out_rva >= export_rva && *out_rva < export_rva + export_size) {
                    free(buffer);
                    cs_tool_set_error(error_buffer, error_buffer_size, "Forwarded exports are not supported for %s", export_name);
                    return 0;
                }
                found = 1;
                break;
            }
        }
    }

    free(buffer);

    if (!found) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to find export %s in %s", export_name, dll_path);
        return 0;
    }
    return 1;
}

static int cs_state_inject_probe_dll(DWORD process_id, char* error_buffer, size_t error_buffer_size) {
    char dll_path[CS_TOOL_MAX_PATH];
    unsigned long long existing_module_base;
    char remote_kernel32_path[CS_TOOL_MAX_PATH];
    unsigned long long remote_kernel32_base;
    DWORD load_library_rva;
    unsigned long long remote_load_library;
    HANDLE process_handle;
    LPVOID remote_path_buffer;
    HANDLE thread_handle;
    DWORD thread_exit_code;

    existing_module_base = 0ULL;
    if (cs_state_find_module_in_process(process_id,
                                        "cs_runtime_probe32.dll",
                                        &existing_module_base,
                                        NULL,
                                        0U,
                                        error_buffer,
                                        error_buffer_size)) {
        return 1;
    }

    if (!cs_state_get_probe_dll_path(dll_path, sizeof(dll_path), error_buffer, error_buffer_size) ||
        !cs_state_find_module_in_process(process_id,
                                         "kernel32.dll",
                                         &remote_kernel32_base,
                                         remote_kernel32_path,
                                         sizeof(remote_kernel32_path),
                                         error_buffer,
                                         error_buffer_size) ||
        !cs_state_read_export_rva32(remote_kernel32_path,
                                    "LoadLibraryA",
                                    &load_library_rva,
                                    error_buffer,
                                    error_buffer_size)) {
        return 0;
    }

    remote_load_library = remote_kernel32_base + (unsigned long long)load_library_rva;
    process_handle = OpenProcess(PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ,
                                 FALSE,
                                 process_id);
    if (process_handle == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open Counter-Strike process for DLL injection.");
        return 0;
    }

    remote_path_buffer = VirtualAllocEx(process_handle, NULL, strlen(dll_path) + 1U, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (remote_path_buffer == NULL) {
        CloseHandle(process_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to allocate remote memory for DLL path.");
        return 0;
    }

    if (!WriteProcessMemory(process_handle, remote_path_buffer, dll_path, strlen(dll_path) + 1U, NULL)) {
        (void)VirtualFreeEx(process_handle, remote_path_buffer, 0U, MEM_RELEASE);
        CloseHandle(process_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to write runtime probe DLL path into Counter-Strike process.");
        return 0;
    }

    thread_handle = CreateRemoteThread(process_handle,
                                       NULL,
                                       0U,
                                       (LPTHREAD_START_ROUTINE)(ULONG_PTR)remote_load_library,
                                       remote_path_buffer,
                                       0U,
                                       NULL);
    if (thread_handle == NULL) {
        (void)VirtualFreeEx(process_handle, remote_path_buffer, 0U, MEM_RELEASE);
        CloseHandle(process_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to create remote thread for runtime probe injection.");
        return 0;
    }

    if (WaitForSingleObject(thread_handle, 10000U) != WAIT_OBJECT_0) {
        CloseHandle(thread_handle);
        (void)VirtualFreeEx(process_handle, remote_path_buffer, 0U, MEM_RELEASE);
        CloseHandle(process_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Timed out while waiting for runtime probe injection.");
        return 0;
    }

    thread_exit_code = 0U;
    if (!GetExitCodeThread(thread_handle, &thread_exit_code) || thread_exit_code == 0U) {
        CloseHandle(thread_handle);
        (void)VirtualFreeEx(process_handle, remote_path_buffer, 0U, MEM_RELEASE);
        CloseHandle(process_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Counter-Strike failed to load cs_runtime_probe32.dll.");
        return 0;
    }

    CloseHandle(thread_handle);
    (void)VirtualFreeEx(process_handle, remote_path_buffer, 0U, MEM_RELEASE);
    CloseHandle(process_handle);
    return 1;
}

static int cs_state_open_probe_mapping(DWORD process_id,
                                       HANDLE* out_mapping_handle,
                                       CsRuntimeProbeSnapshot** out_snapshot,
                                       char* error_buffer,
                                       size_t error_buffer_size) {
    char mapping_name[128];
    HANDLE mapping_handle;
    CsRuntimeProbeSnapshot* snapshot;

    if (!cs_state_build_probe_mapping_name(process_id, mapping_name, sizeof(mapping_name), error_buffer, error_buffer_size)) {
        return 0;
    }

    mapping_handle = OpenFileMappingA(FILE_MAP_READ, FALSE, mapping_name);
    if (mapping_handle == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open runtime probe mapping %s.", mapping_name);
        return 0;
    }

    snapshot = (CsRuntimeProbeSnapshot*)MapViewOfFile(mapping_handle, FILE_MAP_READ, 0U, 0U, sizeof(CsRuntimeProbeSnapshot));
    if (snapshot == NULL) {
        CloseHandle(mapping_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to map runtime probe shared memory.");
        return 0;
    }

    *out_mapping_handle = mapping_handle;
    *out_snapshot = snapshot;
    return 1;
}

static int cs_state_wait_for_probe_mapping(DWORD process_id,
                                           HANDLE* out_mapping_handle,
                                           CsRuntimeProbeSnapshot** out_snapshot,
                                           char* error_buffer,
                                           size_t error_buffer_size) {
    int attempt;

    for (attempt = 0; attempt < 40; ++attempt) {
        if (cs_state_open_probe_mapping(process_id, out_mapping_handle, out_snapshot, error_buffer, error_buffer_size)) {
            return 1;
        }
        Sleep(100U);
    }
    return 0;
}

static int cs_state_read_probe_snapshot(const CsRuntimeProbeSnapshot* shared_snapshot,
                                        CsRuntimeProbeSnapshot* local_snapshot,
                                        char* error_buffer,
                                        size_t error_buffer_size) {
    unsigned int sequence_before;
    unsigned int sequence_after;
    int attempt;

    for (attempt = 0; attempt < 8; ++attempt) {
        sequence_before = shared_snapshot->sequence;
        if ((sequence_before & 1U) != 0U) {
            Sleep(1U);
            continue;
        }

        *local_snapshot = *shared_snapshot;
        sequence_after = shared_snapshot->sequence;
        if (sequence_before == sequence_after && (sequence_after & 1U) == 0U) {
            return 1;
        }
    }

    cs_tool_set_error(error_buffer, error_buffer_size, "Failed to read a stable runtime probe snapshot.");
    return 0;
}
#endif

static int cs_state_command_start(const CsStateTraceCommand* command, char* error_buffer, size_t error_buffer_size) {
#ifdef _WIN32
    char session_path[CS_TOOL_MAX_PATH];
    char state_path[CS_TOOL_MAX_PATH];
    char trace_path[CS_TOOL_MAX_PATH];
    char stop_path[CS_TOOL_MAX_PATH];
    char exe_path[CS_TOOL_MAX_PATH];
    char start_time[64];
    char end_time[64];
    CsCaptureOptions options;
    CsCaptureState state;
    DWORD process_id;

    memset(&options, 0, sizeof(options));
    memset(&state, 0, sizeof(state));
    memset(start_time, 0, sizeof(start_time));
    memset(end_time, 0, sizeof(end_time));
    process_id = 0U;

    if (!cs_state_validate_common(command, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_get_session_file_path(command->session_root, command->session_id, "session.json", session_path, sizeof(session_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "capture_state.json", state_path, sizeof(state_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "state_trace.jsonl", trace_path, sizeof(trace_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "state_trace.stop", stop_path, sizeof(stop_path))) {
        cs_tool_set_error(error_buffer, error_buffer_size, "State trace path is too long.");
        return 0;
    }

    if (!cs_tool_read_session_json(session_path,
                                   &options,
                                   start_time,
                                   sizeof(start_time),
                                   end_time,
                                   sizeof(end_time),
                                   error_buffer,
                                   error_buffer_size) ||
        !cs_state_reload_capture_state(state_path, &state, error_buffer, error_buffer_size) ||
        !cs_state_find_game_process(&process_id, error_buffer, error_buffer_size) ||
        !cs_state_inject_probe_dll(process_id, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_file_exists(trace_path) && !cs_tool_write_text_file_atomic(trace_path, "", error_buffer, error_buffer_size)) {
        return 0;
    }

    if (cs_tool_file_exists(stop_path)) {
        (void)remove(stop_path);
    }

    if (strcmp(state.state_trace_status, "starting") == 0 || strcmp(state.state_trace_status, "running") == 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "State trace is already running for session %s.", command->session_id);
        return 0;
    }

    if (state.session_id[0] == '\0') {
        (void)cs_tool_copy_string(state.session_id, sizeof(state.session_id), command->session_id);
    }
    if (state.teacher_source[0] == '\0') {
        (void)cs_tool_copy_string(state.teacher_source, sizeof(state.teacher_source), options.teacher_source);
    }
    (void)cs_tool_copy_string(state.state_trace_status, sizeof(state.state_trace_status), "starting");

    if (!cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_get_executable_path(exe_path, sizeof(exe_path), error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_spawn_state_trace_worker(exe_path,
                                          command->session_root,
                                          command->session_id,
                                          error_buffer,
                                          error_buffer_size)) {
        return 0;
    }

    printf("Started state trace for session %s\n", command->session_id);
    return 1;
#else
    (void)command;
    cs_tool_set_error(error_buffer, error_buffer_size, "State trace is only implemented on Windows.");
    return 0;
#endif
}

static int cs_state_command_stop(const CsStateTraceCommand* command, char* error_buffer, size_t error_buffer_size) {
    char stop_path[CS_TOOL_MAX_PATH];
    char stop_text[64];

    if (!cs_state_validate_common(command, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_get_session_file_path(command->session_root, command->session_id, "state_trace.stop", stop_path, sizeof(stop_path))) {
        cs_tool_set_error(error_buffer, error_buffer_size, "State trace stop path is too long.");
        return 0;
    }

    cs_tool_now_iso8601(stop_text, sizeof(stop_text));
    if (!cs_tool_write_text_file_atomic(stop_path, stop_text, error_buffer, error_buffer_size)) {
        return 0;
    }

    printf("Requested state trace stop for session %s\n", command->session_id);
    return 1;
}

static int cs_state_command_status(const CsStateTraceCommand* command, char* error_buffer, size_t error_buffer_size) {
    char state_path[CS_TOOL_MAX_PATH];
    CsCaptureState state;

    memset(&state, 0, sizeof(state));
    if (!cs_state_validate_common(command, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_get_session_file_path(command->session_root, command->session_id, "capture_state.json", state_path, sizeof(state_path))) {
        cs_tool_set_error(error_buffer, error_buffer_size, "State path is too long.");
        return 0;
    }

    if (!cs_tool_read_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
        return 0;
    }

    printf("session_id=%s\n", state.session_id);
    printf("capture_status=%s\n", state.status);
    printf("state_trace_status=%s\n", state.state_trace_status);
    printf("teacher_source=%s\n", state.teacher_source);
    printf("state_trace_count=%d\n", state.state_trace_count);
    printf("last_state_timestamp=%s\n", state.last_state_timestamp);
    printf("captured_frame_count=%d\n", state.captured_frame_count);
    printf("last_frame_index=%d\n", state.last_frame_index);

#ifdef _WIN32
    {
        DWORD process_id;
        HANDLE mapping_handle;
        CsRuntimeProbeSnapshot* shared_snapshot;
        CsRuntimeProbeSnapshot snapshot;

        process_id = 0U;
        mapping_handle = NULL;
        shared_snapshot = NULL;
        memset(&snapshot, 0, sizeof(snapshot));

        if (cs_state_find_game_process(&process_id, error_buffer, error_buffer_size) &&
            cs_state_open_probe_mapping(process_id, &mapping_handle, &shared_snapshot, error_buffer, error_buffer_size) &&
            cs_state_read_probe_snapshot(shared_snapshot, &snapshot, error_buffer, error_buffer_size)) {
            printf("runtime_probe_status=%s\n", snapshot.status);
            printf("runtime_probe_has_pose=%d\n", snapshot.has_pose);
            printf("runtime_probe_timestamp=%s\n", snapshot.timestamp);
            printf("runtime_probe_error=%s\n", snapshot.last_error);
        }

        if (shared_snapshot != NULL) {
            (void)UnmapViewOfFile(shared_snapshot);
        }
        if (mapping_handle != NULL) {
            CloseHandle(mapping_handle);
        }
    }
#endif

    return 1;
}

static int cs_state_worker_loop(const CsStateTraceCommand* command, char* error_buffer, size_t error_buffer_size) {
#ifdef _WIN32
    char session_path[CS_TOOL_MAX_PATH];
    char state_path[CS_TOOL_MAX_PATH];
    char trace_path[CS_TOOL_MAX_PATH];
    char stop_path[CS_TOOL_MAX_PATH];
    char start_time[64];
    char end_time[64];
    int next_frame_index;
    CsCaptureOptions options;
    CsCaptureState state;
    DWORD process_id;
    HANDLE mapping_handle;
    CsRuntimeProbeSnapshot* shared_snapshot;
    CsRuntimeProbeSnapshot snapshot;

    memset(&options, 0, sizeof(options));
    memset(&state, 0, sizeof(state));
    memset(start_time, 0, sizeof(start_time));
    memset(end_time, 0, sizeof(end_time));
    memset(&snapshot, 0, sizeof(snapshot));
    process_id = 0U;
    mapping_handle = NULL;
    shared_snapshot = NULL;

    if (!cs_tool_get_session_file_path(command->session_root, command->session_id, "session.json", session_path, sizeof(session_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "capture_state.json", state_path, sizeof(state_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "state_trace.jsonl", trace_path, sizeof(trace_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "state_trace.stop", stop_path, sizeof(stop_path))) {
        cs_tool_set_error(error_buffer, error_buffer_size, "State trace worker path is too long.");
        return 0;
    }

    if (!cs_tool_read_session_json(session_path,
                                   &options,
                                   start_time,
                                   sizeof(start_time),
                                   end_time,
                                   sizeof(end_time),
                                   error_buffer,
                                   error_buffer_size) ||
        !cs_state_reload_capture_state(state_path, &state, error_buffer, error_buffer_size) ||
        !cs_state_count_existing_records(trace_path, &next_frame_index, error_buffer, error_buffer_size) ||
        !cs_state_find_game_process(&process_id, error_buffer, error_buffer_size) ||
        !cs_state_inject_probe_dll(process_id, error_buffer, error_buffer_size) ||
        !cs_state_wait_for_probe_mapping(process_id, &mapping_handle, &shared_snapshot, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (state.teacher_source[0] == '\0') {
        (void)cs_tool_copy_string(state.teacher_source, sizeof(state.teacher_source), options.teacher_source);
    }
    (void)cs_tool_copy_string(state.state_trace_status, sizeof(state.state_trace_status), "running");
    state.state_trace_count = next_frame_index;

    if (!cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
        (void)UnmapViewOfFile(shared_snapshot);
        CloseHandle(mapping_handle);
        return 0;
    }

    for (;;) {
        int current_last_frame_index;
        int capture_stopped;

        if (!cs_state_reload_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
            (void)UnmapViewOfFile(shared_snapshot);
            CloseHandle(mapping_handle);
            return 0;
        }

        current_last_frame_index = state.last_frame_index;
        capture_stopped = (strcmp(state.status, "stopped") == 0 || strcmp(state.status, "error") == 0);

        if (!cs_state_read_probe_snapshot(shared_snapshot, &snapshot, error_buffer, error_buffer_size)) {
            (void)cs_tool_copy_string(state.state_trace_status, sizeof(state.state_trace_status), "error");
            (void)cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size);
            (void)UnmapViewOfFile(shared_snapshot);
            CloseHandle(mapping_handle);
            return 0;
        }

        if (!snapshot.has_pose) {
            (void)cs_tool_copy_string(state.state_trace_status, sizeof(state.state_trace_status), snapshot.status[0] != '\0' ? snapshot.status : "waiting_probe");
            (void)cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size);
        } else {
            while (next_frame_index <= current_last_frame_index) {
                CsStateTraceRecord record;

                memset(&record, 0, sizeof(record));
                record.frame_index = next_frame_index;
                (void)cs_tool_copy_string(record.timestamp, sizeof(record.timestamp), snapshot.timestamp);
                record.pos_x = (double)snapshot.pos_x;
                record.pos_y = (double)snapshot.pos_y;
                record.pos_z = (double)snapshot.pos_z;
                record.yaw = (double)snapshot.yaw;
                record.pitch = (double)snapshot.pitch;
                record.velocity_x = (double)snapshot.velocity_x;
                record.velocity_y = (double)snapshot.velocity_y;
                record.velocity_z = (double)snapshot.velocity_z;

                if (!cs_tool_append_state_trace_record(trace_path, &record, error_buffer, error_buffer_size)) {
                    if (cs_state_reload_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
                        (void)cs_tool_copy_string(state.state_trace_status, sizeof(state.state_trace_status), "error");
                        (void)cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size);
                    }
                    (void)UnmapViewOfFile(shared_snapshot);
                    CloseHandle(mapping_handle);
                    return 0;
                }

                if (!cs_state_reload_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
                    (void)UnmapViewOfFile(shared_snapshot);
                    CloseHandle(mapping_handle);
                    return 0;
                }

                state.state_trace_count = next_frame_index + 1;
                (void)cs_tool_copy_string(state.last_state_timestamp, sizeof(state.last_state_timestamp), record.timestamp);
                (void)cs_tool_copy_string(state.state_trace_status, sizeof(state.state_trace_status), snapshot.status[0] != '\0' ? snapshot.status : "running");
                if (state.teacher_source[0] == '\0') {
                    (void)cs_tool_copy_string(state.teacher_source, sizeof(state.teacher_source), options.teacher_source);
                }

                if (!cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
                    (void)UnmapViewOfFile(shared_snapshot);
                    CloseHandle(mapping_handle);
                    return 0;
                }

                next_frame_index++;
            }
        }

        if (cs_tool_file_exists(stop_path) || capture_stopped) {
            break;
        }

        Sleep(50U);
    }

    if (!cs_state_reload_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
        (void)UnmapViewOfFile(shared_snapshot);
        CloseHandle(mapping_handle);
        return 0;
    }

    (void)cs_tool_copy_string(state.state_trace_status, sizeof(state.state_trace_status), "stopped");
    (void)UnmapViewOfFile(shared_snapshot);
    CloseHandle(mapping_handle);
    return cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size);
#else
    (void)command;
    cs_tool_set_error(error_buffer, error_buffer_size, "State trace worker is only implemented on Windows.");
    return 0;
#endif
}

static void cs_state_print_usage(void) {
    printf("Usage:\n");
    printf("  cs_state_trace start --session-id <id> [--session-root <path>]\n");
    printf("  cs_state_trace stop --session-id <id> [--session-root <path>]\n");
    printf("  cs_state_trace status --session-id <id> [--session-root <path>]\n");
}

int main(int argc, char** argv) {
    CsStateTraceCommand command;
    char error_buffer[CS_TOOL_MAX_TEXT];
    const char* subcommand;
    int ok;

    memset(&command, 0, sizeof(command));
    memset(error_buffer, 0, sizeof(error_buffer));

    if (argc < 2) {
        cs_state_print_usage();
        return 1;
    }

    subcommand = argv[1];
    if (!cs_state_parse_options(argc, argv, 2, &command, error_buffer, sizeof(error_buffer))) {
        fprintf(stderr, "%s\n", error_buffer);
        return 1;
    }

    if (strcmp(subcommand, "start") == 0) {
        ok = cs_state_command_start(&command, error_buffer, sizeof(error_buffer));
    } else if (strcmp(subcommand, "stop") == 0) {
        ok = cs_state_command_stop(&command, error_buffer, sizeof(error_buffer));
    } else if (strcmp(subcommand, "status") == 0) {
        ok = cs_state_command_status(&command, error_buffer, sizeof(error_buffer));
    } else if (strcmp(subcommand, "__state_trace_loop") == 0) {
        ok = cs_state_worker_loop(&command, error_buffer, sizeof(error_buffer));
    } else {
        cs_state_print_usage();
        return 1;
    }

    if (!ok) {
        fprintf(stderr, "%s\n", error_buffer);
        return 2;
    }

    return 0;
}
