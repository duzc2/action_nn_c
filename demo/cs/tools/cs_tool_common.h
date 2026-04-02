#ifndef ACTION_C_DEMO_CS_TOOL_COMMON_H
#define ACTION_C_DEMO_CS_TOOL_COMMON_H

/*
 * Shared helper declarations for the Version 1 CS demo tools.
 *
 * The current implementation only covers the first three foundational tasks:
 * 1. fixed place dictionary;
 * 2. capture session tool;
 * 3. label session tool.
 *
 * The helpers stay intentionally small and explicit so they are easy to
 * inspect, test and evolve during the early versions of the demo.
 */

#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define CS_TOOL_MAX_PATH 1024
#define CS_TOOL_MAX_TEXT 256
#define CS_TOOL_MAX_PLACES 64
#define CS_TOOL_MAX_SEGMENTS 2048

typedef struct CsPlaceEntryTag {
    int place_id;
    char place_token[64];
    char display_name[64];
    int enabled_in_v1;
    char notes[128];
} CsPlaceEntry;

typedef struct CsPlaceDictionaryTag {
    char dictionary_name[64];
    char map_name[64];
    int version;
    CsPlaceEntry entries[CS_TOOL_MAX_PLACES];
    size_t entry_count;
} CsPlaceDictionary;

typedef struct CsLabelSegmentTag {
    int segment_id;
    int start_frame;
    int end_frame;
    int place_id;
    char place_token[64];
} CsLabelSegment;

typedef struct CsLabelSegmentsFileTag {
    char session_id[64];
    CsLabelSegment segments[CS_TOOL_MAX_SEGMENTS];
    size_t segment_count;
} CsLabelSegmentsFile;

typedef struct CsCaptureStateTag {
    char session_id[64];
    char status[32];
    int captured_frame_count;
    int last_frame_index;
    char current_place_token[64];
    int current_place_id;
} CsCaptureState;

typedef struct CsCaptureOptionsTag {
    char session_root[CS_TOOL_MAX_PATH];
    char session_id[64];
    char map_name[64];
    int width;
    int height;
    int capture_fps;
    char team[32];
    char notes[128];
} CsCaptureOptions;

void cs_tool_set_error(char* buffer, size_t buffer_size, const char* fmt, ...);
int cs_tool_copy_string(char* dst, size_t dst_size, const char* src);
int cs_tool_join_path(char* out_path, size_t out_size, const char* lhs, const char* rhs);
int cs_tool_join_two_paths(char* out_path, size_t out_size, const char* a, const char* b, const char* c);
int cs_tool_make_dirs(const char* path, char* error_buffer, size_t error_buffer_size);
int cs_tool_file_exists(const char* path);
int cs_tool_read_text_file(const char* path, char** out_text, size_t* out_size, char* error_buffer, size_t error_buffer_size);
int cs_tool_write_text_file_atomic(const char* path, const char* text, char* error_buffer, size_t error_buffer_size);
int cs_tool_copy_file(const char* src_path, const char* dst_path, char* error_buffer, size_t error_buffer_size);
int cs_tool_copy_binary_file(const char* src_path, const char* dst_path, char* error_buffer, size_t error_buffer_size);
void cs_tool_free_text(char* text);
void cs_tool_now_iso8601(char* buffer, size_t buffer_size);
int cs_tool_get_executable_path(char* out_path, size_t out_size, char* error_buffer, size_t error_buffer_size);
int cs_tool_get_current_directory(char* out_path, size_t out_size, char* error_buffer, size_t error_buffer_size);

int cs_tool_get_session_dir(const char* session_root, const char* session_id, char* out_path, size_t out_size);
int cs_tool_get_frames_dir(const char* session_root, const char* session_id, char* out_path, size_t out_size);
int cs_tool_get_session_file_path(const char* session_root, const char* session_id, const char* file_name, char* out_path, size_t out_size);

int cs_tool_find_default_dictionary(char* out_path, size_t out_size, char* error_buffer, size_t error_buffer_size);
int cs_tool_load_dictionary(const char* path, CsPlaceDictionary* dictionary, char* error_buffer, size_t error_buffer_size);
const CsPlaceEntry* cs_tool_find_place_by_token(const CsPlaceDictionary* dictionary, const char* token);

int cs_tool_write_capture_state(const char* path, const CsCaptureState* state, char* error_buffer, size_t error_buffer_size);
int cs_tool_read_capture_state(const char* path, CsCaptureState* state, char* error_buffer, size_t error_buffer_size);

int cs_tool_write_label_segments(const char* path, const CsLabelSegmentsFile* segments, char* error_buffer, size_t error_buffer_size);
int cs_tool_read_label_segments(const char* path, CsLabelSegmentsFile* segments, char* error_buffer, size_t error_buffer_size);

int cs_tool_write_session_json(const char* path, const CsCaptureOptions* options, const char* start_time, const char* end_time, char* error_buffer, size_t error_buffer_size);
int cs_tool_read_session_json(const char* path,
                              CsCaptureOptions* options,
                              char* start_time,
                              size_t start_time_size,
                              char* end_time,
                              size_t end_time_size,
                              char* error_buffer,
                              size_t error_buffer_size);

#ifdef _WIN32
HWND cs_tool_find_counter_strike_window(void);
int cs_tool_get_window_client_size(HWND hwnd, int* out_width, int* out_height, char* error_buffer, size_t error_buffer_size);
int cs_tool_capture_window_client_to_bmp(HWND hwnd, const char* output_path, int expected_width, int expected_height, char* error_buffer, size_t error_buffer_size);
int cs_tool_spawn_capture_worker(const char* exe_path, const char* session_root, const char* session_id, int capture_fps, int width, int height, char* error_buffer, size_t error_buffer_size);
#endif

#endif
