#include "cs_tool_common.h"

/*
 * The helper implementation deliberately uses only small, explicit building
 * blocks. That keeps the Version 1 tools readable and keeps later debugging
 * focused on the actual data pipeline instead of hidden framework behavior.
 */

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#endif

void cs_tool_set_error(char* buffer, size_t buffer_size, const char* fmt, ...) {
    va_list args;

    if (buffer == NULL || buffer_size == 0U) {
        return;
    }

    buffer[0] = '\0';
    va_start(args, fmt);
    (void)vsnprintf(buffer, buffer_size, fmt, args);
    va_end(args);
}

int cs_tool_copy_string(char* dst, size_t dst_size, const char* src) {
    int written_size;

    if (dst == NULL || dst_size == 0U || src == NULL) {
        return 0;
    }

    written_size = snprintf(dst, dst_size, "%s", src);
    if (written_size < 0 || (size_t)written_size >= dst_size) {
        dst[0] = '\0';
        return 0;
    }

    return 1;
}

int cs_tool_join_path(char* out_path, size_t out_size, const char* lhs, const char* rhs) {
    int written_size;

    if (out_path == NULL || out_size == 0U || lhs == NULL || rhs == NULL) {
        return 0;
    }

    written_size = snprintf(out_path, out_size, "%s\\%s", lhs, rhs);
    if (written_size < 0 || (size_t)written_size >= out_size) {
        out_path[0] = '\0';
        return 0;
    }

    return 1;
}

int cs_tool_join_two_paths(char* out_path, size_t out_size, const char* a, const char* b, const char* c) {
    char temp_path[CS_TOOL_MAX_PATH];

    if (!cs_tool_join_path(temp_path, sizeof(temp_path), a, b)) {
        return 0;
    }

    return cs_tool_join_path(out_path, out_size, temp_path, c);
}

int cs_tool_make_dirs(const char* path, char* error_buffer, size_t error_buffer_size) {
    char temp[CS_TOOL_MAX_PATH];
    size_t index;
    size_t length;

    if (path == NULL || path[0] == '\0') {
        cs_tool_set_error(error_buffer, error_buffer_size, "Directory path is empty.");
        return 0;
    }

    if (!cs_tool_copy_string(temp, sizeof(temp), path)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Directory path is too long: %s", path);
        return 0;
    }

    length = strlen(temp);
    for (index = 0U; index < length; ++index) {
        if (temp[index] == '\\' || temp[index] == '/') {
            char saved_character;

            if (index == 2U && temp[1] == ':') {
                continue;
            }

            saved_character = temp[index];
            temp[index] = '\0';
            if (temp[0] != '\0') {
#ifdef _WIN32
                if (_mkdir(temp) != 0 && errno != EEXIST) {
                    cs_tool_set_error(error_buffer, error_buffer_size, "Failed to create directory: %s", temp);
                    return 0;
                }
#endif
            }
            temp[index] = saved_character;
        }
    }

#ifdef _WIN32
    if (_mkdir(temp) != 0 && errno != EEXIST) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to create directory: %s", temp);
        return 0;
    }
#endif

    return 1;
}

int cs_tool_file_exists(const char* path) {
    struct stat stat_buffer;

    if (path == NULL || path[0] == '\0') {
        return 0;
    }

    return stat(path, &stat_buffer) == 0;
}

int cs_tool_read_text_file(const char* path, char** out_text, size_t* out_size, char* error_buffer, size_t error_buffer_size) {
    FILE* file_handle;
    char* text_buffer;
    long file_size;
    size_t read_size;

    if (out_text == NULL || out_size == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Output buffer pointers are invalid.");
        return 0;
    }

    *out_text = NULL;
    *out_size = 0U;

    file_handle = fopen(path, "rb");
    if (file_handle == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open file for reading: %s", path);
        return 0;
    }

    if (fseek(file_handle, 0L, SEEK_END) != 0) {
        fclose(file_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to seek file: %s", path);
        return 0;
    }

    file_size = ftell(file_handle);
    if (file_size < 0L) {
        fclose(file_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to get file size: %s", path);
        return 0;
    }

    if (fseek(file_handle, 0L, SEEK_SET) != 0) {
        fclose(file_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to rewind file: %s", path);
        return 0;
    }

    text_buffer = (char*)malloc((size_t)file_size + 1U);
    if (text_buffer == NULL) {
        fclose(file_handle);
        cs_tool_set_error(error_buffer, error_buffer_size, "Out of memory while reading: %s", path);
        return 0;
    }

    read_size = fread(text_buffer, 1U, (size_t)file_size, file_handle);
    fclose(file_handle);

    if (read_size != (size_t)file_size) {
        free(text_buffer);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to read complete file: %s", path);
        return 0;
    }

    text_buffer[read_size] = '\0';
    *out_text = text_buffer;
    *out_size = read_size;
    return 1;
}

int cs_tool_write_text_file_atomic(const char* path, const char* text, char* error_buffer, size_t error_buffer_size) {
    char temp_path[CS_TOOL_MAX_PATH];
    char parent_dir[CS_TOOL_MAX_PATH];
    const char* last_slash;
    FILE* file_handle;

    if (path == NULL || text == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Path or text is null.");
        return 0;
    }

    last_slash = strrchr(path, '\\');
    if (last_slash != NULL) {
        size_t parent_size;

        parent_size = (size_t)(last_slash - path);
        if (parent_size >= sizeof(parent_dir)) {
            cs_tool_set_error(error_buffer, error_buffer_size, "Parent path is too long: %s", path);
            return 0;
        }

        memcpy(parent_dir, path, parent_size);
        parent_dir[parent_size] = '\0';
        if (!cs_tool_make_dirs(parent_dir, error_buffer, error_buffer_size)) {
            return 0;
        }
    }

    if (!cs_tool_copy_string(temp_path, sizeof(temp_path), path)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Target path is too long: %s", path);
        return 0;
    }

    if (strlen(temp_path) + 5U >= sizeof(temp_path)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Temporary path is too long: %s", path);
        return 0;
    }
    strcat(temp_path, ".tmp");

    file_handle = fopen(temp_path, "wb");
    if (file_handle == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open file for writing: %s", temp_path);
        return 0;
    }

    if (fputs(text, file_handle) < 0) {
        fclose(file_handle);
        remove(temp_path);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to write file: %s", temp_path);
        return 0;
    }

    if (fclose(file_handle) != 0) {
        remove(temp_path);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to close file: %s", temp_path);
        return 0;
    }

#ifdef _WIN32
    if (!MoveFileExA(temp_path, path, MOVEFILE_REPLACE_EXISTING)) {
        remove(temp_path);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to replace file: %s", path);
        return 0;
    }
#endif

    return 1;
}

int cs_tool_copy_file(const char* src_path, const char* dst_path, char* error_buffer, size_t error_buffer_size) {
    char* text;
    size_t text_size;

    text = NULL;
    text_size = 0U;

    if (!cs_tool_read_text_file(src_path, &text, &text_size, error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)text_size;

    if (!cs_tool_write_text_file_atomic(dst_path, text, error_buffer, error_buffer_size)) {
        free(text);
        return 0;
    }

    free(text);
    return 1;
}

int cs_tool_copy_binary_file(const char* src_path, const char* dst_path, char* error_buffer, size_t error_buffer_size) {
#ifdef _WIN32
    char parent_dir[CS_TOOL_MAX_PATH];
    const char* last_slash;

    last_slash = strrchr(dst_path, '\\');
    if (last_slash != NULL) {
        size_t parent_size;

        parent_size = (size_t)(last_slash - dst_path);
        if (parent_size >= sizeof(parent_dir)) {
            cs_tool_set_error(error_buffer, error_buffer_size, "Destination parent path is too long: %s", dst_path);
            return 0;
        }

        memcpy(parent_dir, dst_path, parent_size);
        parent_dir[parent_size] = '\0';
        if (!cs_tool_make_dirs(parent_dir, error_buffer, error_buffer_size)) {
            return 0;
        }
    }

    if (!CopyFileA(src_path, dst_path, FALSE)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to copy binary file from %s to %s", src_path, dst_path);
        return 0;
    }
    return 1;
#else
    FILE* src;
    FILE* dst;
    unsigned char buffer[8192];
    size_t read_size;

    src = fopen(src_path, "rb");
    if (src == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open binary source: %s", src_path);
        return 0;
    }

    dst = fopen(dst_path, "wb");
    if (dst == NULL) {
        fclose(src);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open binary destination: %s", dst_path);
        return 0;
    }

    for (;;) {
        read_size = fread(buffer, 1U, sizeof(buffer), src);
        if (read_size > 0U && fwrite(buffer, 1U, read_size, dst) != read_size) {
            fclose(src);
            fclose(dst);
            cs_tool_set_error(error_buffer, error_buffer_size, "Failed to write binary destination: %s", dst_path);
            return 0;
        }
        if (read_size < sizeof(buffer)) {
            break;
        }
    }

    fclose(src);
    fclose(dst);
    return 1;
#endif
}

void cs_tool_free_text(char* text) {
    free(text);
}

void cs_tool_now_iso8601(char* buffer, size_t buffer_size) {
    time_t current_time;
    struct tm local_time_info;

    if (buffer == NULL || buffer_size == 0U) {
        return;
    }

    current_time = time(NULL);
#ifdef _WIN32
    (void)localtime_s(&local_time_info, &current_time);
#else
    local_time_info = *localtime(&current_time);
#endif

    (void)snprintf(buffer,
                   buffer_size,
                   "%04d-%02d-%02dT%02d:%02d:%02d",
                   local_time_info.tm_year + 1900,
                   local_time_info.tm_mon + 1,
                   local_time_info.tm_mday,
                   local_time_info.tm_hour,
                   local_time_info.tm_min,
                   local_time_info.tm_sec);
}

int cs_tool_get_executable_path(char* out_path, size_t out_size, char* error_buffer, size_t error_buffer_size) {
#ifdef _WIN32
    DWORD copied_size;

    copied_size = GetModuleFileNameA(NULL, out_path, (DWORD)out_size);
    if (copied_size == 0U || copied_size >= (DWORD)out_size) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to get executable path.");
        return 0;
    }
    return 1;
#else
    (void)out_path;
    (void)out_size;
    cs_tool_set_error(error_buffer, error_buffer_size, "Executable path helper is only implemented on Windows.");
    return 0;
#endif
}

int cs_tool_get_current_directory(char* out_path, size_t out_size, char* error_buffer, size_t error_buffer_size) {
#ifdef _WIN32
    DWORD copied_size;

    copied_size = GetCurrentDirectoryA((DWORD)out_size, out_path);
    if (copied_size == 0U || copied_size >= (DWORD)out_size) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to get current directory.");
        return 0;
    }
    return 1;
#else
    (void)out_path;
    (void)out_size;
    cs_tool_set_error(error_buffer, error_buffer_size, "Current directory helper is only implemented on Windows.");
    return 0;
#endif
}

int cs_tool_get_session_dir(const char* session_root, const char* session_id, char* out_path, size_t out_size) {
    return cs_tool_join_path(out_path, out_size, session_root, session_id);
}

int cs_tool_get_frames_dir(const char* session_root, const char* session_id, char* out_path, size_t out_size) {
    return cs_tool_join_two_paths(out_path, out_size, session_root, session_id, "frames");
}

int cs_tool_get_session_file_path(const char* session_root, const char* session_id, const char* file_name, char* out_path, size_t out_size) {
    return cs_tool_join_two_paths(out_path, out_size, session_root, session_id, file_name);
}

static const char* cs_tool_skip_spaces(const char* cursor) {
    while (cursor != NULL &&
           *cursor != '\0' &&
           (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t')) {
        cursor++;
    }
    return cursor;
}

static const char* cs_tool_find_key(const char* text, const char* key) {
    char pattern[128];
    int written_size;

    written_size = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (written_size < 0 || (size_t)written_size >= sizeof(pattern)) {
        return NULL;
    }

    return strstr(text, pattern);
}

static int cs_tool_extract_string_after_key(const char* text, const char* key, char* out_value, size_t out_size) {
    const char* key_position;
    const char* colon;
    const char* first_quote;
    const char* second_quote;
    size_t value_length;

    key_position = cs_tool_find_key(text, key);
    if (key_position == NULL) {
        return 0;
    }

    colon = strchr(key_position, ':');
    if (colon == NULL) {
        return 0;
    }

    first_quote = strchr(colon, '"');
    if (first_quote == NULL) {
        return 0;
    }

    second_quote = strchr(first_quote + 1, '"');
    if (second_quote == NULL) {
        return 0;
    }

    value_length = (size_t)(second_quote - (first_quote + 1));
    if (value_length + 1U > out_size) {
        return 0;
    }

    memcpy(out_value, first_quote + 1, value_length);
    out_value[value_length] = '\0';
    return 1;
}

static int cs_tool_extract_int_after_key(const char* text, const char* key, int* out_value) {
    const char* key_position;
    const char* colon;
    const char* value_position;
    int scanned_value;

    key_position = cs_tool_find_key(text, key);
    if (key_position == NULL) {
        return 0;
    }

    colon = strchr(key_position, ':');
    if (colon == NULL) {
        return 0;
    }

    value_position = cs_tool_skip_spaces(colon + 1);
    if (value_position == NULL) {
        return 0;
    }

    if (sscanf(value_position, "%d", &scanned_value) != 1) {
        return 0;
    }

    *out_value = scanned_value;
    return 1;
}

static int cs_tool_extract_bool_after_key(const char* text, const char* key, int* out_value) {
    const char* key_position;
    const char* colon;
    const char* value_position;

    key_position = cs_tool_find_key(text, key);
    if (key_position == NULL) {
        return 0;
    }

    colon = strchr(key_position, ':');
    if (colon == NULL) {
        return 0;
    }

    value_position = cs_tool_skip_spaces(colon + 1);
    if (value_position == NULL) {
        return 0;
    }

    if (strncmp(value_position, "true", 4U) == 0) {
        *out_value = 1;
        return 1;
    }
    if (strncmp(value_position, "false", 5U) == 0) {
        *out_value = 0;
        return 1;
    }
    return 0;
}

int cs_tool_find_default_dictionary(char* out_path, size_t out_size, char* error_buffer, size_t error_buffer_size) {
    char current_directory[CS_TOOL_MAX_PATH];
    char executable_directory[CS_TOOL_MAX_PATH];
    char probe_base[CS_TOOL_MAX_PATH];
    char* roots[2];
    size_t root_index;

    if (!cs_tool_get_current_directory(current_directory, sizeof(current_directory), error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_get_executable_path(executable_directory, sizeof(executable_directory), error_buffer, error_buffer_size)) {
        return 0;
    }

    {
        char* last_slash;

        last_slash = strrchr(executable_directory, '\\');
        if (last_slash != NULL) {
            *last_slash = '\0';
        }
    }

    roots[0] = current_directory;
    roots[1] = executable_directory;

    for (root_index = 0U; root_index < 2U; ++root_index) {
        int level;

        if (!cs_tool_copy_string(probe_base, sizeof(probe_base), roots[root_index])) {
            continue;
        }

        for (level = 0; level < 8; ++level) {
            char candidate[CS_TOOL_MAX_PATH];

            if (snprintf(candidate,
                         sizeof(candidate),
                         "%s\\demo\\cs\\config\\place_dictionary.json",
                         probe_base) >= 0 &&
                cs_tool_file_exists(candidate)) {
                return cs_tool_copy_string(out_path, out_size, candidate);
            }

            {
                char* last_slash;

                last_slash = strrchr(probe_base, '\\');
                if (last_slash == NULL) {
                    break;
                }
                *last_slash = '\0';
            }
        }
    }

    cs_tool_set_error(error_buffer, error_buffer_size, "Failed to locate demo/cs/config/place_dictionary.json");
    return 0;
}

int cs_tool_load_dictionary(const char* path, CsPlaceDictionary* dictionary, char* error_buffer, size_t error_buffer_size) {
    char* text;
    size_t text_size;
    const char* cursor;

    if (dictionary == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Dictionary output is null.");
        return 0;
    }

    memset(dictionary, 0, sizeof(*dictionary));
    text = NULL;
    text_size = 0U;

    if (!cs_tool_read_text_file(path, &text, &text_size, error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)text_size;

    if (!cs_tool_extract_string_after_key(text, "dictionary_name", dictionary->dictionary_name, sizeof(dictionary->dictionary_name)) ||
        !cs_tool_extract_string_after_key(text, "map_name", dictionary->map_name, sizeof(dictionary->map_name)) ||
        !cs_tool_extract_int_after_key(text, "version", &dictionary->version)) {
        free(text);
        cs_tool_set_error(error_buffer, error_buffer_size, "Dictionary file is missing top-level fields: %s", path);
        return 0;
    }

    cursor = text;
    while ((cursor = cs_tool_find_key(cursor, "place_id")) != NULL) {
        CsPlaceEntry* entry;

        if (dictionary->entry_count >= CS_TOOL_MAX_PLACES) {
            free(text);
            cs_tool_set_error(error_buffer, error_buffer_size, "Dictionary has too many entries: %s", path);
            return 0;
        }

        entry = &dictionary->entries[dictionary->entry_count];
        memset(entry, 0, sizeof(*entry));

        if (!cs_tool_extract_int_after_key(cursor, "place_id", &entry->place_id) ||
            !cs_tool_extract_string_after_key(cursor, "place_token", entry->place_token, sizeof(entry->place_token)) ||
            !cs_tool_extract_string_after_key(cursor, "display_name", entry->display_name, sizeof(entry->display_name)) ||
            !cs_tool_extract_bool_after_key(cursor, "enabled_in_v1", &entry->enabled_in_v1) ||
            !cs_tool_extract_string_after_key(cursor, "notes", entry->notes, sizeof(entry->notes))) {
            free(text);
            cs_tool_set_error(error_buffer, error_buffer_size, "Dictionary entry is malformed: %s", path);
            return 0;
        }

        dictionary->entry_count++;
        cursor += 9;
    }

    free(text);

    if (dictionary->entry_count == 0U) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Dictionary contains no entries: %s", path);
        return 0;
    }

    return 1;
}

const CsPlaceEntry* cs_tool_find_place_by_token(const CsPlaceDictionary* dictionary, const char* token) {
    size_t index;

    if (dictionary == NULL || token == NULL) {
        return NULL;
    }

    for (index = 0U; index < dictionary->entry_count; ++index) {
        if (strcmp(dictionary->entries[index].place_token, token) == 0) {
            return &dictionary->entries[index];
        }
    }

    return NULL;
}

int cs_tool_write_capture_state(const char* path, const CsCaptureState* state, char* error_buffer, size_t error_buffer_size) {
    char json_text[1024];
    int written_size;

    written_size = snprintf(json_text,
                            sizeof(json_text),
                            "{\n"
                            "  \"session_id\": \"%s\",\n"
                            "  \"status\": \"%s\",\n"
                            "  \"captured_frame_count\": %d,\n"
                            "  \"last_frame_index\": %d,\n"
                            "  \"current_place_token\": \"%s\",\n"
                            "  \"current_place_id\": %d\n"
                            "}\n",
                            state->session_id,
                            state->status,
                            state->captured_frame_count,
                            state->last_frame_index,
                            state->current_place_token,
                            state->current_place_id);

    if (written_size < 0 || (size_t)written_size >= sizeof(json_text)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Capture state JSON buffer overflow.");
        return 0;
    }

    return cs_tool_write_text_file_atomic(path, json_text, error_buffer, error_buffer_size);
}

int cs_tool_read_capture_state(const char* path, CsCaptureState* state, char* error_buffer, size_t error_buffer_size) {
    char* text;
    size_t text_size;

    if (state == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Capture state output is null.");
        return 0;
    }

    memset(state, 0, sizeof(*state));
    text = NULL;
    text_size = 0U;

    if (!cs_tool_read_text_file(path, &text, &text_size, error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)text_size;

    if (!cs_tool_extract_string_after_key(text, "session_id", state->session_id, sizeof(state->session_id)) ||
        !cs_tool_extract_string_after_key(text, "status", state->status, sizeof(state->status)) ||
        !cs_tool_extract_int_after_key(text, "captured_frame_count", &state->captured_frame_count) ||
        !cs_tool_extract_int_after_key(text, "last_frame_index", &state->last_frame_index) ||
        !cs_tool_extract_string_after_key(text, "current_place_token", state->current_place_token, sizeof(state->current_place_token)) ||
        !cs_tool_extract_int_after_key(text, "current_place_id", &state->current_place_id)) {
        free(text);
        cs_tool_set_error(error_buffer, error_buffer_size, "Capture state file is malformed: %s", path);
        return 0;
    }

    free(text);
    return 1;
}

int cs_tool_write_label_segments(const char* path, const CsLabelSegmentsFile* segments, char* error_buffer, size_t error_buffer_size) {
    char* text;
    size_t buffer_size;
    size_t offset;
    size_t index;

    buffer_size = 4096U + segments->segment_count * 256U;
    text = (char*)malloc(buffer_size);
    if (text == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Out of memory while writing label segments.");
        return 0;
    }

    offset = (size_t)snprintf(text, buffer_size,
                              "{\n"
                              "  \"session_id\": \"%s\",\n"
                              "  \"segments\": [\n",
                              segments->session_id);

    for (index = 0U; index < segments->segment_count; ++index) {
        const CsLabelSegment* segment;
        int written_size;

        segment = &segments->segments[index];
        written_size = snprintf(text + offset,
                                buffer_size - offset,
                                "    {\n"
                                "      \"segment_id\": %d,\n"
                                "      \"start_frame\": %d,\n"
                                "      \"end_frame\": %d,\n"
                                "      \"place_token\": \"%s\",\n"
                                "      \"place_id\": %d\n"
                                "    }%s\n",
                                segment->segment_id,
                                segment->start_frame,
                                segment->end_frame,
                                segment->place_token,
                                segment->place_id,
                                (index + 1U < segments->segment_count) ? "," : "");
        if (written_size < 0 || (size_t)written_size >= buffer_size - offset) {
            free(text);
            cs_tool_set_error(error_buffer, error_buffer_size, "Label segments JSON buffer overflow.");
            return 0;
        }
        offset += (size_t)written_size;
    }

    (void)snprintf(text + offset, buffer_size - offset, "  ]\n}\n");
    if (!cs_tool_write_text_file_atomic(path, text, error_buffer, error_buffer_size)) {
        free(text);
        return 0;
    }

    free(text);
    return 1;
}

int cs_tool_read_label_segments(const char* path, CsLabelSegmentsFile* segments, char* error_buffer, size_t error_buffer_size) {
    char* text;
    size_t text_size;
    const char* cursor;

    if (segments == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Label segments output is null.");
        return 0;
    }

    memset(segments, 0, sizeof(*segments));

    if (!cs_tool_file_exists(path)) {
        return 1;
    }

    text = NULL;
    text_size = 0U;
    if (!cs_tool_read_text_file(path, &text, &text_size, error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)text_size;

    if (!cs_tool_extract_string_after_key(text, "session_id", segments->session_id, sizeof(segments->session_id))) {
        free(text);
        cs_tool_set_error(error_buffer, error_buffer_size, "Label segments file is malformed: %s", path);
        return 0;
    }

    cursor = text;
    while ((cursor = cs_tool_find_key(cursor, "segment_id")) != NULL) {
        CsLabelSegment* segment;

        if (segments->segment_count >= CS_TOOL_MAX_SEGMENTS) {
            free(text);
            cs_tool_set_error(error_buffer, error_buffer_size, "Too many label segments in: %s", path);
            return 0;
        }

        segment = &segments->segments[segments->segment_count];
        memset(segment, 0, sizeof(*segment));

        if (!cs_tool_extract_int_after_key(cursor, "segment_id", &segment->segment_id) ||
            !cs_tool_extract_int_after_key(cursor, "start_frame", &segment->start_frame) ||
            !cs_tool_extract_int_after_key(cursor, "end_frame", &segment->end_frame) ||
            !cs_tool_extract_string_after_key(cursor, "place_token", segment->place_token, sizeof(segment->place_token)) ||
            !cs_tool_extract_int_after_key(cursor, "place_id", &segment->place_id)) {
            free(text);
            cs_tool_set_error(error_buffer, error_buffer_size, "Label segment is malformed: %s", path);
            return 0;
        }

        segments->segment_count++;
        cursor += 11;
    }

    free(text);
    return 1;
}

int cs_tool_write_session_json(const char* path, const CsCaptureOptions* options, const char* start_time, const char* end_time, char* error_buffer, size_t error_buffer_size) {
    char json_text[2048];
    int written_size;

    written_size = snprintf(json_text,
                            sizeof(json_text),
                            "{\n"
                            "  \"session_id\": \"%s\",\n"
                            "  \"map_name\": \"%s\",\n"
                            "  \"resolution\": {\n"
                            "    \"width\": %d,\n"
                            "    \"height\": %d\n"
                            "  },\n"
                            "  \"capture_fps\": %d,\n"
                            "  \"team\": \"%s\",\n"
                            "  \"start_time\": \"%s\",\n"
                            "  \"end_time\": \"%s\",\n"
                            "  \"notes\": \"%s\"\n"
                            "}\n",
                            options->session_id,
                            options->map_name,
                            options->width,
                            options->height,
                            options->capture_fps,
                            options->team,
                            start_time,
                            end_time,
                            options->notes);

    if (written_size < 0 || (size_t)written_size >= sizeof(json_text)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Session JSON buffer overflow.");
        return 0;
    }

    return cs_tool_write_text_file_atomic(path, json_text, error_buffer, error_buffer_size);
}

int cs_tool_read_session_json(const char* path,
                              CsCaptureOptions* options,
                              char* start_time,
                              size_t start_time_size,
                              char* end_time,
                              size_t end_time_size,
                              char* error_buffer,
                              size_t error_buffer_size) {
    char* text;
    size_t text_size;

    if (options == NULL || start_time == NULL || end_time == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Session JSON outputs are null.");
        return 0;
    }

    memset(options, 0, sizeof(*options));
    start_time[0] = '\0';
    end_time[0] = '\0';
    text = NULL;
    text_size = 0U;

    if (!cs_tool_read_text_file(path, &text, &text_size, error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)text_size;

    if (!cs_tool_extract_string_after_key(text, "session_id", options->session_id, sizeof(options->session_id)) ||
        !cs_tool_extract_string_after_key(text, "map_name", options->map_name, sizeof(options->map_name)) ||
        !cs_tool_extract_int_after_key(text, "width", &options->width) ||
        !cs_tool_extract_int_after_key(text, "height", &options->height) ||
        !cs_tool_extract_int_after_key(text, "capture_fps", &options->capture_fps) ||
        !cs_tool_extract_string_after_key(text, "team", options->team, sizeof(options->team)) ||
        !cs_tool_extract_string_after_key(text, "start_time", start_time, start_time_size) ||
        !cs_tool_extract_string_after_key(text, "end_time", end_time, end_time_size) ||
        !cs_tool_extract_string_after_key(text, "notes", options->notes, sizeof(options->notes))) {
        free(text);
        cs_tool_set_error(error_buffer, error_buffer_size, "Session JSON is malformed: %s", path);
        return 0;
    }

    free(text);
    return 1;
}

#ifdef _WIN32
typedef struct CsWindowSearchTag {
    HWND hwnd;
} CsWindowSearch;

static int cs_tool_contains_text(const char* haystack, const char* needle) {
    return haystack != NULL && needle != NULL && strstr(haystack, needle) != NULL;
}

static BOOL CALLBACK cs_tool_enum_windows_callback(HWND hwnd, LPARAM lparam) {
    char title[256];
    char class_name[128];
    CsWindowSearch* search;

    if (!IsWindowVisible(hwnd)) {
        return TRUE;
    }

    search = (CsWindowSearch*)lparam;
    title[0] = '\0';
    class_name[0] = '\0';

    (void)GetWindowTextA(hwnd, title, (int)sizeof(title));
    (void)GetClassNameA(hwnd, class_name, (int)sizeof(class_name));

    if (strcmp(class_name, "Valve001") == 0 ||
        cs_tool_contains_text(title, "Counter-Strike") ||
        cs_tool_contains_text(title, "Counter Strike")) {
        search->hwnd = hwnd;
        return FALSE;
    }

    return TRUE;
}

HWND cs_tool_find_counter_strike_window(void) {
    CsWindowSearch search;

    search.hwnd = NULL;
    (void)EnumWindows(cs_tool_enum_windows_callback, (LPARAM)&search);
    return search.hwnd;
}

int cs_tool_get_window_client_size(HWND hwnd, int* out_width, int* out_height, char* error_buffer, size_t error_buffer_size) {
    RECT client_rect;

    if (hwnd == NULL || out_width == NULL || out_height == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Window handle or size outputs are invalid.");
        return 0;
    }

    if (!GetClientRect(hwnd, &client_rect)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to read game window client rect.");
        return 0;
    }

    *out_width = client_rect.right - client_rect.left;
    *out_height = client_rect.bottom - client_rect.top;
    return 1;
}

int cs_tool_capture_window_client_to_bmp(HWND hwnd, const char* output_path, int expected_width, int expected_height, char* error_buffer, size_t error_buffer_size) {
    RECT client_rect;
    POINT top_left;
    int width;
    int height;
    HDC screen_dc;
    HDC memory_dc;
    HBITMAP bitmap_handle;
    HGDIOBJ old_bitmap;
    BITMAPINFO bitmap_info;
    unsigned char* pixel_buffer;
    int stride;
    BITMAPFILEHEADER file_header;
    FILE* output_file;

    if (hwnd == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Game window handle is null.");
        return 0;
    }

    if (!GetClientRect(hwnd, &client_rect)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to read game window client rect.");
        return 0;
    }

    width = client_rect.right - client_rect.left;
    height = client_rect.bottom - client_rect.top;

    if (width != expected_width || height != expected_height) {
        cs_tool_set_error(error_buffer,
                          error_buffer_size,
                          "Game window size mismatch. Expected %dx%d, got %dx%d.",
                          expected_width,
                          expected_height,
                          width,
                          height);
        return 0;
    }

    top_left.x = 0;
    top_left.y = 0;
    if (!ClientToScreen(hwnd, &top_left)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to translate client coordinates.");
        return 0;
    }

    screen_dc = GetDC(NULL);
    if (screen_dc == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to get screen device context.");
        return 0;
    }

    memory_dc = CreateCompatibleDC(screen_dc);
    if (memory_dc == NULL) {
        ReleaseDC(NULL, screen_dc);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to create memory device context.");
        return 0;
    }

    bitmap_handle = CreateCompatibleBitmap(screen_dc, width, height);
    if (bitmap_handle == NULL) {
        DeleteDC(memory_dc);
        ReleaseDC(NULL, screen_dc);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to create capture bitmap.");
        return 0;
    }

    old_bitmap = SelectObject(memory_dc, bitmap_handle);
    if (old_bitmap == NULL) {
        DeleteObject(bitmap_handle);
        DeleteDC(memory_dc);
        ReleaseDC(NULL, screen_dc);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to select capture bitmap.");
        return 0;
    }

    if (!BitBlt(memory_dc, 0, 0, width, height, screen_dc, top_left.x, top_left.y, SRCCOPY | CAPTUREBLT)) {
        SelectObject(memory_dc, old_bitmap);
        DeleteObject(bitmap_handle);
        DeleteDC(memory_dc);
        ReleaseDC(NULL, screen_dc);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to copy game window pixels.");
        return 0;
    }

    ZeroMemory(&bitmap_info, sizeof(bitmap_info));
    bitmap_info.bmiHeader.biSize = sizeof(bitmap_info.bmiHeader);
    bitmap_info.bmiHeader.biWidth = width;
    bitmap_info.bmiHeader.biHeight = -height;
    bitmap_info.bmiHeader.biPlanes = 1;
    bitmap_info.bmiHeader.biBitCount = 24;
    bitmap_info.bmiHeader.biCompression = BI_RGB;

    stride = ((width * 3 + 3) / 4) * 4;
    pixel_buffer = (unsigned char*)malloc((size_t)stride * (size_t)height);
    if (pixel_buffer == NULL) {
        SelectObject(memory_dc, old_bitmap);
        DeleteObject(bitmap_handle);
        DeleteDC(memory_dc);
        ReleaseDC(NULL, screen_dc);
        cs_tool_set_error(error_buffer, error_buffer_size, "Out of memory while capturing bitmap.");
        return 0;
    }

    if (!GetDIBits(memory_dc,
                   bitmap_handle,
                   0U,
                   (UINT)height,
                   pixel_buffer,
                   &bitmap_info,
                   DIB_RGB_COLORS)) {
        free(pixel_buffer);
        SelectObject(memory_dc, old_bitmap);
        DeleteObject(bitmap_handle);
        DeleteDC(memory_dc);
        ReleaseDC(NULL, screen_dc);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to read bitmap bits.");
        return 0;
    }

    memset(&file_header, 0, sizeof(file_header));
    file_header.bfType = 0x4D42U;
    file_header.bfOffBits = (DWORD)(sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER));
    file_header.bfSize = file_header.bfOffBits + (DWORD)((size_t)stride * (size_t)height);

    output_file = fopen(output_path, "wb");
    if (output_file == NULL) {
        free(pixel_buffer);
        SelectObject(memory_dc, old_bitmap);
        DeleteObject(bitmap_handle);
        DeleteDC(memory_dc);
        ReleaseDC(NULL, screen_dc);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open bitmap output file: %s", output_path);
        return 0;
    }

    if (fwrite(&file_header, sizeof(file_header), 1U, output_file) != 1U ||
        fwrite(&bitmap_info.bmiHeader, sizeof(bitmap_info.bmiHeader), 1U, output_file) != 1U ||
        fwrite(pixel_buffer, 1U, (size_t)stride * (size_t)height, output_file) != (size_t)stride * (size_t)height) {
        fclose(output_file);
        free(pixel_buffer);
        SelectObject(memory_dc, old_bitmap);
        DeleteObject(bitmap_handle);
        DeleteDC(memory_dc);
        ReleaseDC(NULL, screen_dc);
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to write bitmap file: %s", output_path);
        return 0;
    }

    fclose(output_file);
    free(pixel_buffer);
    SelectObject(memory_dc, old_bitmap);
    DeleteObject(bitmap_handle);
    DeleteDC(memory_dc);
    ReleaseDC(NULL, screen_dc);
    return 1;
}

int cs_tool_spawn_capture_worker(const char* exe_path, const char* session_root, const char* session_id, int capture_fps, int width, int height, char* error_buffer, size_t error_buffer_size) {
    char command_line[2048];
    STARTUPINFOA startup_info;
    PROCESS_INFORMATION process_info;
    int written_size;

    written_size = snprintf(command_line,
                            sizeof(command_line),
                            "\"%s\" __capture_loop --session-root \"%s\" --session-id \"%s\" --capture-fps %d --width %d --height %d",
                            exe_path,
                            session_root,
                            session_id,
                            capture_fps,
                            width,
                            height);
    if (written_size < 0 || (size_t)written_size >= sizeof(command_line)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Capture worker command line is too long.");
        return 0;
    }

    ZeroMemory(&startup_info, sizeof(startup_info));
    ZeroMemory(&process_info, sizeof(process_info));
    startup_info.cb = sizeof(startup_info);

    if (!CreateProcessA(exe_path,
                        command_line,
                        NULL,
                        NULL,
                        FALSE,
                        DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                        NULL,
                        NULL,
                        &startup_info,
                        &process_info)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to spawn capture worker process.");
        return 0;
    }

    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);
    return 1;
}
#endif
