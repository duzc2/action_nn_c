#include "cs_tool_common.h"

/*
 * Version 1 dataset_build tool.
 *
 * It converts raw labeled sessions into deterministic train/val/test sample
 * lists.  The implementation intentionally favors explicit, inspectable logic
 * over clever abstractions because Version 1 is primarily about making the
 * data pipeline stable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

typedef struct CsBuildOptionsTag {
    char raw_root[CS_TOOL_MAX_PATH];
    char output_root[CS_TOOL_MAX_PATH];
    int min_frame_step;
    int dedup_enabled;
    int blur_filter_enabled;
    int split_train;
    int split_val;
    int split_test;
} CsBuildOptions;

typedef struct CsSessionInfoTag {
    char session_id[64];
    int split_kind;
} CsSessionInfo;

typedef struct CsSampleRecordTag {
    char sample_id[128];
    char image_rel_path[CS_TOOL_MAX_PATH];
    char session_id[64];
    int frame_index;
    char timestamp[64];
    double pos_x;
    double pos_y;
    double pos_z;
    double yaw;
    double pitch;
    char place_token[64];
    int place_id;
    char label_source[32];
    int split_kind;
} CsSampleRecord;

typedef struct CsBuildSummaryTag {
    int raw_session_count;
    int raw_frame_count;
    int state_trace_count;
    int filtered_frame_count;
    int dedup_removed_count;
    int blur_removed_count;
    int teacher_alignment_drop_count;
    int projection_outside_count;
    int projection_ambiguous_count;
    int projection_disabled_count;
    int train_count;
    int val_count;
    int test_count;
} CsBuildSummary;

typedef struct CsTraceFrameTag {
    int frame_index;
    char frame_path[CS_TOOL_MAX_PATH];
} CsTraceFrame;

typedef struct CsProjectionZoneTag {
    int place_id;
    char place_token[64];
    int enabled_in_v1;
    int priority;
    double z_min;
    double z_max;
    double min_x;
    double min_y;
    double max_x;
    double max_y;
} CsProjectionZone;

typedef struct CsProjectionConfigTag {
    char projection_name[64];
    char map_name[64];
    char dictionary_name[64];
    int version;
    double boundary_margin;
    CsProjectionZone zones[CS_TOOL_MAX_PLACES];
    size_t zone_count;
} CsProjectionConfig;

enum {
    CS_PROJECTION_OK = 0,
    CS_PROJECTION_OUTSIDE = 1,
    CS_PROJECTION_AMBIGUOUS = 2,
    CS_PROJECTION_DISABLED = 3
};

enum {
    CS_SPLIT_TRAIN = 0,
    CS_SPLIT_VAL = 1,
    CS_SPLIT_TEST = 2
};

static int cs_arg_is_option(const char* value, const char* expected) {
    return value != NULL && expected != NULL && strcmp(value, expected) == 0;
}

static int cs_parse_int_value(const char* text, int* out_value) {
    int scanned_value;

    if (text == NULL || out_value == NULL) {
        return 0;
    }

    if (sscanf(text, "%d", &scanned_value) != 1) {
        return 0;
    }

    *out_value = scanned_value;
    return 1;
}

static int cs_build_default_paths(CsBuildOptions* options, char* error_buffer, size_t error_buffer_size) {
    char current_directory[CS_TOOL_MAX_PATH];

    if (!cs_tool_get_current_directory(current_directory, sizeof(current_directory), error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_join_two_paths(options->raw_root, sizeof(options->raw_root), current_directory, "demo", "cs\\data\\v1_area\\raw") ||
        !cs_tool_join_two_paths(options->output_root, sizeof(options->output_root), current_directory, "demo", "cs\\data\\v1_area\\processed")) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Default dataset paths are too long.");
        return 0;
    }

    return 1;
}

static int cs_build_parse_split(const char* text, CsBuildOptions* options) {
    int train_value;
    int val_value;
    int test_value;

    train_value = 0;
    val_value = 0;
    test_value = 0;

    if (sscanf(text, "train:val:test=%d:%d:%d", &train_value, &val_value, &test_value) != 3) {
        return 0;
    }

    if (train_value <= 0 || val_value <= 0 || test_value <= 0) {
        return 0;
    }

    options->split_train = train_value;
    options->split_val = val_value;
    options->split_test = test_value;
    return 1;
}

static int cs_build_parse_options(int argc, char** argv, int start_index, CsBuildOptions* options, char* error_buffer, size_t error_buffer_size) {
    int index;

    memset(options, 0, sizeof(*options));
    options->min_frame_step = 3;
    options->dedup_enabled = 1;
    options->blur_filter_enabled = 1;
    options->split_train = 8;
    options->split_val = 1;
    options->split_test = 1;

    if (!cs_build_default_paths(options, error_buffer, error_buffer_size)) {
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

        if (cs_arg_is_option(option_name, "--raw-root")) {
            if (!cs_tool_copy_string(options->raw_root, sizeof(options->raw_root), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Raw root is too long.");
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--output-root")) {
            if (!cs_tool_copy_string(options->output_root, sizeof(options->output_root), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Output root is too long.");
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--min-frame-step")) {
            if (!cs_parse_int_value(option_value, &options->min_frame_step) || options->min_frame_step <= 0) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Invalid min-frame-step: %s", option_value);
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--dedup")) {
            options->dedup_enabled = (strcmp(option_value, "on") == 0) ? 1 : 0;
        } else if (cs_arg_is_option(option_name, "--blur-filter")) {
            options->blur_filter_enabled = (strcmp(option_value, "on") == 0) ? 1 : 0;
        } else if (cs_arg_is_option(option_name, "--split")) {
            if (!cs_build_parse_split(option_value, options)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Invalid split value: %s", option_value);
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

static unsigned int cs_hash_string(const char* text) {
    unsigned int hash_value;

    hash_value = 2166136261u;
    while (text != NULL && *text != '\0') {
        hash_value ^= (unsigned int)(unsigned char)(*text);
        hash_value *= 16777619u;
        text++;
    }

    return hash_value;
}

static int cs_build_choose_split(const CsBuildOptions* options, const char* session_id) {
    unsigned int hash_value;
    int total;
    int bucket;

    total = options->split_train + options->split_val + options->split_test;
    hash_value = cs_hash_string(session_id);
    bucket = (int)(hash_value % (unsigned int)total);

    if (bucket < options->split_train) {
        return CS_SPLIT_TRAIN;
    }
    if (bucket < options->split_train + options->split_val) {
        return CS_SPLIT_VAL;
    }
    return CS_SPLIT_TEST;
}

static int cs_build_is_bad_frame(const char* path) {
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA attributes;
    unsigned long long file_size;

    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &attributes)) {
        return 1;
    }

    file_size = ((unsigned long long)attributes.nFileSizeHigh << 32) | attributes.nFileSizeLow;
    return file_size < 4096ULL;
#else
    (void)path;
    return 0;
#endif
}

static unsigned long long cs_build_hash_file(const char* path, char* error_buffer, size_t error_buffer_size) {
    FILE* file_handle;
    unsigned char buffer[8192];
    size_t read_size;
    unsigned long long hash_value;

    file_handle = fopen(path, "rb");
    if (file_handle == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open sample for hashing: %s", path);
        return 0ULL;
    }

    hash_value = 1469598103934665603ULL;
    for (;;) {
        read_size = fread(buffer, 1U, sizeof(buffer), file_handle);
        if (read_size > 0U) {
            size_t index;
            for (index = 0U; index < read_size; ++index) {
                hash_value ^= (unsigned long long)buffer[index];
                hash_value *= 1099511628211ULL;
            }
        }
        if (read_size < sizeof(buffer)) {
            break;
        }
    }

    fclose(file_handle);
    return hash_value;
}

static int cs_build_collect_sessions(const char* raw_root,
                                     CsSessionInfo** out_sessions,
                                     size_t* out_session_count,
                                     const CsBuildOptions* options,
                                     char* error_buffer,
                                     size_t error_buffer_size) {
#ifdef _WIN32
    char search_pattern[CS_TOOL_MAX_PATH];
    WIN32_FIND_DATAA find_data;
    HANDLE find_handle;
    CsSessionInfo* sessions;
    size_t session_capacity;
    size_t session_count;

    sessions = NULL;
    session_capacity = 0U;
    session_count = 0U;

    if (snprintf(search_pattern, sizeof(search_pattern), "%s\\*", raw_root) < 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Raw root search pattern is too long.");
        return 0;
    }

    find_handle = FindFirstFileA(search_pattern, &find_data);
    if (find_handle == INVALID_HANDLE_VALUE) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to enumerate raw root: %s", raw_root);
        return 0;
    }

    do {
        if ((find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0U) {
            continue;
        }
        if (strcmp(find_data.cFileName, ".") == 0 || strcmp(find_data.cFileName, "..") == 0) {
            continue;
        }

        if (session_count == session_capacity) {
            size_t new_capacity;
            CsSessionInfo* new_sessions;

            new_capacity = (session_capacity == 0U) ? 8U : session_capacity * 2U;
            new_sessions = (CsSessionInfo*)realloc(sessions, new_capacity * sizeof(CsSessionInfo));
            if (new_sessions == NULL) {
                FindClose(find_handle);
                free(sessions);
                cs_tool_set_error(error_buffer, error_buffer_size, "Out of memory while collecting sessions.");
                return 0;
            }

            sessions = new_sessions;
            session_capacity = new_capacity;
        }

        memset(&sessions[session_count], 0, sizeof(CsSessionInfo));
        (void)cs_tool_copy_string(sessions[session_count].session_id, sizeof(sessions[session_count].session_id), find_data.cFileName);
        sessions[session_count].split_kind = cs_build_choose_split(options, find_data.cFileName);
        session_count++;
    } while (FindNextFileA(find_handle, &find_data));

    FindClose(find_handle);
    *out_sessions = sessions;
    *out_session_count = session_count;
    return 1;
#else
    (void)raw_root;
    (void)out_sessions;
    (void)out_session_count;
    (void)options;
    cs_tool_set_error(error_buffer, error_buffer_size, "Dataset build is only implemented on Windows.");
    return 0;
#endif
}

static const char* cs_skip_spaces(const char* cursor) {
    while (cursor != NULL &&
           *cursor != '\0' &&
           (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t')) {
        cursor++;
    }
    return cursor;
}

static const char* cs_find_key(const char* text, const char* key) {
    char pattern[128];
    int written_size;

    written_size = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (written_size < 0 || (size_t)written_size >= sizeof(pattern)) {
        return NULL;
    }
    return strstr(text, pattern);
}

static int cs_extract_string_after_key(const char* text, const char* key, char* out_value, size_t out_size) {
    const char* key_position;
    const char* colon;
    const char* first_quote;
    const char* second_quote;
    size_t value_length;

    key_position = cs_find_key(text, key);
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

static int cs_extract_int_after_key(const char* text, const char* key, int* out_value) {
    const char* key_position;
    const char* colon;
    const char* value_position;
    int scanned_value;

    key_position = cs_find_key(text, key);
    if (key_position == NULL) {
        return 0;
    }

    colon = strchr(key_position, ':');
    if (colon == NULL) {
        return 0;
    }

    value_position = cs_skip_spaces(colon + 1);
    if (value_position == NULL) {
        return 0;
    }

    if (sscanf(value_position, "%d", &scanned_value) != 1) {
        return 0;
    }

    *out_value = scanned_value;
    return 1;
}

static int cs_extract_double_after_key(const char* text, const char* key, double* out_value) {
    const char* key_position;
    const char* colon;
    const char* value_position;
    double scanned_value;

    key_position = cs_find_key(text, key);
    if (key_position == NULL) {
        return 0;
    }

    colon = strchr(key_position, ':');
    if (colon == NULL) {
        return 0;
    }

    value_position = cs_skip_spaces(colon + 1);
    if (value_position == NULL) {
        return 0;
    }

    if (sscanf(value_position, "%lf", &scanned_value) != 1) {
        return 0;
    }

    *out_value = scanned_value;
    return 1;
}

static int cs_extract_bool_after_key(const char* text, const char* key, int* out_value) {
    const char* key_position;
    const char* colon;
    const char* value_position;

    key_position = cs_find_key(text, key);
    if (key_position == NULL) {
        return 0;
    }

    colon = strchr(key_position, ':');
    if (colon == NULL) {
        return 0;
    }

    value_position = cs_skip_spaces(colon + 1);
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

static int cs_build_find_default_projection(char* out_path, size_t out_size, char* error_buffer, size_t error_buffer_size) {
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
                         "%s\\demo\\cs\\config\\de_dust2_v1_teacher_projection.json",
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

    cs_tool_set_error(error_buffer, error_buffer_size, "Failed to locate demo/cs/config/de_dust2_v1_teacher_projection.json");
    return 0;
}

static int cs_build_collect_trace_frames(const char* frames_dir,
                                         CsTraceFrame** out_frames,
                                         size_t* out_frame_count,
                                         char* error_buffer,
                                         size_t error_buffer_size) {
#ifdef _WIN32
    char search_pattern[CS_TOOL_MAX_PATH];
    WIN32_FIND_DATAA find_data;
    HANDLE find_handle;
    CsTraceFrame* frames;
    size_t frame_capacity;
    size_t frame_count;

    frames = NULL;
    frame_capacity = 0U;
    frame_count = 0U;

    if (snprintf(search_pattern, sizeof(search_pattern), "%s\\frame_*.bmp", frames_dir) < 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Frame search pattern is too long.");
        return 0;
    }

    find_handle = FindFirstFileA(search_pattern, &find_data);
    if (find_handle == INVALID_HANDLE_VALUE) {
        *out_frames = NULL;
        *out_frame_count = 0U;
        return 1;
    }

    do {
        int frame_index;

        if ((find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0U) {
            continue;
        }
        if (sscanf(find_data.cFileName, "frame_%d.bmp", &frame_index) != 1) {
            continue;
        }

        if (frame_count == frame_capacity) {
            size_t new_capacity;
            CsTraceFrame* new_frames;

            new_capacity = (frame_capacity == 0U) ? 128U : frame_capacity * 2U;
            new_frames = (CsTraceFrame*)realloc(frames, new_capacity * sizeof(CsTraceFrame));
            if (new_frames == NULL) {
                FindClose(find_handle);
                free(frames);
                cs_tool_set_error(error_buffer, error_buffer_size, "Out of memory while collecting trace frames.");
                return 0;
            }
            frames = new_frames;
            frame_capacity = new_capacity;
        }

        memset(&frames[frame_count], 0, sizeof(CsTraceFrame));
        frames[frame_count].frame_index = frame_index;
        if (snprintf(frames[frame_count].frame_path, sizeof(frames[frame_count].frame_path), "%s\\%s", frames_dir, find_data.cFileName) < 0) {
            FindClose(find_handle);
            free(frames);
            cs_tool_set_error(error_buffer, error_buffer_size, "Frame path is too long.");
            return 0;
        }
        frame_count++;
    } while (FindNextFileA(find_handle, &find_data));

    FindClose(find_handle);
    *out_frames = frames;
    *out_frame_count = frame_count;
    return 1;
#else
    (void)frames_dir;
    (void)out_frames;
    (void)out_frame_count;
    cs_tool_set_error(error_buffer, error_buffer_size, "Frame enumeration is only implemented on Windows.");
    return 0;
#endif
}

static int cs_build_load_state_trace(const char* path,
                                     CsStateTraceRecord** out_records,
                                     size_t* out_record_count,
                                     char* error_buffer,
                                     size_t error_buffer_size) {
    char* text;
    size_t text_size;
    char* line_start;
    CsStateTraceRecord* records;
    size_t record_capacity;
    size_t record_count;

    text = NULL;
    text_size = 0U;
    records = NULL;
    record_capacity = 0U;
    record_count = 0U;

    if (!cs_tool_file_exists(path)) {
        *out_records = NULL;
        *out_record_count = 0U;
        return 1;
    }

    if (!cs_tool_read_text_file(path, &text, &text_size, error_buffer, error_buffer_size)) {
        return 0;
    }

    line_start = text;
    while (line_start != NULL && *line_start != '\0') {
        char* line_end;
        char saved_character;

        line_end = strchr(line_start, '\n');
        if (line_end != NULL) {
            saved_character = *line_end;
            *line_end = '\0';
        } else {
            saved_character = '\0';
        }

        if (line_start[0] != '\0') {
            if (record_count == record_capacity) {
                size_t new_capacity;
                CsStateTraceRecord* new_records;

                new_capacity = (record_capacity == 0U) ? 128U : record_capacity * 2U;
                new_records = (CsStateTraceRecord*)realloc(records, new_capacity * sizeof(CsStateTraceRecord));
                if (new_records == NULL) {
                    free(records);
                    free(text);
                    cs_tool_set_error(error_buffer, error_buffer_size, "Out of memory while loading state trace.");
                    return 0;
                }
                records = new_records;
                record_capacity = new_capacity;
            }

            if (!cs_tool_parse_state_trace_line(line_start, &records[record_count], error_buffer, error_buffer_size)) {
                free(records);
                free(text);
                return 0;
            }
            record_count++;
        }

        if (line_end == NULL) {
            break;
        }

        *line_end = saved_character;
        line_start = line_end + 1;
    }

    free(text);
    *out_records = records;
    *out_record_count = record_count;
    return 1;
}

static const CsStateTraceRecord* cs_build_find_trace_record(const CsStateTraceRecord* records, size_t record_count, int frame_index) {
    size_t index;

    for (index = 0U; index < record_count; ++index) {
        if (records[index].frame_index == frame_index) {
            return &records[index];
        }
    }
    return NULL;
}

static int cs_build_load_projection(const char* path,
                                    CsProjectionConfig* projection,
                                    char* error_buffer,
                                    size_t error_buffer_size) {
    char* text;
    size_t text_size;
    const char* cursor;

    memset(projection, 0, sizeof(*projection));
    text = NULL;
    text_size = 0U;

    if (!cs_tool_read_text_file(path, &text, &text_size, error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)text_size;
    if (!cs_extract_string_after_key(text, "projection_name", projection->projection_name, sizeof(projection->projection_name)) ||
        !cs_extract_string_after_key(text, "map_name", projection->map_name, sizeof(projection->map_name)) ||
        !cs_extract_string_after_key(text, "dictionary_name", projection->dictionary_name, sizeof(projection->dictionary_name)) ||
        !cs_extract_int_after_key(text, "version", &projection->version) ||
        !cs_extract_double_after_key(text, "boundary_margin", &projection->boundary_margin)) {
        free(text);
        cs_tool_set_error(error_buffer, error_buffer_size, "Projection file is missing top-level fields: %s", path);
        return 0;
    }

    cursor = text;
    while ((cursor = cs_find_key(cursor, "place_id")) != NULL) {
        CsProjectionZone* zone;

        if (projection->zone_count >= CS_TOOL_MAX_PLACES) {
            free(text);
            cs_tool_set_error(error_buffer, error_buffer_size, "Projection file has too many zones: %s", path);
            return 0;
        }

        zone = &projection->zones[projection->zone_count];
        memset(zone, 0, sizeof(*zone));
        if (!cs_extract_int_after_key(cursor, "place_id", &zone->place_id) ||
            !cs_extract_string_after_key(cursor, "place_token", zone->place_token, sizeof(zone->place_token)) ||
            !cs_extract_bool_after_key(cursor, "enabled_in_v1", &zone->enabled_in_v1) ||
            !cs_extract_int_after_key(cursor, "priority", &zone->priority) ||
            !cs_extract_double_after_key(cursor, "z_min", &zone->z_min) ||
            !cs_extract_double_after_key(cursor, "z_max", &zone->z_max) ||
            !cs_extract_double_after_key(cursor, "min_x", &zone->min_x) ||
            !cs_extract_double_after_key(cursor, "min_y", &zone->min_y) ||
            !cs_extract_double_after_key(cursor, "max_x", &zone->max_x) ||
            !cs_extract_double_after_key(cursor, "max_y", &zone->max_y)) {
            free(text);
            cs_tool_set_error(error_buffer, error_buffer_size, "Projection zone is malformed or not using aabb_2d: %s", path);
            return 0;
        }

        projection->zone_count++;
        cursor += 1;  /* advance past closing delimiter, cs_find_key skips to next entry */
    }

    free(text);
    return 1;
}

static int cs_build_validate_projection(const CsProjectionConfig* projection,
                                        const CsPlaceDictionary* dictionary,
                                        const CsCaptureOptions* session_options,
                                        const char* session_id,
                                        char* error_buffer,
                                        size_t error_buffer_size) {
    size_t index;

    if (strcmp(session_options->map_name, "de_dust2") != 0 || strcmp(projection->map_name, "de_dust2") != 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Only de_dust2 is supported in Version 1.");
        return 0;
    }
    if (strcmp(projection->map_name, session_options->map_name) != 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Projection map_name does not match session %s.", session_id);
        return 0;
    }
    if (strcmp(projection->dictionary_name, dictionary->dictionary_name) != 0 || projection->version != dictionary->version) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Projection dictionary metadata does not match place dictionary.");
        return 0;
    }

    for (index = 0U; index < projection->zone_count; ++index) {
        const CsPlaceEntry* entry;

        entry = cs_tool_find_place_by_id(dictionary, projection->zones[index].place_id);
        if (entry == NULL || strcmp(entry->place_token, projection->zones[index].place_token) != 0) {
            cs_tool_set_error(error_buffer, error_buffer_size, "Projection zone metadata does not match place dictionary.");
            return 0;
        }
    }

    return 1;
}

static double cs_build_normalize_yaw(double yaw_value) {
    while (yaw_value < 0.0) {
        yaw_value += 360.0;
    }
    while (yaw_value >= 360.0) {
        yaw_value -= 360.0;
    }
    return yaw_value;
}

static int cs_build_project_record(const CsProjectionConfig* projection,
                                   const CsStateTraceRecord* record,
                                   const CsProjectionZone** out_zone) {
    const CsProjectionZone* best_zone;
    int best_priority;
    int best_priority_count;
    size_t index;

    best_zone = NULL;
    best_priority = 0;
    best_priority_count = 0;

    for (index = 0U; index < projection->zone_count; ++index) {
        const CsProjectionZone* zone;
        double boundary_distance;
        double x_margin;
        double y_margin;

        zone = &projection->zones[index];
        if (record->pos_z < zone->z_min || record->pos_z > zone->z_max) {
            continue;
        }
        if (record->pos_x < zone->min_x || record->pos_x > zone->max_x ||
            record->pos_y < zone->min_y || record->pos_y > zone->max_y) {
            continue;
        }

        x_margin = record->pos_x - zone->min_x;
        if ((zone->max_x - record->pos_x) < x_margin) {
            x_margin = zone->max_x - record->pos_x;
        }
        y_margin = record->pos_y - zone->min_y;
        if ((zone->max_y - record->pos_y) < y_margin) {
            y_margin = zone->max_y - record->pos_y;
        }
        boundary_distance = (x_margin < y_margin) ? x_margin : y_margin;
        if (boundary_distance < projection->boundary_margin) {
            return CS_PROJECTION_AMBIGUOUS;
        }

        if (best_zone == NULL || zone->priority < best_priority) {
            best_zone = zone;
            best_priority = zone->priority;
            best_priority_count = 1;
        } else if (zone->priority == best_priority) {
            best_priority_count++;
        }
    }

    if (best_zone == NULL) {
        return CS_PROJECTION_OUTSIDE;
    }
    if (best_priority_count > 1) {
        return CS_PROJECTION_AMBIGUOUS;
    }
    if (!best_zone->enabled_in_v1) {
        *out_zone = best_zone;
        return CS_PROJECTION_DISABLED;
    }

    *out_zone = best_zone;
    return CS_PROJECTION_OK;
}

static int cs_build_write_split_list(const char* output_root,
                                     const char* split_name,
                                     const CsSampleRecord* samples,
                                     size_t sample_count,
                                     int split_kind,
                                     char* error_buffer,
                                     size_t error_buffer_size) {
    char path[CS_TOOL_MAX_PATH];
    FILE* file_handle;
    size_t index;
    int first_entry;

    if (snprintf(path, sizeof(path), "%s\\%s_list.json", output_root, split_name) < 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Split list path is too long for %s.", split_name);
        return 0;
    }

    if (!cs_tool_make_dirs(output_root, error_buffer, error_buffer_size)) {
        return 0;
    }

    file_handle = fopen(path, "wb");
    if (file_handle == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open split list for writing: %s", path);
        return 0;
    }

    fprintf(file_handle, "{\n  \"dataset_version\": \"v1_area\",\n  \"split\": \"%s\",\n  \"samples\": [\n", split_name);
    first_entry = 1;

    for (index = 0U; index < sample_count; ++index) {
        const CsSampleRecord* sample;

        sample = &samples[index];
        if (sample->split_kind != split_kind) {
            continue;
        }

        if (!first_entry) {
            fprintf(file_handle, ",\n");
        }
        first_entry = 0;

        fprintf(file_handle,
                "    {\n"
                "      \"sample_id\": \"%s\",\n"
                "      \"image_path\": \"%s\",\n"
                "      \"session_id\": \"%s\",\n"
                "      \"frame_index\": %d,\n"
                "      \"timestamp\": \"%s\",\n"
                "      \"teacher_pose\": {\n"
                "        \"pos_x\": %.6f,\n"
                "        \"pos_y\": %.6f,\n"
                "        \"pos_z\": %.6f,\n"
                "        \"yaw\": %.6f,\n"
                "        \"pitch\": %.6f\n"
                "      },\n"
                "      \"place_token\": \"%s\",\n"
                "      \"place_id\": %d,\n"
                "      \"label_source\": \"%s\"\n"
                "    }",
                sample->sample_id,
                sample->image_rel_path,
                sample->session_id,
                sample->frame_index,
                sample->timestamp,
                sample->pos_x,
                sample->pos_y,
                sample->pos_z,
                sample->yaw,
                sample->pitch,
                sample->place_token,
                sample->place_id,
                sample->label_source);
    }

    fprintf(file_handle, "\n  ]\n}\n");
    fclose(file_handle);
    return 1;
}

static int cs_build_write_report(const char* output_root,
                                 const CsBuildSummary* summary,
                                 char* error_buffer,
                                 size_t error_buffer_size) {
    char path[CS_TOOL_MAX_PATH];
    char json_text[3072];
    int written_size;

    if (snprintf(path, sizeof(path), "%s\\build_report.json", output_root) < 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Build report path is too long.");
        return 0;
    }

    written_size = snprintf(json_text,
                            sizeof(json_text),
                            "{\n"
                            "  \"raw_session_count\": %d,\n"
                            "  \"raw_frame_count\": %d,\n"
                            "  \"state_trace_count\": %d,\n"
                            "  \"filtered_frame_count\": %d,\n"
                            "  \"dedup_removed_count\": %d,\n"
                            "  \"blur_removed_count\": %d,\n"
                            "  \"teacher_alignment_drop_count\": %d,\n"
                            "  \"projection_outside_count\": %d,\n"
                            "  \"projection_ambiguous_count\": %d,\n"
                            "  \"projection_disabled_count\": %d,\n"
                            "  \"train_count\": %d,\n"
                            "  \"val_count\": %d,\n"
                            "  \"test_count\": %d\n"
                            "}\n",
                            summary->raw_session_count,
                            summary->raw_frame_count,
                            summary->state_trace_count,
                            summary->filtered_frame_count,
                            summary->dedup_removed_count,
                            summary->blur_removed_count,
                            summary->teacher_alignment_drop_count,
                            summary->projection_outside_count,
                            summary->projection_ambiguous_count,
                            summary->projection_disabled_count,
                            summary->train_count,
                            summary->val_count,
                            summary->test_count);

    if (written_size < 0 || (size_t)written_size >= sizeof(json_text)) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Build report JSON buffer overflow.");
        return 0;
    }

    return cs_tool_write_text_file_atomic(path, json_text, error_buffer, error_buffer_size);
}

static int cs_build_run(const CsBuildOptions* options, char* error_buffer, size_t error_buffer_size) {
    char dictionary_path[CS_TOOL_MAX_PATH];
    char projection_path[CS_TOOL_MAX_PATH];
    CsPlaceDictionary dictionary;
    CsProjectionConfig projection;
    CsSessionInfo* sessions;
    size_t session_count;
    CsSampleRecord* samples;
    size_t sample_count;
    size_t sample_capacity;
    unsigned long long* hashes;
    size_t hash_count;
    size_t hash_capacity;
    size_t session_index;
    CsBuildSummary summary;

    memset(&dictionary, 0, sizeof(dictionary));
    memset(&projection, 0, sizeof(projection));
    sessions = NULL;
    session_count = 0U;
    samples = NULL;
    sample_count = 0U;
    sample_capacity = 0U;
    hashes = NULL;
    hash_count = 0U;
    hash_capacity = 0U;
    memset(&summary, 0, sizeof(summary));

    if (!cs_tool_find_default_dictionary(dictionary_path, sizeof(dictionary_path), error_buffer, error_buffer_size) ||
        !cs_tool_load_dictionary(dictionary_path, &dictionary, error_buffer, error_buffer_size) ||
        !cs_build_find_default_projection(projection_path, sizeof(projection_path), error_buffer, error_buffer_size) ||
        !cs_build_load_projection(projection_path, &projection, error_buffer, error_buffer_size) ||
        !cs_build_collect_sessions(options->raw_root, &sessions, &session_count, options, error_buffer, error_buffer_size)) {
        return 0;
    }

    summary.raw_session_count = (int)session_count;

    for (session_index = 0U; session_index < session_count; ++session_index) {
        char session_path[CS_TOOL_MAX_PATH];
        char trace_path[CS_TOOL_MAX_PATH];
        char frames_dir[CS_TOOL_MAX_PATH];
        char start_time[64];
        char end_time[64];
        CsCaptureOptions session_options;
        CsTraceFrame* frames;
        size_t frame_count;
        CsStateTraceRecord* records;
        size_t record_count;
        size_t frame_list_index;
        int last_kept_frame;

        memset(&session_options, 0, sizeof(session_options));
        memset(start_time, 0, sizeof(start_time));
        memset(end_time, 0, sizeof(end_time));
        frames = NULL;
        frame_count = 0U;
        records = NULL;
        record_count = 0U;

        if (!cs_tool_get_session_file_path(options->raw_root, sessions[session_index].session_id, "session.json", session_path, sizeof(session_path)) ||
            !cs_tool_get_session_file_path(options->raw_root, sessions[session_index].session_id, "state_trace.jsonl", trace_path, sizeof(trace_path)) ||
            !cs_tool_get_frames_dir(options->raw_root, sessions[session_index].session_id, frames_dir, sizeof(frames_dir))) {
            free(sessions);
            free(samples);
            free(hashes);
            cs_tool_set_error(error_buffer, error_buffer_size, "Session path is too long.");
            return 0;
        }

        if (!cs_tool_read_session_json(session_path,
                                       &session_options,
                                       start_time,
                                       sizeof(start_time),
                                       end_time,
                                       sizeof(end_time),
                                       error_buffer,
                                       error_buffer_size) ||
            !cs_build_validate_projection(&projection, &dictionary, &session_options, sessions[session_index].session_id, error_buffer, error_buffer_size) ||
            !cs_build_collect_trace_frames(frames_dir, &frames, &frame_count, error_buffer, error_buffer_size) ||
            !cs_build_load_state_trace(trace_path, &records, &record_count, error_buffer, error_buffer_size)) {
            free(sessions);
            free(samples);
            free(hashes);
            free(frames);
            free(records);
            return 0;
        }

        summary.raw_frame_count += (int)frame_count;
        summary.state_trace_count += (int)record_count;
        last_kept_frame = -1000000000;

        for (frame_list_index = 0U; frame_list_index < frame_count; ++frame_list_index) {
            const CsTraceFrame* frame_info;
            const CsStateTraceRecord* record;
            const CsProjectionZone* projected_zone;
            int projection_status;

            frame_info = &frames[frame_list_index];
            if (frame_info->frame_index - last_kept_frame < options->min_frame_step) {
                continue;
            }

            record = cs_build_find_trace_record(records, record_count, frame_info->frame_index);
            if (record == NULL) {
                summary.teacher_alignment_drop_count++;
                continue;
            }

            projection_status = cs_build_project_record(&projection, record, &projected_zone);
            if (projection_status == CS_PROJECTION_OUTSIDE) {
                summary.projection_outside_count++;
                continue;
            }
            if (projection_status == CS_PROJECTION_AMBIGUOUS) {
                summary.projection_ambiguous_count++;
                continue;
            }
            if (projection_status == CS_PROJECTION_DISABLED) {
                summary.projection_disabled_count++;
                continue;
            }

            if (options->blur_filter_enabled && cs_build_is_bad_frame(frame_info->frame_path)) {
                summary.blur_removed_count++;
                continue;
            }

            if (options->dedup_enabled) {
                unsigned long long file_hash;
                int duplicate_found;
                size_t hash_index;

                file_hash = cs_build_hash_file(frame_info->frame_path, error_buffer, error_buffer_size);
                if (file_hash == 0ULL) {
                    free(sessions);
                    free(samples);
                    free(hashes);
                    free(frames);
                    free(records);
                    return 0;
                }

                duplicate_found = 0;
                for (hash_index = 0U; hash_index < hash_count; ++hash_index) {
                    if (hashes[hash_index] == file_hash) {
                        duplicate_found = 1;
                        break;
                    }
                }

                if (duplicate_found) {
                    summary.dedup_removed_count++;
                    continue;
                }

                if (hash_count == hash_capacity) {
                    size_t new_capacity;
                    unsigned long long* new_hashes;

                    new_capacity = (hash_capacity == 0U) ? 128U : hash_capacity * 2U;
                    new_hashes = (unsigned long long*)realloc(hashes, new_capacity * sizeof(unsigned long long));
                    if (new_hashes == NULL) {
                        free(sessions);
                        free(samples);
                        free(hashes);
                        free(frames);
                        free(records);
                        cs_tool_set_error(error_buffer, error_buffer_size, "Out of memory while growing dedup table.");
                        return 0;
                    }
                    hashes = new_hashes;
                    hash_capacity = new_capacity;
                }
                hashes[hash_count++] = file_hash;
            }

            if (sample_count == sample_capacity) {
                size_t new_capacity;
                CsSampleRecord* new_samples;

                new_capacity = (sample_capacity == 0U) ? 128U : sample_capacity * 2U;
                new_samples = (CsSampleRecord*)realloc(samples, new_capacity * sizeof(CsSampleRecord));
                if (new_samples == NULL) {
                    free(sessions);
                    free(samples);
                    free(hashes);
                    free(frames);
                    free(records);
                    cs_tool_set_error(error_buffer, error_buffer_size, "Out of memory while growing sample list.");
                    return 0;
                }
                samples = new_samples;
                sample_capacity = new_capacity;
            }

            {
                char dst_dir[CS_TOOL_MAX_PATH];
                char dst_path[CS_TOOL_MAX_PATH];
                char rel_path[CS_TOOL_MAX_PATH];

                if (snprintf(dst_dir, sizeof(dst_dir), "%s\\samples\\%s", options->output_root, sessions[session_index].session_id) < 0 ||
                    snprintf(dst_path, sizeof(dst_path), "%s\\frame_%06d.bmp", dst_dir, frame_info->frame_index) < 0 ||
                    snprintf(rel_path, sizeof(rel_path), "samples\\%s\\frame_%06d.bmp", sessions[session_index].session_id, frame_info->frame_index) < 0) {
                    free(sessions);
                    free(samples);
                    free(hashes);
                    free(frames);
                    free(records);
                    cs_tool_set_error(error_buffer, error_buffer_size, "Destination frame path is too long.");
                    return 0;
                }

                if (!cs_tool_make_dirs(dst_dir, error_buffer, error_buffer_size) ||
                    !cs_tool_copy_binary_file(frame_info->frame_path, dst_path, error_buffer, error_buffer_size)) {
                    free(sessions);
                    free(samples);
                    free(hashes);
                    free(frames);
                    free(records);
                    return 0;
                }

                memset(&samples[sample_count], 0, sizeof(CsSampleRecord));
                (void)snprintf(samples[sample_count].sample_id,
                               sizeof(samples[sample_count].sample_id),
                               "%s_%06d",
                               sessions[session_index].session_id,
                               frame_info->frame_index);
                (void)cs_tool_copy_string(samples[sample_count].image_rel_path, sizeof(samples[sample_count].image_rel_path), rel_path);
                (void)cs_tool_copy_string(samples[sample_count].session_id, sizeof(samples[sample_count].session_id), sessions[session_index].session_id);
                (void)cs_tool_copy_string(samples[sample_count].timestamp, sizeof(samples[sample_count].timestamp), record->timestamp);
                (void)cs_tool_copy_string(samples[sample_count].place_token, sizeof(samples[sample_count].place_token), projected_zone->place_token);
                (void)cs_tool_copy_string(samples[sample_count].label_source, sizeof(samples[sample_count].label_source), "teacher_projected");
                samples[sample_count].frame_index = frame_info->frame_index;
                samples[sample_count].pos_x = record->pos_x;
                samples[sample_count].pos_y = record->pos_y;
                samples[sample_count].pos_z = record->pos_z;
                samples[sample_count].yaw = cs_build_normalize_yaw(record->yaw);
                samples[sample_count].pitch = record->pitch;
                samples[sample_count].place_id = projected_zone->place_id;
                samples[sample_count].split_kind = sessions[session_index].split_kind;

                if (samples[sample_count].split_kind == CS_SPLIT_TRAIN) {
                    summary.train_count++;
                } else if (samples[sample_count].split_kind == CS_SPLIT_VAL) {
                    summary.val_count++;
                } else {
                    summary.test_count++;
                }

                summary.filtered_frame_count++;
                last_kept_frame = frame_info->frame_index;
                sample_count++;
            }
        }

        free(frames);
        free(records);
    }

    if (!cs_build_write_split_list(options->output_root, "train", samples, sample_count, CS_SPLIT_TRAIN, error_buffer, error_buffer_size) ||
        !cs_build_write_split_list(options->output_root, "val", samples, sample_count, CS_SPLIT_VAL, error_buffer, error_buffer_size) ||
        !cs_build_write_split_list(options->output_root, "test", samples, sample_count, CS_SPLIT_TEST, error_buffer, error_buffer_size) ||
        !cs_build_write_report(options->output_root, &summary, error_buffer, error_buffer_size)) {
        free(sessions);
        free(samples);
        free(hashes);
        return 0;
    }

    printf("Built dataset from %d sessions.\n", summary.raw_session_count);
    printf("Raw frames/state trace: %d / %d\n", summary.raw_frame_count, summary.state_trace_count);
    printf("Filtered samples: %d\n", summary.filtered_frame_count);
    printf("Alignment drops: %d\n", summary.teacher_alignment_drop_count);
    printf("Projection outside/ambiguous/disabled: %d / %d / %d\n",
           summary.projection_outside_count,
           summary.projection_ambiguous_count,
           summary.projection_disabled_count);
    printf("Train/Val/Test: %d / %d / %d\n", summary.train_count, summary.val_count, summary.test_count);

    free(sessions);
    free(samples);
    free(hashes);
    return 1;
}

static void cs_build_print_usage(void) {
    printf("Usage:\n");
    printf("  cs_dataset_build run [--raw-root <path>] [--output-root <path>] [--min-frame-step <n>] [--dedup on|off] [--blur-filter on|off] [--split train:val:test=8:1:1]\n");
}

int main(int argc, char** argv) {
    CsBuildOptions options;
    char error_buffer[CS_TOOL_MAX_TEXT];
    int ok;

    memset(&options, 0, sizeof(options));
    memset(error_buffer, 0, sizeof(error_buffer));

    if (argc < 2 || strcmp(argv[1], "run") != 0) {
        cs_build_print_usage();
        return 1;
    }

    if (!cs_build_parse_options(argc, argv, 2, &options, error_buffer, sizeof(error_buffer))) {
        fprintf(stderr, "%s\n", error_buffer);
        return 1;
    }

    ok = cs_build_run(&options, error_buffer, sizeof(error_buffer));
    if (!ok) {
        fprintf(stderr, "%s\n", error_buffer);
        return 2;
    }

    return 0;
}
