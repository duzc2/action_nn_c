#include "cs_tool_common.h"

/*
 * Version 1 label_session tool.
 *
 * The tool keeps labeling intentionally simple:
 * - read one fixed dictionary snapshot;
 * - read the current capture state;
 * - append or close one active segment.
 *
 * That matches the Version 1 goal of "segment-level labels with very little
 * user effort".
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct CsLabelOptionsTag {
    char session_root[CS_TOOL_MAX_PATH];
    char session_id[64];
    char place_token[64];
} CsLabelOptions;

static int cs_arg_is_option(const char* value, const char* expected) {
    return value != NULL && expected != NULL && strcmp(value, expected) == 0;
}

static int cs_label_default_session_root(char* out_root, size_t out_size, char* error_buffer, size_t error_buffer_size) {
    char current_directory[CS_TOOL_MAX_PATH];

    if (!cs_tool_get_current_directory(current_directory, sizeof(current_directory), error_buffer, error_buffer_size)) {
        return 0;
    }

    return cs_tool_join_two_paths(out_root, out_size, current_directory, "demo", "cs\\data\\v1_area\\raw");
}

static int cs_label_parse_options(int argc, char** argv, int start_index, CsLabelOptions* options, char* error_buffer, size_t error_buffer_size) {
    memset(options, 0, sizeof(*options));

    if (!cs_label_default_session_root(options->session_root, sizeof(options->session_root), error_buffer, error_buffer_size)) {
        return 0;
    }

    {
        int index;

        index = start_index;
        while (index < argc) {
        const char* option_name = argv[index];
        const char* option_value;

        if (index + 1 >= argc) {
            cs_tool_set_error(error_buffer, error_buffer_size, "Missing value after option: %s", option_name);
            return 0;
        }

        option_value = argv[index + 1];

        if (cs_arg_is_option(option_name, "--session-root")) {
            if (!cs_tool_copy_string(options->session_root, sizeof(options->session_root), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Session root is too long.");
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--session-id")) {
            if (!cs_tool_copy_string(options->session_id, sizeof(options->session_id), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Session id is too long.");
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--place-token")) {
            if (!cs_tool_copy_string(options->place_token, sizeof(options->place_token), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Place token is too long.");
                return 0;
            }
        } else {
            cs_tool_set_error(error_buffer, error_buffer_size, "Unsupported option: %s", option_name);
            return 0;
        }

            index += 2;
        }
    }

    return 1;
}

static int cs_label_validate_common(const CsLabelOptions* options, char* error_buffer, size_t error_buffer_size) {
    if (options->session_id[0] == '\0') {
        cs_tool_set_error(error_buffer, error_buffer_size, "Missing required option: --session-id");
        return 0;
    }
    return 1;
}

static int cs_label_load_snapshot_dictionary(const CsLabelOptions* options, CsPlaceDictionary* dictionary, char* error_buffer, size_t error_buffer_size) {
    char dictionary_path[CS_TOOL_MAX_PATH];

    if (!cs_tool_get_session_file_path(options->session_root,
                                       options->session_id,
                                       "place_dictionary.snapshot.json",
                                       dictionary_path,
                                       sizeof(dictionary_path))) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Dictionary snapshot path is too long.");
        return 0;
    }

    return cs_tool_load_dictionary(dictionary_path, dictionary, error_buffer, error_buffer_size);
}

static int cs_label_command_set(const CsLabelOptions* options, char* error_buffer, size_t error_buffer_size) {
    char state_path[CS_TOOL_MAX_PATH];
    char segments_path[CS_TOOL_MAX_PATH];
    CsPlaceDictionary dictionary;
    const CsPlaceEntry* place_entry;
    CsCaptureState state;
    CsLabelSegmentsFile segments;
    int new_start_frame;

    memset(&dictionary, 0, sizeof(dictionary));
    memset(&state, 0, sizeof(state));
    memset(&segments, 0, sizeof(segments));

    if (!cs_label_validate_common(options, error_buffer, error_buffer_size)) {
        return 0;
    }
    if (options->place_token[0] == '\0') {
        cs_tool_set_error(error_buffer, error_buffer_size, "Missing required option: --place-token");
        return 0;
    }

    if (!cs_label_load_snapshot_dictionary(options, &dictionary, error_buffer, error_buffer_size)) {
        return 0;
    }

    place_entry = cs_tool_find_place_by_token(&dictionary, options->place_token);
    if (place_entry == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Unknown place token: %s", options->place_token);
        return 0;
    }

    if (!cs_tool_get_session_file_path(options->session_root, options->session_id, "capture_state.json", state_path, sizeof(state_path)) ||
        !cs_tool_get_session_file_path(options->session_root, options->session_id, "label_segments.json", segments_path, sizeof(segments_path))) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Label path is too long.");
        return 0;
    }

    if (!cs_tool_read_capture_state(state_path, &state, error_buffer, error_buffer_size) ||
        !cs_tool_read_label_segments(segments_path, &segments, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (segments.session_id[0] == '\0') {
        (void)cs_tool_copy_string(segments.session_id, sizeof(segments.session_id), options->session_id);
    }

    new_start_frame = state.last_frame_index + 1;
    if (new_start_frame < 0) {
        new_start_frame = 0;
    }

    if (segments.segment_count > 0U && segments.segments[segments.segment_count - 1U].end_frame < 0) {
        CsLabelSegment* active_segment;

        active_segment = &segments.segments[segments.segment_count - 1U];

        /*
         * Re-setting the same token should be a no-op.
         * That makes live labeling more forgiving when the user repeats the
         * last command or a helper script retries it.
         */
        if (strcmp(active_segment->place_token, place_entry->place_token) == 0 &&
            active_segment->place_id == place_entry->place_id) {
            (void)cs_tool_copy_string(state.current_place_token, sizeof(state.current_place_token), place_entry->place_token);
            state.current_place_id = place_entry->place_id;
            if (!cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
                return 0;
            }

            printf("Current place token already active: %s (place_id=%d)\n", place_entry->place_token, place_entry->place_id);
            return 1;
        }

        /*
         * If no frame has been captured for the open segment yet, remove that
         * empty segment instead of closing it with an invalid end_frame.
         */
        if (state.last_frame_index < active_segment->start_frame) {
            new_start_frame = active_segment->start_frame;
            memset(active_segment, 0, sizeof(*active_segment));
            segments.segment_count--;
        } else {
            active_segment->end_frame = state.last_frame_index;
            new_start_frame = state.last_frame_index + 1;
        }
    }

    if (segments.segment_count >= CS_TOOL_MAX_SEGMENTS) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Too many label segments.");
        return 0;
    }

    {
        CsLabelSegment* new_segment;

        new_segment = &segments.segments[segments.segment_count];
        memset(new_segment, 0, sizeof(*new_segment));
        new_segment->segment_id = (int)segments.segment_count;
        new_segment->start_frame = new_start_frame;
        new_segment->end_frame = -1;
        new_segment->place_id = place_entry->place_id;
        (void)cs_tool_copy_string(new_segment->place_token, sizeof(new_segment->place_token), place_entry->place_token);
        segments.segment_count++;
    }

    if (!cs_tool_write_label_segments(segments_path, &segments, error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)cs_tool_copy_string(state.current_place_token, sizeof(state.current_place_token), place_entry->place_token);
    state.current_place_id = place_entry->place_id;
    if (!cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
        return 0;
    }

    printf("Set current place token to %s (place_id=%d)\n", place_entry->place_token, place_entry->place_id);
    return 1;
}

static int cs_label_command_current(const CsLabelOptions* options, char* error_buffer, size_t error_buffer_size) {
    char state_path[CS_TOOL_MAX_PATH];
    CsCaptureState state;

    memset(&state, 0, sizeof(state));

    if (!cs_label_validate_common(options, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_get_session_file_path(options->session_root, options->session_id, "capture_state.json", state_path, sizeof(state_path))) {
        cs_tool_set_error(error_buffer, error_buffer_size, "State path is too long.");
        return 0;
    }

    if (!cs_tool_read_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
        return 0;
    }

    printf("current_place_token=%s\n", state.current_place_token);
    printf("current_place_id=%d\n", state.current_place_id);
    printf("last_frame_index=%d\n", state.last_frame_index);
    return 1;
}

static int cs_label_command_close(const CsLabelOptions* options, char* error_buffer, size_t error_buffer_size) {
    char state_path[CS_TOOL_MAX_PATH];
    char segments_path[CS_TOOL_MAX_PATH];
    CsCaptureState state;
    CsLabelSegmentsFile segments;

    memset(&state, 0, sizeof(state));
    memset(&segments, 0, sizeof(segments));

    if (!cs_label_validate_common(options, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_get_session_file_path(options->session_root, options->session_id, "capture_state.json", state_path, sizeof(state_path)) ||
        !cs_tool_get_session_file_path(options->session_root, options->session_id, "label_segments.json", segments_path, sizeof(segments_path))) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Label path is too long.");
        return 0;
    }

    if (!cs_tool_read_capture_state(state_path, &state, error_buffer, error_buffer_size) ||
        !cs_tool_read_label_segments(segments_path, &segments, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (segments.segment_count == 0U) {
        cs_tool_set_error(error_buffer, error_buffer_size, "No label segments exist for this session.");
        return 0;
    }

    if (segments.segments[segments.segment_count - 1U].end_frame >= 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "The current label segment is already closed.");
        return 0;
    }

    /*
     * Closing an empty active segment should not produce an invalid range.
     * Removing the empty segment keeps the file valid for downstream tooling.
     */
    if (state.last_frame_index < segments.segments[segments.segment_count - 1U].start_frame) {
        memset(&segments.segments[segments.segment_count - 1U], 0, sizeof(segments.segments[segments.segment_count - 1U]));
        segments.segment_count--;
    } else {
        segments.segments[segments.segment_count - 1U].end_frame = state.last_frame_index;
    }

    if (!cs_tool_write_label_segments(segments_path, &segments, error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)cs_tool_copy_string(state.current_place_token, sizeof(state.current_place_token), "");
    state.current_place_id = -1;
    if (!cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
        return 0;
    }

    printf("Closed current label segment for session %s\n", options->session_id);
    return 1;
}

static void cs_label_print_usage(void) {
    printf("Usage:\n");
    printf("  cs_label_session set --session-id <id> --place-token <token> [--session-root <path>]\n");
    printf("  cs_label_session current --session-id <id> [--session-root <path>]\n");
    printf("  cs_label_session close --session-id <id> [--session-root <path>]\n");
}

int main(int argc, char** argv) {
    CsLabelOptions options;
    char error_buffer[CS_TOOL_MAX_TEXT];
    const char* subcommand;
    int ok;

    memset(&options, 0, sizeof(options));
    memset(error_buffer, 0, sizeof(error_buffer));

    if (argc < 2) {
        cs_label_print_usage();
        return 1;
    }

    subcommand = argv[1];

    if (!cs_label_parse_options(argc, argv, 2, &options, error_buffer, sizeof(error_buffer))) {
        fprintf(stderr, "%s\n", error_buffer);
        return 1;
    }

    if (strcmp(subcommand, "set") == 0) {
        ok = cs_label_command_set(&options, error_buffer, sizeof(error_buffer));
    } else if (strcmp(subcommand, "current") == 0) {
        ok = cs_label_command_current(&options, error_buffer, sizeof(error_buffer));
    } else if (strcmp(subcommand, "close") == 0) {
        ok = cs_label_command_close(&options, error_buffer, sizeof(error_buffer));
    } else {
        cs_label_print_usage();
        return 1;
    }

    if (!ok) {
        fprintf(stderr, "%s\n", error_buffer);
        return 2;
    }

    return 0;
}
