#include "cs_tool_common.h"

/*
 * Version 1 capture_session tool.
 *
 * Public commands:
 * - start
 * - stop
 * - status
 *
 * A private "__capture_loop" worker keeps capture running in the background so
 * the user can control the session from short CLI commands.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

typedef struct CsCaptureCommandTag {
    char session_root[CS_TOOL_MAX_PATH];
    char session_id[64];
    char map_name[64];
    int width;
    int height;
    int capture_fps;
    char team[32];
    char notes[128];
    char dictionary_path[CS_TOOL_MAX_PATH];
} CsCaptureCommand;

/*
 * The worker and the label tool both update capture_state.json.
 * Re-reading the file before every worker write keeps label ownership on the
 * current place fields and avoids stale in-memory state clobbering labels.
 */
static int cs_capture_reload_state(const char* state_path, CsCaptureState* state, char* error_buffer, size_t error_buffer_size) {
    CsCaptureState latest_state;

    memset(&latest_state, 0, sizeof(latest_state));
    if (!cs_tool_read_capture_state(state_path, &latest_state, error_buffer, error_buffer_size)) {
        return 0;
    }

    *state = latest_state;
    return 1;
}

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

static int cs_capture_default_session_root(char* out_root, size_t out_size, char* error_buffer, size_t error_buffer_size) {
    char current_directory[CS_TOOL_MAX_PATH];

    if (!cs_tool_get_current_directory(current_directory, sizeof(current_directory), error_buffer, error_buffer_size)) {
        return 0;
    }

    return cs_tool_join_two_paths(out_root, out_size, current_directory, "demo", "cs\\data\\v1_area\\raw");
}

static int cs_capture_parse_options(int argc, char** argv, int start_index, CsCaptureCommand* command, char* error_buffer, size_t error_buffer_size) {
    memset(command, 0, sizeof(*command));
    command->width = -1;
    command->height = -1;
    command->capture_fps = -1;

    if (!cs_capture_default_session_root(command->session_root, sizeof(command->session_root), error_buffer, error_buffer_size)) {
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
            if (!cs_tool_copy_string(command->session_root, sizeof(command->session_root), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Session root is too long.");
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--session-id")) {
            if (!cs_tool_copy_string(command->session_id, sizeof(command->session_id), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Session id is too long.");
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--map")) {
            if (!cs_tool_copy_string(command->map_name, sizeof(command->map_name), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Map name is too long.");
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--width")) {
            if (!cs_parse_int_value(option_value, &command->width)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Invalid width value: %s", option_value);
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--height")) {
            if (!cs_parse_int_value(option_value, &command->height)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Invalid height value: %s", option_value);
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--capture-fps")) {
            if (!cs_parse_int_value(option_value, &command->capture_fps)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Invalid capture FPS value: %s", option_value);
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--team")) {
            if (!cs_tool_copy_string(command->team, sizeof(command->team), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Team value is too long.");
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--notes")) {
            if (!cs_tool_copy_string(command->notes, sizeof(command->notes), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Notes value is too long.");
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--dictionary")) {
            if (!cs_tool_copy_string(command->dictionary_path, sizeof(command->dictionary_path), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Dictionary path is too long.");
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

static int cs_capture_validate_start_options(const CsCaptureCommand* command, char* error_buffer, size_t error_buffer_size) {
    if (command->session_id[0] == '\0') {
        cs_tool_set_error(error_buffer, error_buffer_size, "Missing required option: --session-id");
        return 0;
    }
    if (command->map_name[0] == '\0') {
        cs_tool_set_error(error_buffer, error_buffer_size, "Missing required option: --map");
        return 0;
    }
    if (strcmp(command->map_name, "de_dust2") != 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Only de_dust2 is supported in Version 1.");
        return 0;
    }
    if (command->width <= 0 || command->height <= 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Width and height must be positive integers.");
        return 0;
    }
    if (command->capture_fps <= 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Capture FPS must be a positive integer.");
        return 0;
    }
    return 1;
}

static int cs_capture_prepare_session(const CsCaptureCommand* command, char* error_buffer, size_t error_buffer_size) {
    char session_dir[CS_TOOL_MAX_PATH];
    char frames_dir[CS_TOOL_MAX_PATH];
    char session_path[CS_TOOL_MAX_PATH];
    char state_path[CS_TOOL_MAX_PATH];
    char segments_path[CS_TOOL_MAX_PATH];
    char stop_path[CS_TOOL_MAX_PATH];
    char dictionary_snapshot_path[CS_TOOL_MAX_PATH];
    char dictionary_path[CS_TOOL_MAX_PATH];
    char start_time[64];
    CsCaptureOptions options;
    CsCaptureState state;
    CsLabelSegmentsFile segments;

    memset(&options, 0, sizeof(options));
    memset(&state, 0, sizeof(state));
    memset(&segments, 0, sizeof(segments));

    if (!cs_tool_get_session_dir(command->session_root, command->session_id, session_dir, sizeof(session_dir)) ||
        !cs_tool_get_frames_dir(command->session_root, command->session_id, frames_dir, sizeof(frames_dir)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "session.json", session_path, sizeof(session_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "capture_state.json", state_path, sizeof(state_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "label_segments.json", segments_path, sizeof(segments_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "capture.stop", stop_path, sizeof(stop_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "place_dictionary.snapshot.json", dictionary_snapshot_path, sizeof(dictionary_snapshot_path))) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Session path is too long.");
        return 0;
    }

    if (!cs_tool_make_dirs(session_dir, error_buffer, error_buffer_size) ||
        !cs_tool_make_dirs(frames_dir, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (cs_tool_file_exists(stop_path)) {
        (void)remove(stop_path);
    }

    if (command->dictionary_path[0] != '\0') {
        if (!cs_tool_copy_string(dictionary_path, sizeof(dictionary_path), command->dictionary_path)) {
            cs_tool_set_error(error_buffer, error_buffer_size, "Dictionary path is too long.");
            return 0;
        }
    } else if (!cs_tool_find_default_dictionary(dictionary_path, sizeof(dictionary_path), error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_copy_file(dictionary_path, dictionary_snapshot_path, error_buffer, error_buffer_size)) {
        return 0;
    }

    options.width = command->width;
    options.height = command->height;
    options.capture_fps = command->capture_fps;
    (void)cs_tool_copy_string(options.session_root, sizeof(options.session_root), command->session_root);
    (void)cs_tool_copy_string(options.session_id, sizeof(options.session_id), command->session_id);
    (void)cs_tool_copy_string(options.map_name, sizeof(options.map_name), command->map_name);
    (void)cs_tool_copy_string(options.team, sizeof(options.team), command->team[0] == '\0' ? "unknown" : command->team);
    (void)cs_tool_copy_string(options.notes, sizeof(options.notes), command->notes);

    cs_tool_now_iso8601(start_time, sizeof(start_time));
    if (!cs_tool_write_session_json(session_path, &options, start_time, "", error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)cs_tool_copy_string(state.session_id, sizeof(state.session_id), command->session_id);
    (void)cs_tool_copy_string(state.status, sizeof(state.status), "starting");
    (void)cs_tool_copy_string(state.current_place_token, sizeof(state.current_place_token), "");
    state.current_place_id = -1;
    state.last_frame_index = -1;
    state.captured_frame_count = 0;

    if (!cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)cs_tool_copy_string(segments.session_id, sizeof(segments.session_id), command->session_id);
    return cs_tool_write_label_segments(segments_path, &segments, error_buffer, error_buffer_size);
}

static int cs_capture_command_start(const CsCaptureCommand* command, char* error_buffer, size_t error_buffer_size) {
#ifdef _WIN32
    HWND hwnd;
    int actual_width;
    int actual_height;
    char exe_path[CS_TOOL_MAX_PATH];

    if (!cs_capture_validate_start_options(command, error_buffer, error_buffer_size)) {
        return 0;
    }

    hwnd = cs_tool_find_counter_strike_window();
    if (hwnd == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to find Counter-Strike window.");
        return 0;
    }

    if (!cs_tool_get_window_client_size(hwnd, &actual_width, &actual_height, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (actual_width != command->width || actual_height != command->height) {
        cs_tool_set_error(error_buffer,
                          error_buffer_size,
                          "Window size mismatch. Expected %dx%d, got %dx%d.",
                          command->width,
                          command->height,
                          actual_width,
                          actual_height);
        return 0;
    }

    if (!cs_capture_prepare_session(command, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_get_executable_path(exe_path, sizeof(exe_path), error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_spawn_capture_worker(exe_path,
                                      command->session_root,
                                      command->session_id,
                                      command->capture_fps,
                                      command->width,
                                      command->height,
                                      error_buffer,
                                      error_buffer_size)) {
        return 0;
    }

    printf("Started capture session %s\n", command->session_id);
    return 1;
#else
    (void)command;
    cs_tool_set_error(error_buffer, error_buffer_size, "Capture session is only implemented on Windows.");
    return 0;
#endif
}

static int cs_capture_command_stop(const CsCaptureCommand* command, char* error_buffer, size_t error_buffer_size) {
    char stop_path[CS_TOOL_MAX_PATH];
    char stop_text[64];

    if (command->session_id[0] == '\0') {
        cs_tool_set_error(error_buffer, error_buffer_size, "Missing required option: --session-id");
        return 0;
    }

    if (!cs_tool_get_session_file_path(command->session_root, command->session_id, "capture.stop", stop_path, sizeof(stop_path))) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Stop path is too long.");
        return 0;
    }

    cs_tool_now_iso8601(stop_text, sizeof(stop_text));
    if (!cs_tool_write_text_file_atomic(stop_path, stop_text, error_buffer, error_buffer_size)) {
        return 0;
    }

    printf("Requested stop for session %s\n", command->session_id);
    return 1;
}

static int cs_capture_command_status(const CsCaptureCommand* command, char* error_buffer, size_t error_buffer_size) {
    char state_path[CS_TOOL_MAX_PATH];
    CsCaptureState state;

    memset(&state, 0, sizeof(state));

    if (command->session_id[0] == '\0') {
        cs_tool_set_error(error_buffer, error_buffer_size, "Missing required option: --session-id");
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
    printf("status=%s\n", state.status);
    printf("captured_frame_count=%d\n", state.captured_frame_count);
    printf("last_frame_index=%d\n", state.last_frame_index);
    printf("current_place_token=%s\n", state.current_place_token);
    printf("current_place_id=%d\n", state.current_place_id);
    return 1;
}

static int cs_capture_worker_loop(const CsCaptureCommand* command, char* error_buffer, size_t error_buffer_size) {
#ifdef _WIN32
    char state_path[CS_TOOL_MAX_PATH];
    char session_path[CS_TOOL_MAX_PATH];
    char frames_dir[CS_TOOL_MAX_PATH];
    char stop_path[CS_TOOL_MAX_PATH];
    char frame_path[CS_TOOL_MAX_PATH];
    char start_time[64];
    char end_time[64];
    unsigned long sleep_ms;
    int frame_index;
    CsCaptureState state;
    CsCaptureOptions options;

    memset(&state, 0, sizeof(state));
    memset(&options, 0, sizeof(options));

    if (!cs_tool_get_session_file_path(command->session_root, command->session_id, "capture_state.json", state_path, sizeof(state_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "session.json", session_path, sizeof(session_path)) ||
        !cs_tool_get_session_file_path(command->session_root, command->session_id, "capture.stop", stop_path, sizeof(stop_path)) ||
        !cs_tool_get_frames_dir(command->session_root, command->session_id, frames_dir, sizeof(frames_dir))) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Worker session path is too long.");
        return 0;
    }

    if (!cs_tool_read_session_json(session_path,
                                   &options,
                                   start_time,
                                   sizeof(start_time),
                                   end_time,
                                   sizeof(end_time),
                                   error_buffer,
                                   error_buffer_size)) {
        return 0;
    }

    sleep_ms = (unsigned long)(1000 / ((command->capture_fps > 0) ? command->capture_fps : 1));
    frame_index = 0;

    if (!cs_capture_reload_state(state_path, &state, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (state.session_id[0] == '\0') {
        (void)cs_tool_copy_string(state.session_id, sizeof(state.session_id), command->session_id);
    }
    (void)cs_tool_copy_string(state.status, sizeof(state.status), "running");

    if (!cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
        return 0;
    }

    for (;;) {
        HWND hwnd;

        if (cs_tool_file_exists(stop_path)) {
            break;
        }

        hwnd = cs_tool_find_counter_strike_window();
        if (hwnd == NULL) {
            if (!cs_capture_reload_state(state_path, &state, error_buffer, error_buffer_size)) {
                return 0;
            }
            (void)cs_tool_copy_string(state.status, sizeof(state.status), "error");
            (void)cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size);
            cs_tool_set_error(error_buffer, error_buffer_size, "Capture worker lost Counter-Strike window.");
            return 0;
        }

        if (snprintf(frame_path, sizeof(frame_path), "%s\\frame_%06d.bmp", frames_dir, frame_index) < 0) {
            cs_tool_set_error(error_buffer, error_buffer_size, "Frame path formatting failed.");
            return 0;
        }

        if (!cs_tool_capture_window_client_to_bmp(hwnd, frame_path, command->width, command->height, error_buffer, error_buffer_size)) {
            if (!cs_capture_reload_state(state_path, &state, error_buffer, error_buffer_size)) {
                return 0;
            }
            (void)cs_tool_copy_string(state.status, sizeof(state.status), "error");
            (void)cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size);
            return 0;
        }

        if (!cs_capture_reload_state(state_path, &state, error_buffer, error_buffer_size)) {
            return 0;
        }

        (void)cs_tool_copy_string(state.status, sizeof(state.status), "running");
        state.last_frame_index = frame_index;
        state.captured_frame_count = frame_index + 1;
        if (!cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size)) {
            return 0;
        }

        frame_index++;
        Sleep(sleep_ms);
    }

    cs_tool_now_iso8601(end_time, sizeof(end_time));

    (void)cs_tool_copy_string(options.session_root, sizeof(options.session_root), command->session_root);

    if (!cs_tool_write_session_json(session_path, &options, start_time, end_time, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_capture_reload_state(state_path, &state, error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)cs_tool_copy_string(state.status, sizeof(state.status), "stopped");
    return cs_tool_write_capture_state(state_path, &state, error_buffer, error_buffer_size);
#else
    (void)command;
    cs_tool_set_error(error_buffer, error_buffer_size, "Capture worker is only implemented on Windows.");
    return 0;
#endif
}

static void cs_capture_print_usage(void) {
    printf("Usage:\n");
    printf("  cs_capture_session start --session-id <id> --map de_dust2 --width <w> --height <h> --capture-fps <fps> [--session-root <path>] [--team <team>] [--notes <text>] [--dictionary <path>]\n");
    printf("  cs_capture_session stop --session-id <id> [--session-root <path>]\n");
    printf("  cs_capture_session status --session-id <id> [--session-root <path>]\n");
}

int main(int argc, char** argv) {
    CsCaptureCommand command;
    char error_buffer[CS_TOOL_MAX_TEXT];
    const char* subcommand;
    int ok;

    memset(&command, 0, sizeof(command));
    memset(error_buffer, 0, sizeof(error_buffer));

    if (argc < 2) {
        cs_capture_print_usage();
        return 1;
    }

    subcommand = argv[1];

    if (!cs_capture_parse_options(argc, argv, 2, &command, error_buffer, sizeof(error_buffer))) {
        fprintf(stderr, "%s\n", error_buffer);
        return 1;
    }

    if (strcmp(subcommand, "start") == 0) {
        ok = cs_capture_command_start(&command, error_buffer, sizeof(error_buffer));
    } else if (strcmp(subcommand, "stop") == 0) {
        ok = cs_capture_command_stop(&command, error_buffer, sizeof(error_buffer));
    } else if (strcmp(subcommand, "status") == 0) {
        ok = cs_capture_command_status(&command, error_buffer, sizeof(error_buffer));
    } else if (strcmp(subcommand, "__capture_loop") == 0) {
        ok = cs_capture_worker_loop(&command, error_buffer, sizeof(error_buffer));
    } else {
        cs_capture_print_usage();
        return 1;
    }

    if (!ok) {
        fprintf(stderr, "%s\n", error_buffer);
        return 2;
    }

    return 0;
}
