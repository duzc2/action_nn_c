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
    char place_token[64];
    int place_id;
    int split_kind;
} CsSampleRecord;

typedef struct CsBuildSummaryTag {
    int raw_session_count;
    int raw_frame_count;
    int filtered_frame_count;
    int dedup_removed_count;
    int blur_removed_count;
    int train_count;
    int val_count;
    int test_count;
} CsBuildSummary;

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
                "      \"place_token\": \"%s\",\n"
                "      \"place_id\": %d\n"
                "    }",
                sample->sample_id,
                sample->image_rel_path,
                sample->session_id,
                sample->frame_index,
                sample->place_token,
                sample->place_id);
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
    char json_text[2048];
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
                            "  \"filtered_frame_count\": %d,\n"
                            "  \"dedup_removed_count\": %d,\n"
                            "  \"blur_removed_count\": %d,\n"
                            "  \"train_count\": %d,\n"
                            "  \"val_count\": %d,\n"
                            "  \"test_count\": %d\n"
                            "}\n",
                            summary->raw_session_count,
                            summary->raw_frame_count,
                            summary->filtered_frame_count,
                            summary->dedup_removed_count,
                            summary->blur_removed_count,
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

    sessions = NULL;
    session_count = 0U;
    samples = NULL;
    sample_count = 0U;
    sample_capacity = 0U;
    hashes = NULL;
    hash_count = 0U;
    hash_capacity = 0U;
    memset(&summary, 0, sizeof(summary));

    if (!cs_build_collect_sessions(options->raw_root, &sessions, &session_count, options, error_buffer, error_buffer_size)) {
        return 0;
    }

    summary.raw_session_count = (int)session_count;

    for (session_index = 0U; session_index < session_count; ++session_index) {
        char segments_path[CS_TOOL_MAX_PATH];
        char frames_dir[CS_TOOL_MAX_PATH];
        CsLabelSegmentsFile segments;
        size_t segment_index;

        memset(&segments, 0, sizeof(segments));

        if (!cs_tool_get_session_file_path(options->raw_root, sessions[session_index].session_id, "label_segments.json", segments_path, sizeof(segments_path)) ||
            !cs_tool_get_frames_dir(options->raw_root, sessions[session_index].session_id, frames_dir, sizeof(frames_dir))) {
            free(sessions);
            free(samples);
            free(hashes);
            cs_tool_set_error(error_buffer, error_buffer_size, "Session path is too long.");
            return 0;
        }

        if (!cs_tool_read_label_segments(segments_path, &segments, error_buffer, error_buffer_size)) {
            free(sessions);
            free(samples);
            free(hashes);
            return 0;
        }

        for (segment_index = 0U; segment_index < segments.segment_count; ++segment_index) {
            const CsLabelSegment* segment;
            int frame_index;
            int last_kept_frame;

            segment = &segments.segments[segment_index];
            if (segment->end_frame < segment->start_frame) {
                free(sessions);
                free(samples);
                free(hashes);
                cs_tool_set_error(error_buffer,
                                  error_buffer_size,
                                  "Open or invalid label segment found in session %s.",
                                  sessions[session_index].session_id);
                return 0;
            }

            last_kept_frame = -1000000000;
            for (frame_index = segment->start_frame; frame_index <= segment->end_frame; ++frame_index) {
                char src_path[CS_TOOL_MAX_PATH];
                char dst_dir[CS_TOOL_MAX_PATH];
                char dst_path[CS_TOOL_MAX_PATH];
                char rel_path[CS_TOOL_MAX_PATH];
                unsigned long long file_hash;
                int duplicate_found;
                size_t hash_index;

                summary.raw_frame_count++;

                if (frame_index - last_kept_frame < options->min_frame_step) {
                    continue;
                }

                if (snprintf(src_path, sizeof(src_path), "%s\\frame_%06d.bmp", frames_dir, frame_index) < 0) {
                    free(sessions);
                    free(samples);
                    free(hashes);
                    cs_tool_set_error(error_buffer, error_buffer_size, "Source frame path is too long.");
                    return 0;
                }

                if (!cs_tool_file_exists(src_path)) {
                    continue;
                }

                if (options->blur_filter_enabled && cs_build_is_bad_frame(src_path)) {
                    summary.blur_removed_count++;
                    continue;
                }

                file_hash = 0ULL;
                duplicate_found = 0;
                if (options->dedup_enabled) {
                    file_hash = cs_build_hash_file(src_path, error_buffer, error_buffer_size);
                    if (file_hash == 0ULL) {
                        free(sessions);
                        free(samples);
                        free(hashes);
                        return 0;
                    }

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
                        cs_tool_set_error(error_buffer, error_buffer_size, "Out of memory while growing sample list.");
                        return 0;
                    }
                    samples = new_samples;
                    sample_capacity = new_capacity;
                }

                if (options->dedup_enabled) {
                    if (hash_count == hash_capacity) {
                        size_t new_capacity;
                        unsigned long long* new_hashes;

                        new_capacity = (hash_capacity == 0U) ? 128U : hash_capacity * 2U;
                        new_hashes = (unsigned long long*)realloc(hashes, new_capacity * sizeof(unsigned long long));
                        if (new_hashes == NULL) {
                            free(sessions);
                            free(samples);
                            free(hashes);
                            cs_tool_set_error(error_buffer, error_buffer_size, "Out of memory while growing dedup table.");
                            return 0;
                        }
                        hashes = new_hashes;
                        hash_capacity = new_capacity;
                    }
                    hashes[hash_count++] = file_hash;
                }

                if (snprintf(dst_dir, sizeof(dst_dir), "%s\\samples\\%s", options->output_root, sessions[session_index].session_id) < 0 ||
                    snprintf(dst_path, sizeof(dst_path), "%s\\frame_%06d.bmp", dst_dir, frame_index) < 0 ||
                    snprintf(rel_path, sizeof(rel_path), "samples\\%s\\frame_%06d.bmp", sessions[session_index].session_id, frame_index) < 0) {
                    free(sessions);
                    free(samples);
                    free(hashes);
                    cs_tool_set_error(error_buffer, error_buffer_size, "Destination frame path is too long.");
                    return 0;
                }

                if (!cs_tool_make_dirs(dst_dir, error_buffer, error_buffer_size) ||
                    !cs_tool_copy_binary_file(src_path, dst_path, error_buffer, error_buffer_size)) {
                    free(sessions);
                    free(samples);
                    free(hashes);
                    return 0;
                }

                memset(&samples[sample_count], 0, sizeof(CsSampleRecord));
                (void)snprintf(samples[sample_count].sample_id,
                               sizeof(samples[sample_count].sample_id),
                               "%s_%06d",
                               sessions[session_index].session_id,
                               frame_index);
                (void)cs_tool_copy_string(samples[sample_count].image_rel_path, sizeof(samples[sample_count].image_rel_path), rel_path);
                (void)cs_tool_copy_string(samples[sample_count].session_id, sizeof(samples[sample_count].session_id), sessions[session_index].session_id);
                (void)cs_tool_copy_string(samples[sample_count].place_token, sizeof(samples[sample_count].place_token), segment->place_token);
                samples[sample_count].frame_index = frame_index;
                samples[sample_count].place_id = segment->place_id;
                samples[sample_count].split_kind = sessions[session_index].split_kind;

                if (samples[sample_count].split_kind == CS_SPLIT_TRAIN) {
                    summary.train_count++;
                } else if (samples[sample_count].split_kind == CS_SPLIT_VAL) {
                    summary.val_count++;
                } else {
                    summary.test_count++;
                }

                summary.filtered_frame_count++;
                last_kept_frame = frame_index;
                sample_count++;
            }
        }
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
    printf("Filtered samples: %d\n", summary.filtered_frame_count);
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
