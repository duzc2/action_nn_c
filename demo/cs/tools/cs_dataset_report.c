#include "cs_tool_common.h"

/*
 * Version 1 dataset_report tool.
 *
 * The report stays intentionally simple: it scans the generated split lists and
 * emits class counts plus a short markdown summary for human review.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct CsReportOptionsTag {
    char dataset_root[CS_TOOL_MAX_PATH];
    char output_root[CS_TOOL_MAX_PATH];
} CsReportOptions;

typedef struct CsSplitCountsTag {
    int counts[CS_TOOL_MAX_PLACES];
} CsSplitCounts;

typedef struct CsDictionaryCountTag {
    char session_id[64];
    int sample_count;
} CsSessionCoverage;

static int cs_arg_is_option(const char* value, const char* expected) {
    return value != NULL && expected != NULL && strcmp(value, expected) == 0;
}

static int cs_report_default_paths(CsReportOptions* options, char* error_buffer, size_t error_buffer_size) {
    char current_directory[CS_TOOL_MAX_PATH];

    if (!cs_tool_get_current_directory(current_directory, sizeof(current_directory), error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_tool_join_two_paths(options->dataset_root, sizeof(options->dataset_root), current_directory, "demo", "cs\\data\\v1_area\\processed") ||
        !cs_tool_join_two_paths(options->output_root, sizeof(options->output_root), current_directory, "demo", "cs\\data\\v1_area\\reports")) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Default report paths are too long.");
        return 0;
    }
    return 1;
}

static int cs_report_parse_options(int argc, char** argv, int start_index, CsReportOptions* options, char* error_buffer, size_t error_buffer_size) {
    int index;

    memset(options, 0, sizeof(*options));
    if (!cs_report_default_paths(options, error_buffer, error_buffer_size)) {
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

        if (cs_arg_is_option(option_name, "--dataset-root")) {
            if (!cs_tool_copy_string(options->dataset_root, sizeof(options->dataset_root), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Dataset root is too long.");
                return 0;
            }
        } else if (cs_arg_is_option(option_name, "--output-root")) {
            if (!cs_tool_copy_string(options->output_root, sizeof(options->output_root), option_value)) {
                cs_tool_set_error(error_buffer, error_buffer_size, "Output root is too long.");
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

static int cs_report_find_place_index(const CsPlaceDictionary* dictionary, int place_id) {
    size_t index;

    for (index = 0U; index < dictionary->entry_count; ++index) {
        if (dictionary->entries[index].place_id == place_id) {
            return (int)index;
        }
    }
    return -1;
}

static int cs_report_find_or_add_session(CsSessionCoverage* coverage, size_t* coverage_count, const char* session_id) {
    size_t index;

    for (index = 0U; index < *coverage_count; ++index) {
        if (strcmp(coverage[index].session_id, session_id) == 0) {
            return (int)index;
        }
    }

    if (*coverage_count >= 256U) {
        return -1;
    }

    memset(&coverage[*coverage_count], 0, sizeof(coverage[*coverage_count]));
    (void)cs_tool_copy_string(coverage[*coverage_count].session_id, sizeof(coverage[*coverage_count].session_id), session_id);
    (*coverage_count)++;
    return (int)(*coverage_count - 1U);
}

static int cs_report_scan_split_file(const char* path,
                                     const CsPlaceDictionary* dictionary,
                                     CsSplitCounts* counts,
                                     CsSessionCoverage* coverage,
                                     size_t* coverage_count,
                                     char* error_buffer,
                                     size_t error_buffer_size) {
    char* text;
    size_t text_size;
    const char* cursor;

    text = NULL;
    text_size = 0U;

    if (!cs_tool_read_text_file(path, &text, &text_size, error_buffer, error_buffer_size)) {
        return 0;
    }

    (void)text_size;
    cursor = text;
    while ((cursor = cs_find_key(cursor, "sample_id")) != NULL) {
        char session_id[64];
        int place_id;
        int place_index;
        int coverage_index;

        if (!cs_extract_string_after_key(cursor, "session_id", session_id, sizeof(session_id)) ||
            !cs_extract_int_after_key(cursor, "place_id", &place_id)) {
            free(text);
            cs_tool_set_error(error_buffer, error_buffer_size, "Sample list is malformed: %s", path);
            return 0;
        }

        place_index = cs_report_find_place_index(dictionary, place_id);
        if (place_index < 0) {
            free(text);
            cs_tool_set_error(error_buffer, error_buffer_size, "Unknown place_id %d in %s", place_id, path);
            return 0;
        }

        counts->counts[place_index]++;
        coverage_index = cs_report_find_or_add_session(coverage, coverage_count, session_id);
        if (coverage_index >= 0) {
            coverage[coverage_index].sample_count++;
        }

        cursor += 9;
    }

    free(text);
    return 1;
}

static int cs_report_write_class_balance(const char* output_root,
                                         const CsPlaceDictionary* dictionary,
                                         const CsSplitCounts* train_counts,
                                         const CsSplitCounts* val_counts,
                                         const CsSplitCounts* test_counts,
                                         char* error_buffer,
                                         size_t error_buffer_size) {
    char path[CS_TOOL_MAX_PATH];
    FILE* file_handle;
    size_t index;

    if (snprintf(path, sizeof(path), "%s\\class_balance.json", output_root) < 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "class_balance.json path is too long.");
        return 0;
    }

    if (!cs_tool_make_dirs(output_root, error_buffer, error_buffer_size)) {
        return 0;
    }

    file_handle = fopen(path, "wb");
    if (file_handle == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open class balance file: %s", path);
        return 0;
    }

    fprintf(file_handle, "{\n  \"train\": {\n");
    for (index = 0U; index < dictionary->entry_count; ++index) {
        fprintf(file_handle, "    \"%s\": %d%s\n",
                dictionary->entries[index].place_token,
                train_counts->counts[index],
                (index + 1U < dictionary->entry_count) ? "," : "");
    }
    fprintf(file_handle, "  },\n  \"val\": {\n");
    for (index = 0U; index < dictionary->entry_count; ++index) {
        fprintf(file_handle, "    \"%s\": %d%s\n",
                dictionary->entries[index].place_token,
                val_counts->counts[index],
                (index + 1U < dictionary->entry_count) ? "," : "");
    }
    fprintf(file_handle, "  },\n  \"test\": {\n");
    for (index = 0U; index < dictionary->entry_count; ++index) {
        fprintf(file_handle, "    \"%s\": %d%s\n",
                dictionary->entries[index].place_token,
                test_counts->counts[index],
                (index + 1U < dictionary->entry_count) ? "," : "");
    }
    fprintf(file_handle, "  }\n}\n");
    fclose(file_handle);
    return 1;
}

static int cs_report_write_session_coverage(const char* output_root,
                                            const CsSessionCoverage* coverage,
                                            size_t coverage_count,
                                            char* error_buffer,
                                            size_t error_buffer_size) {
    char path[CS_TOOL_MAX_PATH];
    FILE* file_handle;
    size_t index;

    if (snprintf(path, sizeof(path), "%s\\session_coverage.json", output_root) < 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "session_coverage.json path is too long.");
        return 0;
    }

    file_handle = fopen(path, "wb");
    if (file_handle == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open session coverage file: %s", path);
        return 0;
    }

    fprintf(file_handle, "{\n  \"sessions\": [\n");
    for (index = 0U; index < coverage_count; ++index) {
        fprintf(file_handle,
                "    {\n"
                "      \"session_id\": \"%s\",\n"
                "      \"sample_count\": %d\n"
                "    }%s\n",
                coverage[index].session_id,
                coverage[index].sample_count,
                (index + 1U < coverage_count) ? "," : "");
    }
    fprintf(file_handle, "  ]\n}\n");
    fclose(file_handle);
    return 1;
}

static int cs_report_write_markdown(const char* output_root,
                                    const CsPlaceDictionary* dictionary,
                                    const CsSplitCounts* train_counts,
                                    const CsSplitCounts* val_counts,
                                    const CsSplitCounts* test_counts,
                                    char* error_buffer,
                                    size_t error_buffer_size) {
    char path[CS_TOOL_MAX_PATH];
    FILE* file_handle;
    size_t index;

    if (snprintf(path, sizeof(path), "%s\\dataset_report.md", output_root) < 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "dataset_report.md path is too long.");
        return 0;
    }

    file_handle = fopen(path, "wb");
    if (file_handle == NULL) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Failed to open markdown report: %s", path);
        return 0;
    }

    fprintf(file_handle, "# Version 1 Dataset Report\n\n");
    fprintf(file_handle, "| place_token | train | val | test |\n");
    fprintf(file_handle, "|---|---:|---:|---:|\n");
    for (index = 0U; index < dictionary->entry_count; ++index) {
        fprintf(file_handle,
                "| %s | %d | %d | %d |\n",
                dictionary->entries[index].place_token,
                train_counts->counts[index],
                val_counts->counts[index],
                test_counts->counts[index]);
    }

    fclose(file_handle);
    return 1;
}

static int cs_report_run(const CsReportOptions* options, char* error_buffer, size_t error_buffer_size) {
    char dictionary_path[CS_TOOL_MAX_PATH];
    char train_path[CS_TOOL_MAX_PATH];
    char val_path[CS_TOOL_MAX_PATH];
    char test_path[CS_TOOL_MAX_PATH];
    CsPlaceDictionary dictionary;
    CsSplitCounts train_counts;
    CsSplitCounts val_counts;
    CsSplitCounts test_counts;
    CsSessionCoverage coverage[256];
    size_t coverage_count;

    memset(&dictionary, 0, sizeof(dictionary));
    memset(&train_counts, 0, sizeof(train_counts));
    memset(&val_counts, 0, sizeof(val_counts));
    memset(&test_counts, 0, sizeof(test_counts));
    memset(coverage, 0, sizeof(coverage));
    coverage_count = 0U;

    if (!cs_tool_find_default_dictionary(dictionary_path, sizeof(dictionary_path), error_buffer, error_buffer_size) ||
        !cs_tool_load_dictionary(dictionary_path, &dictionary, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (snprintf(train_path, sizeof(train_path), "%s\\train_list.json", options->dataset_root) < 0 ||
        snprintf(val_path, sizeof(val_path), "%s\\val_list.json", options->dataset_root) < 0 ||
        snprintf(test_path, sizeof(test_path), "%s\\test_list.json", options->dataset_root) < 0) {
        cs_tool_set_error(error_buffer, error_buffer_size, "Dataset split path is too long.");
        return 0;
    }

    if (!cs_report_scan_split_file(train_path, &dictionary, &train_counts, coverage, &coverage_count, error_buffer, error_buffer_size) ||
        !cs_report_scan_split_file(val_path, &dictionary, &val_counts, coverage, &coverage_count, error_buffer, error_buffer_size) ||
        !cs_report_scan_split_file(test_path, &dictionary, &test_counts, coverage, &coverage_count, error_buffer, error_buffer_size)) {
        return 0;
    }

    if (!cs_report_write_class_balance(options->output_root, &dictionary, &train_counts, &val_counts, &test_counts, error_buffer, error_buffer_size) ||
        !cs_report_write_session_coverage(options->output_root, coverage, coverage_count, error_buffer, error_buffer_size) ||
        !cs_report_write_markdown(options->output_root, &dictionary, &train_counts, &val_counts, &test_counts, error_buffer, error_buffer_size)) {
        return 0;
    }

    printf("Wrote dataset report to %s\n", options->output_root);
    return 1;
}

static void cs_report_print_usage(void) {
    printf("Usage:\n");
    printf("  cs_dataset_report run [--dataset-root <path>] [--output-root <path>]\n");
}

int main(int argc, char** argv) {
    CsReportOptions options;
    char error_buffer[CS_TOOL_MAX_TEXT];
    int ok;

    memset(&options, 0, sizeof(options));
    memset(error_buffer, 0, sizeof(error_buffer));

    if (argc < 2 || strcmp(argv[1], "run") != 0) {
        cs_report_print_usage();
        return 1;
    }

    if (!cs_report_parse_options(argc, argv, 2, &options, error_buffer, sizeof(error_buffer))) {
        fprintf(stderr, "%s\n", error_buffer);
        return 1;
    }

    ok = cs_report_run(&options, error_buffer, sizeof(error_buffer));
    if (!ok) {
        fprintf(stderr, "%s\n", error_buffer);
        return 2;
    }

    return 0;
}
