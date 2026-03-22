#ifndef PROFILER_H
#define PROFILER_H

#include <stddef.h>

typedef struct {
    const char* network_name;
    const char* network_type;
    const char* output_dir;
} ProfilerGenerateRequest;

int profiler_generate(const ProfilerGenerateRequest* request);

typedef struct {
    const char* input_names;
    size_t input_count;
    const char* output_names;
    size_t output_count;
} ProfIONames;

int profiler_generate_with_io(
    const char* network_name,
    const char* network_type,
    const char* output_dir,
    const ProfIONames* io_names
);

#endif
