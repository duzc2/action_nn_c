/**
 * @file prof_codegen.c
 * @brief Code generation implementation
 */

#include "prof_codegen.h"
#include "prof_error.h"
#include "prof_flatten.h"
#include "prof_path.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ABI_VERSION 1
#define CODE_BUFFER_CAPACITY 524288U

#ifdef _WIN32
/**
 * @brief Open one text file for writing without triggering MSVC CRT deprecation warnings.
 */
static FILE* prof_fopen_write_text(const char* path) {
    FILE* fp = NULL;
    if (path == NULL) {
        return NULL;
    }
    if (fopen_s(&fp, path, "w") != 0) {
        return NULL;
    }
    return fp;
}
#else
static FILE* prof_fopen_write_text(const char* path) {
    if (path == NULL) {
        return NULL;
    }
    return fopen(path, "w");
}
#endif

/**
 * @section prof_codegen_pipeline Generated-module emission strategy
 *
 * The profiler does not emit one monolithic source file. Instead it produces a
 * family of focused modules that mirror the documented workflow: tokenizer,
 * runtime initialization, inference, training, save, load, and shared metadata.
 * This file is therefore less about machine-learning math and more about safely
 * translating a validated network definition into deterministic C99 source.
 *
 * Several design choices are worth calling out because they explain many of the
 * helpers below:
 * - all emitters share one flattened leaf-graph view so execution order stays
 *   consistent across metadata, init, train, infer, save, and load modules;
 * - type-specific configuration is emitted as opaque byte arrays so the profiler
 *   never depends on backend-private struct layouts beyond the typed blob name;
 * - every buffer append goes through a single bounded helper so overflow checks
 *   are centralized and failures stop the pipeline immediately;
 * - optional headers are written only when the caller provides explicit target
 *   paths, which preserves the documented no-implicit-output-directory rule.
 */

/**
 * @brief Flattened leaf-graph view shared by all generation stages.
 *
 * Code generation repeatedly needs the same executable-leaf ordering, so the
 * helper struct caches both the flattened subnet list and the topology metadata
 * derived from it.
 */
typedef struct {
    ProfSubnetList leaf_subnets;
    size_t* topology_order;
    size_t* incoming_counts;
    size_t* outgoing_counts;
} ProfLeafGraph;

/**
 * @section prof_codegen_buffer_helpers Buffer-oriented emission primitives
 *
 * The first helper group is intentionally generic: write a finished blob,
 * append formatted text into a bounded buffer, and serialize opaque config
 * bytes. Every later emitter builds on these primitives so overflow handling,
 * I/O failure handling, and byte-for-byte determinism stay centralized.
 */

/**
 * @brief Write one generated source blob to its final destination path.
 */
static ProfStatus write_file(const char* path, const char* content) {
    FILE* fp;

    if (path == NULL || content == NULL) {
        return PROF_STATUS_PATH_INVALID;
    }

    fp = prof_fopen_write_text(path);
    if (fp == NULL) {
        return PROF_STATUS_IO_FAILED;
    }

    (void)fprintf(fp, "%s", content);
    (void)fclose(fp);
    return PROF_STATUS_OK;
}

/**
 * @brief Append formatted text into a bounded generation buffer.
 *
 * Every code-emission helper builds its output incrementally through this one
 * function so buffer-overflow handling stays centralized and consistent. The
 * return convention is intentionally minimal: zero means the append succeeded,
 * while any failure tells the caller to abort generation immediately.
 */
static int append_format(
    char* buffer,
    size_t buffer_capacity,
    size_t* position,
    const char* format,
    ...
) {
    int written;
    size_t remaining;
    va_list args;

    if (buffer == NULL || position == NULL || format == NULL) {
        return -1;
    }
    if (*position >= buffer_capacity) {
        return -1;
    }

    remaining = buffer_capacity - *position;
    va_start(args, format);
    written = vsnprintf(buffer + *position, remaining, format, args);
    va_end(args);

    if (written < 0 || (size_t)written >= remaining) {
        return -1;
    }

    *position += (size_t)written;
    return 0;
}

/**
 * @brief Emit an opaque type-config blob as a static byte array literal.
 *
 * Code generation never interprets backend-private config layouts directly.
 * Serializing them as byte arrays lets the generated C module reconstruct a
 * typed local variable only at the moment it calls the backend contract.
 */
static int append_type_config_blob(
    char* buffer,
    size_t buffer_capacity,
    size_t* position,
    const char* symbol_name,
    const unsigned char* data,
    size_t data_size
) {
    size_t index;

    if (buffer == NULL || position == NULL || symbol_name == NULL) {
        return -1;
    }

    if (append_format(
            buffer,
            buffer_capacity,
            position,
            "static const unsigned char %s[] = {\n",
            symbol_name) != 0) {
        return -1;
    }

    if (data == NULL || data_size == 0U) {
        if (append_format(buffer, buffer_capacity, position, "    0x00\n};\n\n") != 0) {
            return -1;
        }
        return 0;
    }

    for (index = 0U; index < data_size; ++index) {
        if ((index % 12U) == 0U) {
            if (append_format(buffer, buffer_capacity, position, "    ") != 0) {
                return -1;
            }
        }
        if (append_format(
                buffer,
                buffer_capacity,
                position,
                "0x%02X%s",
                (unsigned int)data[index],
                (index + 1U < data_size) ? ", " : "") != 0) {
            return -1;
        }
        if ((index % 12U) == 11U || (index + 1U) == data_size) {
            if (append_format(buffer, buffer_capacity, position, "\n") != 0) {
                return -1;
            }
        }
    }

    if (append_format(buffer, buffer_capacity, position, "};\n\n") != 0) {
        return -1;
    }

    return 0;
}

/**
 * @brief Release every heap allocation owned by a flattened leaf graph view.
 */
static void prof_leaf_graph_cleanup(ProfLeafGraph* graph) {
    if (graph == NULL) {
        return;
    }

    prof_flatten_free_list(&graph->leaf_subnets);
    free(graph->topology_order);
    free(graph->incoming_counts);
    free(graph->outgoing_counts);
    graph->topology_order = NULL;
    graph->incoming_counts = NULL;
    graph->outgoing_counts = NULL;
}

/**
 * @section prof_codegen_topology_helpers Flattened topology derivation
 *
 * The next helper group derives reusable graph facts from the validated
 * network: executable leaf order, source/sink sizes, and packed sink offsets.
 * These values are emitted into multiple generated files, so they must all come
 * from the same cached leaf graph rather than being recomputed ad hoc.
 */

/**
 * @brief Build the flattened executable graph view required by code generation.
 *
 * Generation deliberately materializes this view once per emitted module. That
 * keeps helper code simple, ensures cleanup happens in one place on failure,
 * and guarantees that metadata, infer/train orchestration, and save/load logic
 * are all derived from the same topological facts.
 */
static ProfStatus prof_leaf_graph_init(
    const NN_NetworkDef* network,
    ProfLeafGraph* graph
) {
    ProfStatus st;

    if (network == NULL || graph == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    (void)memset(graph, 0, sizeof(*graph));

    st = prof_flatten_collect_leaf_subnets(network, &graph->leaf_subnets);
    if (st != PROF_STATUS_OK) {
        return st;
    }

    st = prof_flatten_build_leaf_topology(
        network,
        &graph->leaf_subnets,
        &graph->topology_order,
        &graph->incoming_counts,
        &graph->outgoing_counts
    );
    if (st != PROF_STATUS_OK) {
        prof_leaf_graph_cleanup(graph);
        return st;
    }

    return PROF_STATUS_OK;
}

/**
 * @brief Compute the maximum external input width among source leaf subnets.
 *
 * Source leaves are the only leaves that read directly from the caller-provided
 * input vector, so this helper defines the public graph-input width that the
 * generated runtime exposes.
 */
static size_t prof_codegen_network_input_size(const ProfLeafGraph* graph) {
    size_t input_size = 0U;
    size_t leaf_index;

    if (graph == NULL) {
        return 0U;
    }

    for (leaf_index = 0U; leaf_index < graph->leaf_subnets.count; ++leaf_index) {
        if (graph->incoming_counts[leaf_index] == 0U) {
            NNSubnetDef* subnet = graph->leaf_subnets.items[leaf_index];
            if (subnet != NULL && subnet->input_layer_size > input_size) {
                input_size = subnet->input_layer_size;
            }
        }
    }

    return input_size;
}

/**
 * @brief Compute the concatenated output width of sink leaf subnets.
 *
 * Sink leaves contribute to the final caller-visible output vector in packed
 * order. The generated runtime uses this total to size output buffers and to
 * expose a stable network-level output contract.
 */
static size_t prof_codegen_network_output_size(const ProfLeafGraph* graph) {
    size_t output_size = 0U;
    size_t leaf_index;

    if (graph == NULL) {
        return 0U;
    }

    for (leaf_index = 0U; leaf_index < graph->leaf_subnets.count; ++leaf_index) {
        if (graph->outgoing_counts[leaf_index] == 0U) {
            NNSubnetDef* subnet = graph->leaf_subnets.items[leaf_index];
            if (subnet != NULL) {
                output_size += subnet->output_layer_size;
            }
        }
    }

    return output_size;
}

/**
 * @brief Compute the packed output offset assigned to one sink leaf subnet.
 *
 * Non-sink leaves do not contribute directly to the external output tensor, so
 * the offset is defined only over sink leaves encountered in flattened order.
 */
static size_t prof_codegen_sink_offset(const ProfLeafGraph* graph, size_t target_leaf_index) {
    size_t offset = 0U;
    size_t leaf_index;

    if (graph == NULL) {
        return 0U;
    }

    for (leaf_index = 0U; leaf_index < graph->leaf_subnets.count; ++leaf_index) {
        NNSubnetDef* subnet = graph->leaf_subnets.items[leaf_index];
        if (graph->outgoing_counts[leaf_index] != 0U || subnet == NULL) {
            continue;
        }
        if (leaf_index == target_leaf_index) {
            break;
        }
        offset += subnet->output_layer_size;
    }

    return offset;
}

/**
 * @brief Emit constants that describe flattened leaf ordering and graph sizes.
 *
 * These constants form the structural backbone of every generated module. By
 * emitting them once per file from the same flattened graph view, infer/train
 * scheduling and save/load metadata all agree on leaf indexing and tensor sizes.
 */
static int append_leaf_graph_constants(
    char* buffer,
    size_t buffer_capacity,
    size_t* position,
    const NN_NetworkDef* network,
    const ProfLeafGraph* graph
) {
    size_t leaf_index;

    /* Publish the basic graph dimensions first because later arrays depend on them. */
    if (append_format(
            buffer,
            buffer_capacity,
            position,
            "#define GENERATED_LEAF_COUNT %zuU\n"
            "#define GENERATED_CONNECTION_COUNT %zuU\n"
            "#define GENERATED_NETWORK_INPUT_SIZE %zuU\n"
            "#define GENERATED_NETWORK_OUTPUT_SIZE %zuU\n\n",
            graph->leaf_subnets.count,
            network != NULL ? network->connection_count : (size_t)0U,
            prof_codegen_network_input_size(graph),
            prof_codegen_network_output_size(graph)) != 0) {
        return -1;
    }

    /* Topology order drives execution order for graph-mode infer/train code. */
    if (append_format(buffer, buffer_capacity, position,
            "static const size_t g_topology_order[GENERATED_LEAF_COUNT] = {") != 0) {
        return -1;
    }
    for (leaf_index = 0U; leaf_index < graph->leaf_subnets.count; ++leaf_index) {
        if (append_format(buffer, buffer_capacity, position, "%s%zuU",
                leaf_index == 0U ? "" : ", ", graph->topology_order[leaf_index]) != 0) {
            return -1;
        }
    }
    if (append_format(buffer, buffer_capacity, position, "};\n") != 0) {
        return -1;
    }

    /* Incoming and outgoing counts let generated helpers detect sources and sinks cheaply. */
    if (append_format(buffer, buffer_capacity, position,
            "static const size_t g_incoming_counts[GENERATED_LEAF_COUNT] = {") != 0) {
        return -1;
    }
    for (leaf_index = 0U; leaf_index < graph->leaf_subnets.count; ++leaf_index) {
        if (append_format(buffer, buffer_capacity, position, "%s%zuU",
                leaf_index == 0U ? "" : ", ", graph->incoming_counts[leaf_index]) != 0) {
            return -1;
        }
    }
    if (append_format(buffer, buffer_capacity, position, "};\n") != 0) {
        return -1;
    }

    if (append_format(buffer, buffer_capacity, position,
            "static const size_t g_outgoing_counts[GENERATED_LEAF_COUNT] = {") != 0) {
        return -1;
    }
    for (leaf_index = 0U; leaf_index < graph->leaf_subnets.count; ++leaf_index) {
        if (append_format(buffer, buffer_capacity, position, "%s%zuU",
                leaf_index == 0U ? "" : ", ", graph->outgoing_counts[leaf_index]) != 0) {
            return -1;
        }
    }
    if (append_format(buffer, buffer_capacity, position, "};\n") != 0) {
        return -1;
    }

    /* Sink offsets pack multiple terminal leaves into one caller-visible output vector. */
    if (append_format(buffer, buffer_capacity, position,
            "static const size_t g_sink_output_offsets[GENERATED_LEAF_COUNT] = {") != 0) {
        return -1;
    }
    for (leaf_index = 0U; leaf_index < graph->leaf_subnets.count; ++leaf_index) {
        if (append_format(buffer, buffer_capacity, position, "%s%zuU",
                leaf_index == 0U ? "" : ", ",
                graph->outgoing_counts[leaf_index] == 0U ?
                    prof_codegen_sink_offset(graph, leaf_index) : 0U) != 0) {
            return -1;
        }
    }
    if (append_format(buffer, buffer_capacity, position, "};\n\n") != 0) {
        return -1;
    }

    return 0;
}

/**
 * @brief Emit a compact static table that mirrors root-level connection edges.
 *
 * Generated schedulers should not need to walk the original request object at
 * runtime. This table snapshots the validated connection metadata into plain C
 * arrays so infer/train helpers can route values deterministically.
 */
static int append_connection_table(
    char* buffer,
    size_t buffer_capacity,
    size_t* position,
    const NN_NetworkDef* network,
    const ProfLeafGraph* graph
) {
    size_t connection_index;

    /* Reify connection metadata into a tiny POD table consumable by generated helpers. */
    if (append_format(
            buffer,
            buffer_capacity,
            position,
            "typedef struct {\n"
            "    size_t source_leaf_index;\n"
            "    size_t target_leaf_index;\n"
            "    size_t source_node_index;\n"
            "    size_t target_node_index;\n"
            "    int merge_strategy;\n"
            "} GeneratedConnection;\n\n"
            "static const GeneratedConnection g_connections[(GENERATED_CONNECTION_COUNT == 0U) ? 1U : GENERATED_CONNECTION_COUNT] = {\n") != 0) {
        return -1;
    }

    /* Serialize each validated edge in declaration order so debugging stays intuitive. */
    for (connection_index = 0U; connection_index < network->connection_count; ++connection_index) {
        NNConnectionDef* connection = network->connections[connection_index];
        int source_index;
        int target_index;

        if (connection == NULL) {
            if (append_format(buffer, buffer_capacity, position,
                    "    { 0U, 0U, 0U, 0U, %d },\n", (int)NN_MERGE_SUM) != 0) {
                return -1;
            }
            continue;
        }

        source_index = prof_flatten_find_subnet_index(&graph->leaf_subnets, connection->source_subnet_id);
        target_index = prof_flatten_find_subnet_index(&graph->leaf_subnets, connection->target_subnet_id);
        if (append_format(buffer, buffer_capacity, position,
                "    { %zuU, %zuU, %zuU, %zuU, %d },\n",
                source_index >= 0 ? (size_t)source_index : 0U,
                target_index >= 0 ? (size_t)target_index : 0U,
                connection->source_node_index,
                connection->target_node_index,
                (int)connection->merge_strategy) != 0) {
            return -1;
        }
    }

    if (append_format(buffer, buffer_capacity, position, "};\n\n") != 0) {
        return -1;
    }

    return 0;
}

/**
 * @brief Emit static byte arrays for every leaf inference config blob.
 *
 * Type-specific config remains opaque to the profiler. Emitting raw bytes keeps
 * the generator decoupled from backend-private struct layouts while still
 * letting generated code reconstruct typed config objects locally.
 */
static int append_infer_type_blobs(
    char* buffer,
    size_t buffer_capacity,
    size_t* position,
    const ProfLeafGraph* graph
) {
    size_t leaf_index;
    char symbol_name[64];

    for (leaf_index = 0U; leaf_index < graph->leaf_subnets.count; ++leaf_index) {
        NNSubnetDef* subnet = graph->leaf_subnets.items[leaf_index];
        if (subnet == NULL) {
            continue;
        }
        (void)snprintf(symbol_name, sizeof(symbol_name), "g_infer_type_config_bytes_%zu", leaf_index);
        if (append_type_config_blob(buffer, buffer_capacity, position, symbol_name,
                subnet->infer_type_config_data, subnet->infer_type_config_size) != 0) {
            return -1;
        }
    }

    return 0;
}

/**
 * @brief Emit static byte arrays for every leaf training config blob.
 *
 * Training config is emitted separately from inference config because the two
 * contracts may use different typed payloads even for the same backend.
 */
static int append_train_type_blobs(
    char* buffer,
    size_t buffer_capacity,
    size_t* position,
    const ProfLeafGraph* graph
) {
    size_t leaf_index;
    char symbol_name[64];

    for (leaf_index = 0U; leaf_index < graph->leaf_subnets.count; ++leaf_index) {
        NNSubnetDef* subnet = graph->leaf_subnets.items[leaf_index];
        if (subnet == NULL) {
            continue;
        }
        (void)snprintf(symbol_name, sizeof(symbol_name), "g_train_type_config_bytes_%zu", leaf_index);
        if (append_type_config_blob(buffer, buffer_capacity, position, symbol_name,
                subnet->train_type_config_data, subnet->train_type_config_size) != 0) {
            return -1;
        }
    }

    return 0;
}

/**
 * @brief Write a generated header only when the caller configured a path for it.
 *
 * Some outputs in the documented pipeline are optional. This helper preserves
 * that behaviour by treating a missing header path as a no-op instead of an
 * error, while still creating parent directories for explicit destinations.
 */
static ProfStatus write_optional_header(const char* path, const char* content) {
    ProfStatus st;

    if (path == NULL) {
        return PROF_STATUS_OK;
    }

    st = prof_path_ensure_parent_directory(path);
    if (st != PROF_STATUS_OK) {
        return st;
    }
    return write_file(path, content);
}

/**
 * @brief Initialize the codegen context shared by all module emitters.
 */
void prof_codegen_init(
    ProfCodegenContext* ctx,
    const NN_NetworkDef* network,
    const ProfOutputLayout* output_layout,
    uint64_t network_hash,
    uint64_t layout_hash,
    ProfErrorBuffer* error
) {
    ctx->network = network;
    ctx->output_layout = output_layout;
    ctx->network_hash = network_hash;
    ctx->layout_hash = layout_hash;
    ctx->error = error;
}

/**
 * @section prof_codegen_stage_metadata Early-stage module emission
 *
 * Metadata, tokenizer, and network-init emission happen first because they
 * define the stable structural contract that later runtime modules depend on.
 * Even when some of these files are lightweight today, keeping them explicit in
 * the pipeline preserves the documented multi-stage generation workflow.
 */

/**
 * @brief Module stage 1: emit metadata that binds all later modules together.
 *
 * The metadata header is the common contract consumed by generated train,
 * infer, save, and load modules, so it must be emitted before every other file.
 * Only globally shared facts belong here: hashes, ABI tag, network name, and
 * the flattened leaf count used for cross-module compatibility checks.
 */
ProfStatus prof_codegen_metadata(ProfCodegenContext* ctx, const char* metadata_path) {
    FILE* fp;
    ProfStatus st;
    ProfLeafGraph graph;

    if (ctx == NULL || metadata_path == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    /*
     * Flatten once up front so every later emitter can trust the same leaf view.
     * Even though metadata only writes a few macros, using the shared helper here
     * prevents subtle disagreement about executable leaf count.
     */
    st = prof_leaf_graph_init(ctx->network, &graph);
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to flatten leaf graph for metadata generation");
    }

    st = prof_path_ensure_parent_directory(metadata_path);
    if (st != PROF_STATUS_OK) {
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, st,
            "Failed to create directory for metadata output");
    }

    /* Write a plain C header so generated modules can include it directly. */
    /* The header stays intentionally small because it is included nearly everywhere. */
    fp = prof_fopen_write_text(metadata_path);
    if (fp == NULL) {
        prof_leaf_graph_cleanup(&graph);
        return PROF_STATUS_IO_FAILED;
    }

    (void)fprintf(fp, "/* Network metadata */\n\n");
    (void)fprintf(fp, "#ifndef NETWORK_METADATA_H\n#define NETWORK_METADATA_H\n\n");
    (void)fprintf(fp, "#define NETWORK_HASH 0x%016llxULL\n", (unsigned long long)ctx->network_hash);
    (void)fprintf(fp, "#define LAYOUT_HASH 0x%016llxULL\n", (unsigned long long)ctx->layout_hash);
    (void)fprintf(fp, "#define ABI_VERSION %d\n", ABI_VERSION);
    (void)fprintf(fp, "#define NETWORK_NAME \"%s\"\n",
        ctx->network != NULL ? ctx->network->network_name : "unknown");
    (void)fprintf(fp, "#define SUBNET_COUNT %zu\n", graph.leaf_subnets.count);
    (void)fprintf(fp, "\n#endif\n");
    (void)fclose(fp);
    prof_leaf_graph_cleanup(&graph);
    return PROF_STATUS_OK;
}

/**
 * @brief Module stage 2: emit tokenizer glue for external program I/O.
 *
 * The tokenizer module is intentionally lightweight. It primarily captures the
 * flattened graph's external input/output counts in a stable, compilable form.
 * This keeps the generated pipeline complete even when a demo network uses raw
 * float inputs and does not yet need a real text or binary tokenizer.
 */
ProfStatus prof_codegen_tokenizer(ProfCodegenContext* ctx) {
    const ProfOutputLayout* layout;
    ProfLeafGraph graph;
    ProfStatus st;
    char content[4096];
    size_t total_inputs = 0U;
    size_t total_outputs = 0U;
    size_t leaf_index;

    if (ctx == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    layout = ctx->output_layout;
    if (layout == NULL || layout->tokenizer.c_path == NULL) {
        return PROF_STATUS_PATH_INVALID;
    }

    st = prof_leaf_graph_init(ctx->network, &graph);
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to flatten leaf graph for tokenizer generation");
    }

    /*
     * The tokenizer stub only needs aggregate I/O sizes today. We still compute
     * them from the flattened leaf view so the generated file documents the same
     * external contract seen by infer/train code.
     */
    for (leaf_index = 0U; leaf_index < graph.leaf_subnets.count; ++leaf_index) {
        NNSubnetDef* subnet = graph.leaf_subnets.items[leaf_index];
        if (subnet != NULL) {
            total_inputs += subnet->input_layer_size;
            total_outputs += subnet->output_layer_size;
        }
    }

    (void)snprintf(content, sizeof(content),
        "/* tokenizer.c - Tokenizer module */\n"
        "/* Auto-generated by profiler */\n"
        "/* Flattened leaf inputs: %zu, flattened leaf outputs: %zu */\n\n"
        "#include \"tokenizer.h\"\n\n"
        "void* tokenizer_create(size_t input_count, size_t output_count) {\n"
        "    (void)input_count;\n"
        "    (void)output_count;\n"
        "    return 0;\n"
        "}\n\n"
        "void tokenizer_destroy(void* ctx) {\n"
        "    (void)ctx;\n"
        "}\n\n"
        "int tokenizer_encode(void* ctx, const float* raw_input, size_t raw_count) {\n"
        "    (void)ctx;\n"
        "    (void)raw_input;\n"
        "    (void)raw_count;\n"
        "    return 0;\n"
        "}\n\n"
        "int tokenizer_decode(void* ctx, const float* encoded, size_t encoded_count) {\n"
        "    (void)ctx;\n"
        "    (void)encoded;\n"
        "    (void)encoded_count;\n"
        "    return 0;\n"
        "}\n",
        total_inputs,
        total_outputs);

    st = prof_path_ensure_parent_directory(layout->tokenizer.c_path);
    if (st != PROF_STATUS_OK) {
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, st,
            "Failed to create directory for tokenizer.c");
    }

    st = write_file(layout->tokenizer.c_path, content);
    if (st != PROF_STATUS_OK) {
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, st,
            "Failed to write tokenizer.c");
    }

    st = write_optional_header(
        layout->tokenizer.h_path,
        "/* tokenizer.h - Tokenizer interface */\n"
        "#ifndef TOKENIZER_H\n"
        "#define TOKENIZER_H\n\n"
        "#include <stddef.h>\n\n"
        "void* tokenizer_create(size_t input_count, size_t output_count);\n"
        "void tokenizer_destroy(void* ctx);\n"
        "int tokenizer_encode(void* ctx, const float* raw_input, size_t raw_count);\n"
        "int tokenizer_decode(void* ctx, const float* encoded, size_t encoded_count);\n\n"
        "#endif\n"
    );
    prof_leaf_graph_cleanup(&graph);

    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, st,
            "Failed to write tokenizer.h");
    }

    return PROF_STATUS_OK;
}

/**
 * @brief Module stage 3: emit runtime construction code for every leaf subnet.
 *
 * network_init.c is where generated code binds validated graph metadata to the
 * concrete backend factories exposed through the registry contracts. The file is
 * intentionally simple because its main role is to document and preserve the
 * validated leaf inventory rather than implement heavy runtime logic.
 */
ProfStatus prof_codegen_network_init(ProfCodegenContext* ctx) {
    const ProfOutputLayout* layout;
    ProfLeafGraph graph;
    ProfStatus st;
    char content[8192];
    size_t pos = 0U;
    size_t leaf_index;

    if (ctx == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    layout = ctx->output_layout;
    if (layout == NULL || layout->network_init.c_path == NULL) {
        return PROF_STATUS_PATH_INVALID;
    }

    st = prof_leaf_graph_init(ctx->network, &graph);
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to flatten leaf graph for network_init generation");
    }

    /* Start with a tiny prologue because the generated module is mainly descriptive today. */
    if (append_format(content, sizeof(content), &pos,
            "/* network_init.c - Network initialization module */\n"
            "#include \"network_init.h\"\n\n") != 0) {
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to assemble network_init.c");
    }

    /* Emit one comment per leaf so generated output doubles as an execution manifest. */
    for (leaf_index = 0U; leaf_index < graph.leaf_subnets.count; ++leaf_index) {
        NNSubnetDef* subnet = graph.leaf_subnets.items[leaf_index];
        if (subnet != NULL && append_format(content, sizeof(content), &pos,
                "/* Leaf %zu: %s (%s), input=%zu, output=%zu */\n",
                leaf_index,
                subnet->subnet_id,
                subnet->subnet_type,
                subnet->input_layer_size,
                subnet->output_layer_size) != 0) {
            prof_leaf_graph_cleanup(&graph);
            return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
                "Failed to append network_init comments");
        }
    }

    if (append_format(content, sizeof(content), &pos,
            "\nvoid* network_init_create(void) { return 0; }\n"
            "void network_init_destroy(void* ctx) { (void)ctx; }\n"
            "int network_init_weights(void* ctx, void* network_ctx) {\n"
            "    (void)ctx;\n"
            "    (void)network_ctx;\n"
            "    return 0;\n"
            "}\n") != 0) {
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to finalize network_init.c");
    }

    st = prof_path_ensure_parent_directory(layout->network_init.c_path);
    if (st != PROF_STATUS_OK) {
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, st,
            "Failed to create directory for network_init.c");
    }

    st = write_file(layout->network_init.c_path, content);
    if (st != PROF_STATUS_OK) {
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, st,
            "Failed to write network_init.c");
    }

    st = write_optional_header(
        layout->network_init.h_path,
        "/* network_init.h - Network initialization interface */\n"
        "#ifndef NETWORK_INIT_H\n"
        "#define NETWORK_INIT_H\n\n"
        "void* network_init_create(void);\n"
        "void network_init_destroy(void* ctx);\n"
        "int network_init_weights(void* ctx, void* network_ctx);\n\n"
        "#endif\n"
    );
    prof_leaf_graph_cleanup(&graph);

    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, st,
            "Failed to write network_init.h");
    }

    return PROF_STATUS_OK;
}

/**
 * @section prof_codegen_stage_infer Generated inference scaffolding
 *
 * The inference stage emits both reusable helper code and the public runtime
 * entry points that generated applications call. The surrounding comments focus
 * on how validated graph facts are translated into self-contained C99 source.
 */

/**
 * @brief Emit the runtime structs used by generated infer.c.
 *
 * These structs are the generated runtime's bridge between validated graph
 * metadata and the concrete backend contexts created through registry hooks.
 * They deliberately keep both human-readable identity fields and backend-owned
 * pointers because generated save/load and diagnostics need both views.
 */
static int append_generated_infer_structs(
    char* buffer,
    size_t buffer_capacity,
    size_t* position
) {
    /* Emit plain old data structs so generated infer.c remains self-contained C99. */
    return append_format(
        buffer,
        buffer_capacity,
        position,
        "typedef struct {\n"
        "    const char* subnet_id;\n"
        "    const char* subnet_type;\n"
        "    size_t input_size;\n"
        "    size_t output_size;\n"
        "    const NNGraphInferContract* contract;\n"
        "    void* network_ctx;\n"
        "    NNCodegenInferConfig config;\n"
        "    float* input_buffer;\n"
        "    float* output_buffer;\n"
        "    size_t* average_counts;\n"
        "} GeneratedInferLeaf;\n\n"
        "typedef struct {\n"
        "    size_t leaf_count;\n"
        "    GeneratedInferLeaf leaves[GENERATED_LEAF_COUNT];\n"
        "} InferContext;\n\n"
    );
}

/**
 * @brief Emit helper functions used internally by generated infer.c.
 *
 * The helpers encapsulate repetitive buffer-routing work so the generated public
 * API remains small even when the validated graph contains many leaves. They
 * also encode subtle graph semantics such as averaged merges and sink packing,
 * which are easier to audit once in generated helper code than inline at every
 * call site.
 */
static int append_generated_infer_helpers(
    char* buffer,
    size_t buffer_capacity,
    size_t* position
) {
    /*
     * The emitted helper bundle owns four responsibilities: leaf teardown,
     * graph-input staging, connection routing, and sink-output collection.
     * Keeping them adjacent makes the generated infer.c easier to inspect.
     */
    return append_format(
        buffer,
        buffer_capacity,
        position,
        "static void generated_infer_destroy_leaf(GeneratedInferLeaf* leaf) {\n"
        "    if (leaf == 0) return;\n"
        "    if (leaf->contract != 0 && leaf->contract->destroy != 0 && leaf->network_ctx != 0) {\n"
        "        leaf->contract->destroy(leaf->network_ctx);\n"
        "    }\n"
        "    free(leaf->input_buffer);\n"
        "    free(leaf->output_buffer);\n"
        "    free(leaf->average_counts);\n"
        "    leaf->network_ctx = 0;\n"
        "}\n\n"
        "static void generated_infer_prepare_inputs(InferContext* ctx, const float* input) {\n"
        "    size_t leaf_index;\n"
        "    if (ctx == 0) return;\n"
        "    for (leaf_index = 0U; leaf_index < GENERATED_LEAF_COUNT; ++leaf_index) {\n"
        "        GeneratedInferLeaf* leaf = &ctx->leaves[leaf_index];\n"
        "        if (leaf->input_buffer != 0) memset(leaf->input_buffer, 0, leaf->input_size * sizeof(float));\n"
        "        if (leaf->output_buffer != 0) memset(leaf->output_buffer, 0, leaf->output_size * sizeof(float));\n"
        "        if (leaf->average_counts != 0) memset(leaf->average_counts, 0, leaf->input_size * sizeof(size_t));\n"
        "        if (g_incoming_counts[leaf_index] == 0U && input != 0 && leaf->input_buffer != 0) {\n"
        "            size_t input_index;\n"
        "            size_t copy_count = leaf->input_size < GENERATED_NETWORK_INPUT_SIZE ? leaf->input_size : GENERATED_NETWORK_INPUT_SIZE;\n"
        "            for (input_index = 0U; input_index < copy_count; ++input_index) {\n"
        "                leaf->input_buffer[input_index] = input[input_index];\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n\n"
        "static void generated_infer_finalize_average(GeneratedInferLeaf* leaf) {\n"
        "    size_t node_index;\n"
        "    if (leaf == 0 || leaf->input_buffer == 0 || leaf->average_counts == 0) return;\n"
        "    for (node_index = 0U; node_index < leaf->input_size; ++node_index) {\n"
        "        if (leaf->average_counts[node_index] > 1U) {\n"
        "            leaf->input_buffer[node_index] /= (float)leaf->average_counts[node_index];\n"
        "        }\n"
        "    }\n"
        "}\n\n"
        "static void generated_infer_route_outputs(InferContext* ctx, size_t source_leaf_index) {\n"
        "    size_t connection_index;\n"
        "    if (ctx == 0) return;\n"
        "    for (connection_index = 0U; connection_index < GENERATED_CONNECTION_COUNT; ++connection_index) {\n"
        "        const GeneratedConnection* connection = &g_connections[connection_index];\n"
        "        GeneratedInferLeaf* target_leaf;\n"
        "        const GeneratedInferLeaf* source_leaf;\n"
        "        float value;\n"
        "        if (connection->source_leaf_index != source_leaf_index) continue;\n"
        "        source_leaf = &ctx->leaves[connection->source_leaf_index];\n"
        "        target_leaf = &ctx->leaves[connection->target_leaf_index];\n"
        "        if (source_leaf->output_buffer == 0 || target_leaf->input_buffer == 0) continue;\n"
        "        value = source_leaf->output_buffer[connection->source_node_index];\n"
        "        target_leaf->input_buffer[connection->target_node_index] += value;\n"
        "        if (connection->merge_strategy == 2 && target_leaf->average_counts != 0) {\n"
        "            target_leaf->average_counts[connection->target_node_index] += 1U;\n"
        "        } else if (target_leaf->average_counts != 0 && target_leaf->average_counts[connection->target_node_index] == 0U) {\n"
        "            target_leaf->average_counts[connection->target_node_index] = 1U;\n"
        "        }\n"
        "    }\n"
        "}\n\n"
        "static int generated_infer_execute_graph(InferContext* ctx, const float* input) {\n"
        "    size_t order_index;\n"
        "    if (ctx == 0) return -1;\n"
        "    generated_infer_prepare_inputs(ctx, input);\n"
        "    for (order_index = 0U; order_index < GENERATED_LEAF_COUNT; ++order_index) {\n"
        "        size_t leaf_index = g_topology_order[order_index];\n"
        "        GeneratedInferLeaf* leaf = &ctx->leaves[leaf_index];\n"
        "        if (g_incoming_counts[leaf_index] > 0U) generated_infer_finalize_average(leaf);\n"
        "        if (leaf->contract == 0 || leaf->contract->graph_run == 0 || leaf->network_ctx == 0) return -1;\n"
        "        if (leaf->contract->graph_run(leaf->network_ctx, leaf->input_buffer, leaf->output_buffer) != 0) return -1;\n"
        "        generated_infer_route_outputs(ctx, leaf_index);\n"
        "    }\n"
        "    return 0;\n"
        "}\n\n"
        "static void generated_infer_collect_outputs(const InferContext* ctx, float* output) {\n"
        "    size_t leaf_index;\n"
        "    size_t output_offset = 0U;\n"
        "    if (ctx == 0 || output == 0) return;\n"
        "    for (leaf_index = 0U; leaf_index < GENERATED_LEAF_COUNT; ++leaf_index) {\n"
        "        const GeneratedInferLeaf* leaf = &ctx->leaves[leaf_index];\n"
        "        size_t node_index;\n"
        "        if (g_outgoing_counts[leaf_index] != 0U) continue;\n"
        "        for (node_index = 0U; node_index < leaf->output_size; ++node_index) {\n"
        "            output[output_offset + node_index] = leaf->output_buffer[node_index];\n"
        "        }\n"
        "        output_offset += leaf->output_size;\n"
        "    }\n"
        "}\n\n"
    );
}

/**
 * @brief Module stage 4: emit inference orchestration over the flattened graph.
 *
 * infer.c owns the generated scheduler: it routes buffers along validated graph
 * edges and calls backend graph-run hooks in the shared topological order. This
 * is the main bridge between static validation results and executable runtime
 * behaviour, so most emitted helpers focus on preserving graph invariants.
 */
ProfStatus prof_codegen_infer(ProfCodegenContext* ctx) {
    const ProfOutputLayout* layout;
    const NN_NetworkDef* network;
    ProfLeafGraph graph;
    ProfStatus st;
    char* content;
    size_t pos = 0U;
    size_t leaf_index;

    if (ctx == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    layout = ctx->output_layout;
    network = ctx->network;
    if (layout == NULL || layout->infer.c_path == NULL) {
        return PROF_STATUS_PATH_INVALID;
    }

    /*
     * infer.c needs the full flattened executable graph because scheduling,
     * source/sink detection, and buffer routing all depend on the same topology.
     */
    st = prof_leaf_graph_init(network, &graph);
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to flatten leaf graph for infer generation");
    }

    /* Reserve one large bounded buffer so code emission remains deterministic. */
    /* A fixed upper bound also makes overflow handling consistent across emitters. */
    content = (char*)calloc(CODE_BUFFER_CAPACITY, sizeof(char));
    if (content == NULL) {
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to allocate infer code buffer");
    }

    /* Emit file prologue and shared includes before any topology-derived fragments. */
    if (append_format(content, CODE_BUFFER_CAPACITY, &pos,
            "/* infer.c - Inference module */\n"
            "/* Auto-generated by profiler */\n"
            "/* Network: %s */\n"
            "/* Flattened leaf count: %zu */\n\n"
            "#include \"infer.h\"\n"
            "#include \"nn/nn_codegen_hooks.h\"\n"
            "#include \"nn/nn_graph_contract.h\"\n"
            "#include <stdlib.h>\n"
            "#include <string.h>\n\n",
            network != NULL ? network->network_name : "unknown",
            graph.leaf_subnets.count) != 0) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to assemble infer prologue");
    }

    /* Pull in every backend-specific infer config header referenced by a leaf. */
    for (leaf_index = 0U; leaf_index < graph.leaf_subnets.count; ++leaf_index) {
        NNSubnetDef* subnet = graph.leaf_subnets.items[leaf_index];
        if (subnet != NULL && subnet->infer_config_header_path != NULL &&
            append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "#include \"%s\"\n", subnet->infer_config_header_path) != 0) {
            free(content);
            prof_leaf_graph_cleanup(&graph);
            return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
                "Failed to append infer type include");
        }
    }

    /* Append structural constants, routing tables, type blobs, and runtime helpers in order. */
    if (append_format(content, CODE_BUFFER_CAPACITY, &pos, "\n") != 0 ||
        append_leaf_graph_constants(content, CODE_BUFFER_CAPACITY, &pos, network, &graph) != 0 ||
        append_connection_table(content, CODE_BUFFER_CAPACITY, &pos, network, &graph) != 0 ||
        append_infer_type_blobs(content, CODE_BUFFER_CAPACITY, &pos, &graph) != 0 ||
        append_generated_infer_structs(content, CODE_BUFFER_CAPACITY, &pos) != 0 ||
        append_generated_infer_helpers(content, CODE_BUFFER_CAPACITY, &pos) != 0 ||
        append_format(content, CODE_BUFFER_CAPACITY, &pos,
            "void* infer_create(void) {\n"
            "    InferContext* ctx;\n"
            "    ctx = (InferContext*)calloc(1U, sizeof(InferContext));\n"
            "    if (ctx == 0) return 0;\n"
            "    ctx->leaf_count = GENERATED_LEAF_COUNT;\n") != 0) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to assemble infer helper code");
    }

    /* Rehydrate one generated leaf descriptor per flattened executable subnet. */
    for (leaf_index = 0U; leaf_index < graph.leaf_subnets.count; ++leaf_index) {
        NNSubnetDef* subnet = graph.leaf_subnets.items[leaf_index];
        size_t hidden_index;
        if (subnet == NULL) {
            continue;
        }

        if (append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "    ctx->leaves[%zuU].subnet_id = \"%s\";\n"
                "    ctx->leaves[%zuU].subnet_type = \"%s\";\n"
                "    ctx->leaves[%zuU].input_size = %zuU;\n"
                "    ctx->leaves[%zuU].output_size = %zuU;\n"
                "    ctx->leaves[%zuU].contract = nn_graph_infer_contract_find(\"%s\");\n",
                leaf_index, subnet->subnet_id,
                leaf_index, subnet->subnet_type,
                leaf_index, subnet->input_layer_size,
                leaf_index, subnet->output_layer_size,
                leaf_index, subnet->subnet_type) != 0 ||
            append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "    if (ctx->leaves[%zuU].contract == 0 ||\n"
                "        ctx->leaves[%zuU].contract->create == 0 ||\n"
                "        ctx->leaves[%zuU].contract->destroy == 0 ||\n"
                "        ((GENERATED_LEAF_COUNT == 1U) ?\n"
                "            (ctx->leaves[%zuU].contract->auto_run == 0) :\n"
                "            (ctx->leaves[%zuU].contract->graph_run == 0))) {\n"
                "        infer_destroy(ctx);\n"
                "        return 0;\n"
                "    }\n",
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index) != 0 ||
            append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "    ctx->leaves[%zuU].config.network_type = \"%s\";\n"
                "    ctx->leaves[%zuU].config.input_size = %zuU;\n"
                "    ctx->leaves[%zuU].config.hidden_layer_count = %zuU;\n"
                "    ctx->leaves[%zuU].config.output_size = %zuU;\n"
                "    ctx->leaves[%zuU].config.network_hash = 0x%016llxULL;\n"
                "    ctx->leaves[%zuU].config.layout_hash = 0x%016llxULL;\n"
                "    ctx->leaves[%zuU].config.seed = 0U;\n"
                "    ctx->leaves[%zuU].config.type_config = 0;\n"
                "    ctx->leaves[%zuU].config.type_config_size = 0U;\n"
                "    ctx->leaves[%zuU].config.type_config_type_name = 0;\n",
                leaf_index, subnet->subnet_type,
                leaf_index, subnet->input_layer_size,
                leaf_index, subnet->hidden_layer_count,
                leaf_index, subnet->output_layer_size,
                leaf_index, (unsigned long long)ctx->network_hash,
                leaf_index, (unsigned long long)ctx->layout_hash,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index) != 0) {
            free(content);
            prof_leaf_graph_cleanup(&graph);
            return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
                "Failed to append infer leaf configuration");
        }

        /* Emit a fixed-width hidden-size array so generated config stays plain C. */
        for (hidden_index = 0U; hidden_index < 4U; ++hidden_index) {
            size_t hidden_size = hidden_index < subnet->hidden_layer_count ?
                subnet->hidden_layer_sizes[hidden_index] : 0U;
            if (append_format(content, CODE_BUFFER_CAPACITY, &pos,
                    "    ctx->leaves[%zuU].config.hidden_sizes[%zuU] = %zuU;\n",
                    leaf_index, hidden_index, hidden_size) != 0) {
                free(content);
                prof_leaf_graph_cleanup(&graph);
                return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
                    "Failed to append infer hidden layer configuration");
            }
        }

        /* Graph mode needs per-leaf staging buffers; single-leaf mode can delegate directly. */
        if (graph.leaf_subnets.count > 1U &&
            append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "    ctx->leaves[%zuU].input_buffer = (float*)calloc(%zuU, sizeof(float));\n"
                "    ctx->leaves[%zuU].output_buffer = (float*)calloc(%zuU, sizeof(float));\n"
                "    ctx->leaves[%zuU].average_counts = (size_t*)calloc(%zuU, sizeof(size_t));\n"
                "    if (ctx->leaves[%zuU].input_buffer == 0 ||\n"
                "        ctx->leaves[%zuU].output_buffer == 0 ||\n"
                "        ctx->leaves[%zuU].average_counts == 0) {\n"
                "        infer_destroy(ctx);\n"
                "        return 0;\n"
                "    }\n",
                leaf_index, subnet->input_layer_size,
                leaf_index, subnet->output_layer_size,
                leaf_index, subnet->input_layer_size,
                leaf_index,
                leaf_index,
                leaf_index) != 0) {
            free(content);
            prof_leaf_graph_cleanup(&graph);
            return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
                "Failed to append infer buffer allocation");
        }

        /* Reconstruct the typed config blob on the stack and pass it into the backend factory. */
        if (append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "    {\n"
                "        %s typed_config;\n"
                "        (void)memcpy(&typed_config, g_infer_type_config_bytes_%zu, sizeof(typed_config));\n"
                "        ctx->leaves[%zuU].config.type_config = &typed_config;\n"
                "        ctx->leaves[%zuU].config.type_config_size = sizeof(typed_config);\n"
                "        ctx->leaves[%zuU].config.type_config_type_name = \"%s\";\n"
                "        ctx->leaves[%zuU].network_ctx = ctx->leaves[%zuU].contract->create(&ctx->leaves[%zuU].config);\n"
                "    }\n",
                subnet->infer_config_type_name,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index,
                subnet->infer_config_type_name,
                leaf_index,
                leaf_index,
                leaf_index) != 0 ||
            append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "    if (ctx->leaves[%zuU].network_ctx == 0) {\n"
                "        infer_destroy(ctx);\n"
                "        return 0;\n"
                "    }\n",
                leaf_index) != 0) {
            free(content);
            prof_leaf_graph_cleanup(&graph);
            return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
                "Failed to append infer typed configuration");
        }
    }

    /* Finish by emitting lifecycle helpers and the public auto-run entry points. */
    if (append_format(content, CODE_BUFFER_CAPACITY, &pos,
            "    return ctx;\n"
            "}\n\n"
            "void infer_destroy(void* context) {\n"
            "    InferContext* ctx = (InferContext*)context;\n"
            "    size_t leaf_index;\n"
            "    if (ctx == 0) return;\n"
            "    for (leaf_index = 0U; leaf_index < GENERATED_LEAF_COUNT; ++leaf_index) {\n"
            "        generated_infer_destroy_leaf(&ctx->leaves[leaf_index]);\n"
            "    }\n"
            "    free(ctx);\n"
            "}\n\n"
            "void* infer_get_native_context(void* context) {\n"
            "    InferContext* ctx = (InferContext*)context;\n"
            "    if (ctx == 0 || GENERATED_LEAF_COUNT != 1U) return 0;\n"
            "    return ctx->leaves[0U].network_ctx;\n"
            "}\n\n"
            "size_t infer_get_leaf_count(void* context) {\n"
            "    InferContext* ctx = (InferContext*)context;\n"
            "    return ctx == 0 ? 0U : ctx->leaf_count;\n"
            "}\n\n"
            "void* infer_get_leaf_native_context(void* context, size_t leaf_index) {\n"
            "    InferContext* ctx = (InferContext*)context;\n"
            "    if (ctx == 0 || leaf_index >= GENERATED_LEAF_COUNT) return 0;\n"
            "    return ctx->leaves[leaf_index].network_ctx;\n"
            "}\n\n"
            "const char* infer_get_leaf_id(void* context, size_t leaf_index) {\n"
            "    InferContext* ctx = (InferContext*)context;\n"
            "    if (ctx == 0 || leaf_index >= GENERATED_LEAF_COUNT) return 0;\n"
            "    return ctx->leaves[leaf_index].subnet_id;\n"
            "}\n\n"
            "const char* infer_get_leaf_type(void* context, size_t leaf_index) {\n"
            "    InferContext* ctx = (InferContext*)context;\n"
            "    if (ctx == 0 || leaf_index >= GENERATED_LEAF_COUNT) return 0;\n"
            "    return ctx->leaves[leaf_index].subnet_type;\n"
            "}\n\n"
            "int infer_step(void* context, const void* input, void* output) {\n"
            "    return infer_auto_run(context, input, output);\n"
            "}\n\n"
            "int infer_auto_run(void* context, const void* input, void* output) {\n"
            "    InferContext* ctx = (InferContext*)context;\n"
            "    if (ctx == 0) return -1;\n"
            "    if (GENERATED_LEAF_COUNT == 1U) {\n"
            "        GeneratedInferLeaf* leaf = &ctx->leaves[0U];\n"
            "        if (leaf->contract == 0 || leaf->contract->auto_run == 0 || leaf->network_ctx == 0) return -1;\n"
            "        return leaf->contract->auto_run(leaf->network_ctx, input, output);\n"
            "    }\n"
            "    if (generated_infer_execute_graph(ctx, (const float*)input) != 0) return -1;\n"
            "    generated_infer_collect_outputs(ctx, (float*)output);\n"
            "    return 0;\n"
            "}\n") != 0) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to finalize infer.c");
    }

    st = prof_path_ensure_parent_directory(layout->infer.c_path);
    if (st != PROF_STATUS_OK) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, st,
            "Failed to create directory for infer.c");
    }

    st = write_file(layout->infer.c_path, content);
    free(content);
    prof_leaf_graph_cleanup(&graph);
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, st,
            "Failed to write infer.c");
    }

    st = write_optional_header(
        layout->infer.h_path,
        "/* infer.h - Inference interface */\n"
        "#ifndef INFER_H\n"
        "#define INFER_H\n\n"
        "#include <stddef.h>\n\n"
        "void* infer_create(void);\n"
        "void infer_destroy(void* ctx);\n"
        "void* infer_get_native_context(void* ctx);\n"
        "size_t infer_get_leaf_count(void* ctx);\n"
        "void* infer_get_leaf_native_context(void* ctx, size_t leaf_index);\n"
        "const char* infer_get_leaf_id(void* ctx, size_t leaf_index);\n"
        "const char* infer_get_leaf_type(void* ctx, size_t leaf_index);\n"
        "int infer_step(void* ctx, const void* input, void* output);\n"
        "int infer_auto_run(void* ctx, const void* input, void* output);\n\n"
        "#endif\n"
    );
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, st,
            "Failed to write infer.h");
    }

    return PROF_STATUS_OK;
}

/**
 * @section prof_codegen_stage_train Generated training scaffolding
 *
 * Training generation mirrors inference generation but adds optimizer-facing
 * buffers and reverse routing helpers. Keeping the train helpers grouped here
 * makes it easier to verify that forward scheduling and backward scheduling are
 * derived from the same flattened graph.
 */

/**
 * @brief Emit the inference-side mirror structs reused by generated train.c.
 *
 * train.c needs read-only access to the same leaf metadata infer.c already
 * owns, so it mirrors only the fields required for gradient routing and save/load.
 * Keeping the definition centralized here prevents the two generated modules
 * from drifting apart in subtle layout details. The mirror also establishes the
 * exact ownership boundary between infer-owned forward buffers and train-owned
 * gradient buffers.
 */
static int append_generated_train_infer_mirror(char* buffer, size_t buffer_capacity, size_t* position) {
    /* Mirror infer-side fields first, then append train-only gradient state. */
    return append_format(
        buffer,
        buffer_capacity,
        position,
        "typedef struct {\n"
        "    const char* subnet_id;\n"
        "    const char* subnet_type;\n"
        "    size_t input_size;\n"
        "    size_t output_size;\n"
        "    const NNGraphInferContract* contract;\n"
        "    void* network_ctx;\n"
        "    NNCodegenInferConfig config;\n"
        "    float* input_buffer;\n"
        "    float* output_buffer;\n"
        "    size_t* average_counts;\n"
        "} GeneratedInferLeaf;\n\n"
        "typedef struct {\n"
        "    size_t leaf_count;\n"
        "    GeneratedInferLeaf leaves[GENERATED_LEAF_COUNT];\n"
        "} InferContext;\n\n"
        "typedef struct {\n"
        "    const NNGraphTrainContract* contract;\n"
        "    void* train_ctx;\n"
        "    NNCodegenTrainConfig config;\n"
        "    float* output_grad_buffer;\n"
        "    float* input_grad_buffer;\n"
        "} GeneratedTrainLeaf;\n\n"
        "typedef struct {\n"
        "    size_t leaf_count;\n"
        "    size_t epoch_count;\n"
        "    size_t step_count;\n"
        "    float last_loss;\n"
        "    void* infer_owner;\n"
        "    GeneratedTrainLeaf leaves[GENERATED_LEAF_COUNT];\n"
        "    float last_output[GENERATED_NETWORK_OUTPUT_SIZE == 0U ? 1U : GENERATED_NETWORK_OUTPUT_SIZE];\n"
        "} TrainContext;\n\n"
    );
}

/**
 * @brief Emit helper routines used internally by generated train.c.
 *
 * The generated training helpers keep graph-mode loss accumulation, input
 * routing, and leaf cleanup logic out of the public entry points. They also
 * preserve the documented ordering of graph-mode training: forward routing,
 * sink-loss seeding, reverse gradient propagation, and per-leaf cleanup.
 */
static int append_generated_train_helpers(char* buffer, size_t buffer_capacity, size_t* position) {
    /*
     * The helper bundle mirrors the graph-training pipeline stage by stage so
     * the generated public API can read like orchestration code instead of a
     * wall of buffer-manipulation details.
     */
    return append_format(
        buffer,
        buffer_capacity,
        position,
        "static float generated_train_compute_mse(const float* output, const float* target, size_t count) {\n"
        "    float loss = 0.0f;\n"
        "    size_t index;\n"
        "    if (output == 0 || target == 0 || count == 0U) return 0.0f;\n"
        "    for (index = 0U; index < count; ++index) {\n"
        "        float diff = output[index] - target[index];\n"
        "        loss += diff * diff;\n"
        "    }\n"
        "    return loss / (float)count;\n"
        "}\n\n"
        "static void generated_train_prepare_inputs(InferContext* infer_ctx, const float* input) {\n"
        "    size_t leaf_index;\n"
        "    if (infer_ctx == 0) return;\n"
        "    for (leaf_index = 0U; leaf_index < GENERATED_LEAF_COUNT; ++leaf_index) {\n"
        "        GeneratedInferLeaf* leaf = &infer_ctx->leaves[leaf_index];\n"
        "        if (leaf->input_buffer != 0) memset(leaf->input_buffer, 0, leaf->input_size * sizeof(float));\n"
        "        if (leaf->output_buffer != 0) memset(leaf->output_buffer, 0, leaf->output_size * sizeof(float));\n"
        "        if (leaf->average_counts != 0) memset(leaf->average_counts, 0, leaf->input_size * sizeof(size_t));\n"
        "        if (g_incoming_counts[leaf_index] == 0U && input != 0 && leaf->input_buffer != 0) {\n"
        "            size_t input_index;\n"
        "            size_t copy_count = leaf->input_size < GENERATED_NETWORK_INPUT_SIZE ? leaf->input_size : GENERATED_NETWORK_INPUT_SIZE;\n"
        "            for (input_index = 0U; input_index < copy_count; ++input_index) {\n"
        "                leaf->input_buffer[input_index] = input[input_index];\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n\n"
        "static void generated_train_finalize_average(GeneratedInferLeaf* leaf) {\n"
        "    size_t node_index;\n"
        "    if (leaf == 0 || leaf->input_buffer == 0 || leaf->average_counts == 0) return;\n"
        "    for (node_index = 0U; node_index < leaf->input_size; ++node_index) {\n"
        "        if (leaf->average_counts[node_index] > 1U) {\n"
        "            leaf->input_buffer[node_index] /= (float)leaf->average_counts[node_index];\n"
        "        }\n"
        "    }\n"
        "}\n\n"
        "static void generated_train_route_outputs(InferContext* infer_ctx, size_t source_leaf_index) {\n"
        "    size_t connection_index;\n"
        "    if (infer_ctx == 0) return;\n"
        "    for (connection_index = 0U; connection_index < GENERATED_CONNECTION_COUNT; ++connection_index) {\n"
        "        const GeneratedConnection* connection = &g_connections[connection_index];\n"
        "        GeneratedInferLeaf* target_leaf;\n"
        "        const GeneratedInferLeaf* source_leaf;\n"
        "        float value;\n"
        "        if (connection->source_leaf_index != source_leaf_index) continue;\n"
        "        source_leaf = &infer_ctx->leaves[connection->source_leaf_index];\n"
        "        target_leaf = &infer_ctx->leaves[connection->target_leaf_index];\n"
        "        if (source_leaf->output_buffer == 0 || target_leaf->input_buffer == 0) continue;\n"
        "        value = source_leaf->output_buffer[connection->source_node_index];\n"
        "        target_leaf->input_buffer[connection->target_node_index] += value;\n"
        "        if (connection->merge_strategy == 2 && target_leaf->average_counts != 0) {\n"
        "            target_leaf->average_counts[connection->target_node_index] += 1U;\n"
        "        } else if (target_leaf->average_counts != 0 && target_leaf->average_counts[connection->target_node_index] == 0U) {\n"
        "            target_leaf->average_counts[connection->target_node_index] = 1U;\n"
        "        }\n"
        "    }\n"
        "}\n\n"
        "static void generated_train_clear_gradients(TrainContext* ctx) {\n"
        "    size_t leaf_index;\n"
        "    if (ctx == 0) return;\n"
        "    for (leaf_index = 0U; leaf_index < GENERATED_LEAF_COUNT; ++leaf_index) {\n"
        "        GeneratedTrainLeaf* leaf = &ctx->leaves[leaf_index];\n"
        "        GeneratedInferLeaf* infer_leaf = &((InferContext*)ctx->infer_owner)->leaves[leaf_index];\n"
        "        if (leaf->output_grad_buffer != 0) memset(leaf->output_grad_buffer, 0, infer_leaf->output_size * sizeof(float));\n"
        "        if (leaf->input_grad_buffer != 0) memset(leaf->input_grad_buffer, 0, infer_leaf->input_size * sizeof(float));\n"
        "    }\n"
        "}\n\n"
        "static void generated_train_collect_outputs(const InferContext* infer_ctx, float* output) {\n"
        "    size_t leaf_index;\n"
        "    size_t output_offset = 0U;\n"
        "    if (infer_ctx == 0 || output == 0) return;\n"
        "    for (leaf_index = 0U; leaf_index < GENERATED_LEAF_COUNT; ++leaf_index) {\n"
        "        const GeneratedInferLeaf* leaf = &infer_ctx->leaves[leaf_index];\n"
        "        size_t node_index;\n"
        "        if (g_outgoing_counts[leaf_index] != 0U) continue;\n"
        "        for (node_index = 0U; node_index < leaf->output_size; ++node_index) {\n"
        "            output[output_offset + node_index] = leaf->output_buffer[node_index];\n"
        "        }\n"
        "        output_offset += leaf->output_size;\n"
        "    }\n"
        "}\n\n"
        "static void generated_train_seed_output_gradients(TrainContext* ctx, const InferContext* infer_ctx, const float* target) {\n"
        "    size_t leaf_index;\n"
        "    size_t output_offset = 0U;\n"
        "    if (ctx == 0 || infer_ctx == 0 || target == 0) return;\n"
        "    for (leaf_index = 0U; leaf_index < GENERATED_LEAF_COUNT; ++leaf_index) {\n"
        "        const GeneratedInferLeaf* infer_leaf = &infer_ctx->leaves[leaf_index];\n"
        "        GeneratedTrainLeaf* train_leaf = &ctx->leaves[leaf_index];\n"
        "        size_t node_index;\n"
        "        if (g_outgoing_counts[leaf_index] != 0U || train_leaf->output_grad_buffer == 0 || infer_leaf->output_buffer == 0) continue;\n"
        "        for (node_index = 0U; node_index < infer_leaf->output_size; ++node_index) {\n"
        "            float diff = infer_leaf->output_buffer[node_index] - target[output_offset + node_index];\n"
        "            train_leaf->output_grad_buffer[node_index] = (2.0f * diff) / (float)GENERATED_NETWORK_OUTPUT_SIZE;\n"
        "        }\n"
        "        output_offset += infer_leaf->output_size;\n"
        "    }\n"
        "}\n\n"
        "static void generated_train_route_input_gradients(TrainContext* ctx, const InferContext* infer_ctx, size_t target_leaf_index) {\n"
        "    size_t connection_index;\n"
        "    if (ctx == 0 || infer_ctx == 0) return;\n"
        "    for (connection_index = 0U; connection_index < GENERATED_CONNECTION_COUNT; ++connection_index) {\n"
        "        const GeneratedConnection* connection = &g_connections[connection_index];\n"
        "        const GeneratedInferLeaf* target_infer_leaf;\n"
        "        const GeneratedTrainLeaf* target_train_leaf;\n"
        "        GeneratedTrainLeaf* source_train_leaf;\n"
        "        float grad_value;\n"
        "        size_t avg_count = 1U;\n"
        "        if (connection->target_leaf_index != target_leaf_index) continue;\n"
        "        target_infer_leaf = &infer_ctx->leaves[target_leaf_index];\n"
        "        target_train_leaf = &ctx->leaves[target_leaf_index];\n"
        "        source_train_leaf = &ctx->leaves[connection->source_leaf_index];\n"
        "        if (target_train_leaf->input_grad_buffer == 0 || source_train_leaf->output_grad_buffer == 0) continue;\n"
        "        grad_value = target_train_leaf->input_grad_buffer[connection->target_node_index];\n"
        "        if (connection->merge_strategy == 2 && target_infer_leaf->average_counts != 0) {\n"
        "            avg_count = target_infer_leaf->average_counts[connection->target_node_index];\n"
        "            if (avg_count == 0U) avg_count = 1U;\n"
        "            grad_value /= (float)avg_count;\n"
        "        }\n"
        "        source_train_leaf->output_grad_buffer[connection->source_node_index] += grad_value;\n"
        "    }\n"
        "}\n\n"
    );
}

/**
 * @brief Module stage 5: emit training orchestration and graph backprop glue.
 *
 * The generated training module mirrors inference scheduling but layers in
 * gradient buffers, backward routing, and loss tracking. Keeping its structure
 * close to infer.c reduces the chance that forward and backward graph views
 * diverge over time.
 */
ProfStatus prof_codegen_train(ProfCodegenContext* ctx) {
    const ProfOutputLayout* layout;
    const NN_NetworkDef* network;
    ProfLeafGraph graph;
    ProfStatus st;
    char* content;
    size_t pos = 0U;
    size_t leaf_index;

    if (ctx == NULL) return PROF_STATUS_INVALID_ARGUMENT;
    layout = ctx->output_layout;
    network = ctx->network;
    if (layout == NULL || layout->train.c_path == NULL) return PROF_STATUS_PATH_INVALID;

    st = prof_leaf_graph_init(network, &graph);
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to flatten leaf graph for train generation");
    }

    /* Reuse the same bounded-buffer emission strategy as infer.c for consistency. */
    content = (char*)calloc(CODE_BUFFER_CAPACITY, sizeof(char));
    if (content == NULL) {
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to allocate train code buffer");
    }

    /* Emit shared includes first; later fragments assume these contracts are already visible. */
    if (append_format(content, CODE_BUFFER_CAPACITY, &pos,
            "/* train.c - Training module */\n"
            "#include \"train.h\"\n"
            "#include \"infer.h\"\n"
            "#include \"nn/nn_codegen_hooks.h\"\n"
            "#include \"nn/nn_graph_contract.h\"\n"
            "#include <stdlib.h>\n"
            "#include <string.h>\n\n") != 0) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to assemble train prologue");
    }

    /* Pull in every backend-specific training config header referenced by a leaf. */
    for (leaf_index = 0U; leaf_index < graph.leaf_subnets.count; ++leaf_index) {
        NNSubnetDef* subnet = graph.leaf_subnets.items[leaf_index];
        if (subnet != NULL && subnet->train_config_header_path != NULL &&
            append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "#include \"%s\"\n", subnet->train_config_header_path) != 0) {
            free(content);
            prof_leaf_graph_cleanup(&graph);
            return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
                "Failed to append train type include");
        }
    }

    /* Append the same structural facts used by inference, then layer on train-only helpers. */
    if (append_format(content, CODE_BUFFER_CAPACITY, &pos, "\n") != 0 ||
        append_leaf_graph_constants(content, CODE_BUFFER_CAPACITY, &pos, network, &graph) != 0 ||
        append_connection_table(content, CODE_BUFFER_CAPACITY, &pos, network, &graph) != 0 ||
        append_train_type_blobs(content, CODE_BUFFER_CAPACITY, &pos, &graph) != 0 ||
        append_generated_train_infer_mirror(content, CODE_BUFFER_CAPACITY, &pos) != 0 ||
        append_generated_train_helpers(content, CODE_BUFFER_CAPACITY, &pos) != 0 ||
        append_format(content, CODE_BUFFER_CAPACITY, &pos,
            "void* train_create(void* infer_ctx) {\n"
            "    TrainContext* ctx;\n"
            "    if (infer_ctx == 0) return 0;\n"
            "    ctx = (TrainContext*)calloc(1U, sizeof(TrainContext));\n"
            "    if (ctx == 0) return 0;\n"
            "    ctx->leaf_count = GENERATED_LEAF_COUNT;\n"
            "    ctx->infer_owner = infer_ctx;\n") != 0) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to assemble train helper code");
    }

    /* Bind each generated training leaf to the corresponding registered backend contract. */
    for (leaf_index = 0U; leaf_index < graph.leaf_subnets.count; ++leaf_index) {
        NNSubnetDef* subnet = graph.leaf_subnets.items[leaf_index];
        if (subnet == NULL) {
            continue;
        }
        if (append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "    ctx->leaves[%zuU].contract = nn_graph_train_contract_find(\"%s\");\n"
                "    if (ctx->leaves[%zuU].contract == 0 ||\n"
                "        ctx->leaves[%zuU].contract->create == 0 ||\n"
                "        ctx->leaves[%zuU].contract->destroy == 0 ||\n"
                "        ((GENERATED_LEAF_COUNT == 1U) ?\n"
                "            (ctx->leaves[%zuU].contract->step_with_data == 0) :\n"
                "            (ctx->leaves[%zuU].contract->step_with_output_gradient == 0))) {\n"
                "        train_destroy(ctx);\n"
                "        return 0;\n"
                "    }\n",
                leaf_index, subnet->subnet_type,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index) != 0 ||
            append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "    ctx->leaves[%zuU].output_grad_buffer = (float*)calloc(%zuU, sizeof(float));\n"
                "    ctx->leaves[%zuU].input_grad_buffer = (float*)calloc(%zuU, sizeof(float));\n"
                "    if (ctx->leaves[%zuU].output_grad_buffer == 0 ||\n"
                "        ctx->leaves[%zuU].input_grad_buffer == 0) {\n"
                "        train_destroy(ctx);\n"
                "        return 0;\n"
                "    }\n",
                leaf_index, subnet->output_layer_size,
                leaf_index, subnet->input_layer_size,
                leaf_index,
                leaf_index) != 0 ||
            append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "    ctx->leaves[%zuU].config.learning_rate = 0.0f;\n"
                "    ctx->leaves[%zuU].config.momentum = 0.0f;\n"
                "    ctx->leaves[%zuU].config.weight_decay = 0.0f;\n"
                "    ctx->leaves[%zuU].config.batch_size = 1U;\n"
                "    ctx->leaves[%zuU].config.seed = 0U;\n"
                "    ctx->leaves[%zuU].config.type_config = 0;\n"
                "    ctx->leaves[%zuU].config.type_config_size = 0U;\n"
                "    ctx->leaves[%zuU].config.type_config_type_name = 0;\n",
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index) != 0 ||
            /* Rehydrate typed training config and create the backend-specific train context. */
            append_format(content, CODE_BUFFER_CAPACITY, &pos,
                "    {\n"
                "        %s typed_config;\n"
                "        (void)memcpy(&typed_config, g_train_type_config_bytes_%zu, sizeof(typed_config));\n"
                "        ctx->leaves[%zuU].config.type_config = &typed_config;\n"
                "        ctx->leaves[%zuU].config.type_config_size = sizeof(typed_config);\n"
                "        ctx->leaves[%zuU].config.type_config_type_name = \"%s\";\n"
                "        ctx->leaves[%zuU].train_ctx = ctx->leaves[%zuU].contract->create(\n"
                "            infer_get_leaf_native_context(infer_ctx, %zuU),\n"
                "            &ctx->leaves[%zuU].config);\n"
                "    }\n"
                "    if (ctx->leaves[%zuU].train_ctx == 0) {\n"
                "        train_destroy(ctx);\n"
                "        return 0;\n"
                "    }\n",
                subnet->train_config_type_name,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index,
                subnet->train_config_type_name,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index,
                leaf_index) != 0) {
            free(content);
            prof_leaf_graph_cleanup(&graph);
            return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
                "Failed to append train leaf configuration");
        }
    }

    /* Emit lifecycle, training-step, epoch, and loss access helpers as the public API. */
    if (append_format(content, CODE_BUFFER_CAPACITY, &pos,
            "    return ctx;\n}\n\n"
            "void train_destroy(void* context) {\n"
            "    TrainContext* ctx = (TrainContext*)context;\n"
            "    size_t leaf_index;\n"
            "    if (ctx == 0) return;\n"
            "    for (leaf_index = 0U; leaf_index < GENERATED_LEAF_COUNT; ++leaf_index) {\n"
            "        if (ctx->leaves[leaf_index].contract != 0 && ctx->leaves[leaf_index].contract->destroy != 0 && ctx->leaves[leaf_index].train_ctx != 0) {\n"
            "            ctx->leaves[leaf_index].contract->destroy(ctx->leaves[leaf_index].train_ctx);\n"
            "        }\n"
            "        free(ctx->leaves[leaf_index].output_grad_buffer);\n"
            "        free(ctx->leaves[leaf_index].input_grad_buffer);\n"
            "    }\n"
            "    free(ctx);\n}\n\n"
            "int train_step(void* context, const void* input, const void* target) {\n"
            "    TrainContext* ctx = (TrainContext*)context;\n"
            "    if (ctx == 0) return -1;\n"
            "    if (GENERATED_LEAF_COUNT == 1U) {\n"
            "        if (ctx->leaves[0U].contract == 0 || ctx->leaves[0U].contract->step_with_data == 0 || ctx->leaves[0U].train_ctx == 0) return -1;\n"
            "        if (ctx->leaves[0U].contract->step_with_data(ctx->leaves[0U].train_ctx, input, target) != 0) return -1;\n"
            "    } else {\n"
            "        InferContext* infer_ctx = (InferContext*)ctx->infer_owner;\n"
            "        const float* input_values = (const float*)input;\n"
            "        const float* target_values = (const float*)target;\n"
            "        size_t order_index;\n"
            "        if (infer_ctx == 0 || target_values == 0) return -1;\n"
            "        generated_train_prepare_inputs(infer_ctx, input_values);\n"
            "        for (order_index = 0U; order_index < GENERATED_LEAF_COUNT; ++order_index) {\n"
            "            size_t idx = g_topology_order[order_index];\n"
            "            GeneratedInferLeaf* infer_leaf = &infer_ctx->leaves[idx];\n"
            "            if (g_incoming_counts[idx] > 0U) generated_train_finalize_average(infer_leaf);\n"
            "            if (infer_leaf->contract == 0 || infer_leaf->contract->graph_run == 0 || infer_leaf->network_ctx == 0) return -1;\n"
            "            if (infer_leaf->contract->graph_run(infer_leaf->network_ctx, infer_leaf->input_buffer, infer_leaf->output_buffer) != 0) return -1;\n"
            "            generated_train_route_outputs(infer_ctx, idx);\n"
            "        }\n"
            "        generated_train_collect_outputs(infer_ctx, ctx->last_output);\n"
            "        ctx->last_loss = generated_train_compute_mse(ctx->last_output, target_values, GENERATED_NETWORK_OUTPUT_SIZE);\n"
            "        generated_train_clear_gradients(ctx);\n"
            "        generated_train_seed_output_gradients(ctx, infer_ctx, target_values);\n"
            "        for (order_index = GENERATED_LEAF_COUNT; order_index > 0U; --order_index) {\n"
            "            size_t idx = g_topology_order[order_index - 1U];\n"
            "            GeneratedInferLeaf* infer_leaf = &infer_ctx->leaves[idx];\n"
            "            GeneratedTrainLeaf* train_leaf = &ctx->leaves[idx];\n"
            "            if (train_leaf->contract == 0 || train_leaf->contract->step_with_output_gradient == 0 || train_leaf->train_ctx == 0) return -1;\n"
            "            if (train_leaf->contract->step_with_output_gradient(\n"
            "                    train_leaf->train_ctx,\n"
            "                    infer_leaf->input_buffer,\n"
            "                    train_leaf->output_grad_buffer,\n"
            "                    train_leaf->input_grad_buffer) != 0) {\n"
            "                return -1;\n"
            "            }\n"
            "            generated_train_route_input_gradients(ctx, infer_ctx, idx);\n"
            "        }\n"
            "    }\n"
            "    ctx->step_count += 1U;\n"
            "    return 0;\n}\n\n"
            "int train_epoch(void* context, int epoch) {\n"
            "    TrainContext* ctx = (TrainContext*)context;\n"
            "    (void)epoch;\n"
            "    if (ctx == 0) return -1;\n"
            "    ctx->epoch_count += 1U;\n"
            "    return 0;\n}\n\n"
            "int train_auto_run(void* context, int epochs) {\n"
            "    TrainContext* ctx = (TrainContext*)context;\n"
            "    int index;\n"
            "    if (ctx == 0) return -1;\n"
            "    for (index = 0; index < epochs; ++index) {\n"
            "        if (train_epoch(ctx, index) != 0) return -1;\n"
            "    }\n"
            "    return 0;\n}\n\n"
            "float train_get_loss(void* context) {\n"
            "    TrainContext* ctx = (TrainContext*)context;\n"
            "    if (ctx == 0) return 0.0f;\n"
            "    if (GENERATED_LEAF_COUNT == 1U) {\n"
            "        size_t epochs = 0U;\n"
            "        size_t steps = 0U;\n"
            "        float avg_loss = 0.0f;\n"
            "        if (ctx->leaves[0U].contract != 0 && ctx->leaves[0U].contract->get_stats != 0) {\n"
            "            ctx->leaves[0U].contract->get_stats(ctx->leaves[0U].train_ctx, &epochs, &steps, &avg_loss);\n"
            "        }\n"
            "        return avg_loss;\n"
            "    }\n"
            "    return ctx->last_loss;\n}\n") != 0) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to finalize train.c");
    }

    st = prof_path_ensure_parent_directory(layout->train.c_path);
    if (st != PROF_STATUS_OK) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, st,
            "Failed to create directory for train.c");
    }

    st = write_file(layout->train.c_path, content);
    free(content);
    prof_leaf_graph_cleanup(&graph);
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, st,
            "Failed to write train.c");
    }

    st = write_optional_header(
        layout->train.h_path,
        "/* train.h - Training interface */\n"
        "#ifndef TRAIN_H\n"
        "#define TRAIN_H\n\n"
        "void* train_create(void* infer_ctx);\n"
        "void train_destroy(void* ctx);\n"
        "int train_step(void* ctx, const void* input, const void* target);\n"
        "int train_epoch(void* ctx, int epoch);\n"
        "int train_auto_run(void* ctx, int epochs);\n"
        "float train_get_loss(void* ctx);\n\n"
        "#endif\n"
    );
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, st,
            "Failed to write train.h");
    }

    return PROF_STATUS_OK;
}

/**
 * @section prof_codegen_stage_weights Weight compatibility scaffolding
 *
 * Save/load emission is split into shared structural arrays plus two runtime
 * modules. This separation lets both directions reuse the exact same notion of
 * leaf identity, topology order, and compatibility metadata.
 */

/**
 * @brief Emit shared weight-array metadata used by save/load modules.
 *
 * Save and load modules both need the same expected leaf IDs, types, and stable
 * topological order so they can serialize and deserialize compatible files.
 * Centralizing these arrays avoids the risk that one module silently encodes a
 * different notion of leaf identity from the other.
 */
static int append_weight_runtime_arrays(
    char* buffer,
    size_t buffer_capacity,
    size_t* position,
    const NN_NetworkDef* network,
    const ProfLeafGraph* graph
) {
    size_t order_index;

    /* Reuse the standard graph constants so save/load share the same leaf numbering. */
    if (append_leaf_graph_constants(buffer, buffer_capacity, position, network, graph) != 0) {
        return -1;
    }
    if (append_format(buffer, buffer_capacity, position,
            "static const char* g_expected_leaf_ids[GENERATED_LEAF_COUNT] = {") != 0) {
        return -1;
    }
    for (order_index = 0U; order_index < graph->leaf_subnets.count; ++order_index) {
        NNSubnetDef* subnet = graph->leaf_subnets.items[order_index];
        if (append_format(buffer, buffer_capacity, position, "%s\"%s\"",
                order_index == 0U ? "" : ", ", subnet != NULL ? subnet->subnet_id : "") != 0) {
            return -1;
        }
    }
    if (append_format(buffer, buffer_capacity, position, "};\n") != 0) {
        return -1;
    }
    if (append_format(buffer, buffer_capacity, position,
            "static const char* g_expected_leaf_types[GENERATED_LEAF_COUNT] = {") != 0) {
        return -1;
    }
    for (order_index = 0U; order_index < graph->leaf_subnets.count; ++order_index) {
        NNSubnetDef* subnet = graph->leaf_subnets.items[order_index];
        if (append_format(buffer, buffer_capacity, position, "%s\"%s\"",
                order_index == 0U ? "" : ", ", subnet != NULL ? subnet->subnet_type : "") != 0) {
            return -1;
        }
    }
    if (append_format(buffer, buffer_capacity, position, "};\n\n") != 0) {
        return -1;
    }
    return 0;
}

/**
 * @brief Module stage 6: emit parameter serialization code with hash metadata.
 *
 * Save code must mirror the exact leaf ordering used everywhere else so weight
 * files remain compatible with the generated loader and runtime metadata. The
 * emitted file writes a small global header first, then one leaf record per
 * topology slot, then delegates payload serialization to each backend contract.
 */
ProfStatus prof_codegen_weights_save(ProfCodegenContext* ctx) {
    const ProfOutputLayout* layout;
    ProfLeafGraph graph;
    ProfStatus st;
    char* content;
    size_t pos = 0U;

    if (ctx == NULL) return PROF_STATUS_INVALID_ARGUMENT;
    layout = ctx->output_layout;
    if (layout == NULL || layout->weights_save.c_path == NULL) return PROF_STATUS_PATH_INVALID;

    st = prof_leaf_graph_init(ctx->network, &graph);
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to flatten leaf graph for weights_save generation");
    }

    content = (char*)calloc(CODE_BUFFER_CAPACITY, sizeof(char));
    if (content == NULL) {
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to allocate weights_save code buffer");
    }

    /* Emit one compact module that writes metadata followed by ordered leaf payloads. */
    /* The generated save path never needs the original request object at runtime. */
    if (append_format(content, CODE_BUFFER_CAPACITY, &pos,
            "/* weights_save.c - Weights saving module */\n"
            "#include \"weights_save.h\"\n"
            "#include \"infer.h\"\n"
            "#include \"nn/nn_graph_contract.h\"\n"
            "#include <stdint.h>\n"
            "#include <stdio.h>\n"
            "#include <string.h>\n\n"
            "typedef struct { uint64_t network_hash; uint64_t layout_hash; uint32_t abi_version; uint32_t leaf_count; } GeneratedWeightsHeader;\n"
            "typedef struct { uint32_t id_length; uint32_t type_length; } GeneratedLeafHeader;\n\n"
            "#ifdef _WIN32\n"
            "static FILE* generated_open_file_write(const char* file_path) {\n"
            "    FILE* fp = 0;\n"
            "    if (file_path == 0) return 0;\n"
            "    if (fopen_s(&fp, file_path, \"wb\") != 0) return 0;\n"
            "    return fp;\n"
            "}\n"
            "#else\n"
            "static FILE* generated_open_file_write(const char* file_path) {\n"
            "    if (file_path == 0) return 0;\n"
            "    return fopen(file_path, \"wb\");\n"
            "}\n"
            "#endif\n\n") != 0 ||
        append_weight_runtime_arrays(content, CODE_BUFFER_CAPACITY, &pos, ctx->network, &graph) != 0 ||
        append_format(content, CODE_BUFFER_CAPACITY, &pos,
            "int weights_save_to_file(void* infer_ctx, const char* file_path) {\n"
            "    FILE* fp;\n"
            "    GeneratedWeightsHeader header;\n"
            "    size_t order_index;\n"
            "    if (infer_ctx == 0 || file_path == 0) return -1;\n"
            "    fp = generated_open_file_write(file_path);\n"
            "    if (fp == 0) return -3;\n"
            "    header.network_hash = 0x%016llxULL;\n"
            "    header.layout_hash = 0x%016llxULL;\n"
            "    header.abi_version = 1U;\n"
            "    header.leaf_count = (uint32_t)GENERATED_LEAF_COUNT;\n"
            "    if (fwrite(&header, sizeof(header), 1, fp) != 1) { fclose(fp); return -4; }\n"
            "    for (order_index = 0U; order_index < GENERATED_LEAF_COUNT; ++order_index) {\n"
            "        size_t leaf_index = g_topology_order[order_index];\n"
            "        const char* leaf_id = infer_get_leaf_id(infer_ctx, leaf_index);\n"
            "        const char* leaf_type = infer_get_leaf_type(infer_ctx, leaf_index);\n"
            "        void* native_ctx = infer_get_leaf_native_context(infer_ctx, leaf_index);\n"
            "        const NNGraphInferContract* contract;\n"
            "        GeneratedLeafHeader leaf_header;\n"
            "        if (leaf_id == 0 || leaf_type == 0 || native_ctx == 0) { fclose(fp); return -5; }\n"
            "        contract = nn_graph_infer_contract_find(leaf_type);\n"
            "        if (contract == 0 || contract->save_weights == 0) { fclose(fp); return -6; }\n"
            "        leaf_header.id_length = (uint32_t)strlen(leaf_id);\n"
            "        leaf_header.type_length = (uint32_t)strlen(leaf_type);\n"
            "        if (fwrite(&leaf_header, sizeof(leaf_header), 1, fp) != 1 ||\n"
            "            fwrite(leaf_id, 1, leaf_header.id_length, fp) != leaf_header.id_length ||\n"
            "            fwrite(leaf_type, 1, leaf_header.type_length, fp) != leaf_header.type_length) {\n"
            "            fclose(fp); return -7;\n"
            "        }\n"
            "        if (contract->save_weights(native_ctx, fp) == 0) { fclose(fp); return -8; }\n"
            "    }\n"
            "    fclose(fp);\n"
            "    return 0;\n"
            "}\n",
            (unsigned long long)ctx->network_hash,
            (unsigned long long)ctx->layout_hash) != 0) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to assemble weights_save.c");
    }

    st = prof_path_ensure_parent_directory(layout->weights_save.c_path);
    if (st != PROF_STATUS_OK) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, st,
            "Failed to create directory for weights_save.c");
    }

    st = write_file(layout->weights_save.c_path, content);
    free(content);
    prof_leaf_graph_cleanup(&graph);
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, st,
            "Failed to write weights_save.c");
    }

    st = write_optional_header(
        layout->weights_save.h_path,
        "/* weights_save.h - Weights saving interface */\n"
        "#ifndef WEIGHTS_SAVE_H\n"
        "#define WEIGHTS_SAVE_H\n\n"
        "int weights_save_to_file(void* infer_ctx, const char* file_path);\n\n"
        "#endif\n"
    );
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, st,
            "Failed to write weights_save.h");
    }

    return PROF_STATUS_OK;
}

/**
 * @brief Module stage 7: emit guarded weight-loading code with hash checks.
 *
 * The loader validates hashes, ABI version, leaf count, leaf IDs, and leaf
 * types before letting any backend-specific deserializer consume file data.
 * This keeps compatibility failures deterministic and prevents partially loaded
 * weights from corrupting otherwise valid runtime contexts.
 */
ProfStatus prof_codegen_weights_load(ProfCodegenContext* ctx) {
    const ProfOutputLayout* layout;
    ProfLeafGraph graph;
    ProfStatus st;
    char* content;
    size_t pos = 0U;

    if (ctx == NULL) return PROF_STATUS_INVALID_ARGUMENT;
    layout = ctx->output_layout;
    if (layout == NULL || layout->weights_load.c_path == NULL) return PROF_STATUS_PATH_INVALID;

    st = prof_leaf_graph_init(ctx->network, &graph);
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to flatten leaf graph for weights_load generation");
    }

    content = (char*)calloc(CODE_BUFFER_CAPACITY, sizeof(char));
    if (content == NULL) {
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to allocate weights_load code buffer");
    }

    /* Emit a guarded loader that validates metadata before touching leaf state. */
    /* validate_only mode reuses the same parser so tooling sees identical checks. */
    if (append_format(content, CODE_BUFFER_CAPACITY, &pos,
            "/* weights_load.c - Weights loading module */\n"
            "#include \"weights_load.h\"\n"
            "#include \"infer.h\"\n"
            "#include \"nn/nn_graph_contract.h\"\n"
            "#include <stdint.h>\n"
            "#include <stdio.h>\n"
            "#include <string.h>\n\n"
            "typedef struct { uint64_t network_hash; uint64_t layout_hash; uint32_t abi_version; uint32_t leaf_count; } GeneratedWeightsHeader;\n"
            "typedef struct { uint32_t id_length; uint32_t type_length; } GeneratedLeafHeader;\n\n"
            "#ifdef _WIN32\n"
            "static FILE* generated_open_file_read(const char* file_path) {\n"
            "    FILE* fp = 0;\n"
            "    if (file_path == 0) return 0;\n"
            "    if (fopen_s(&fp, file_path, \"rb\") != 0) return 0;\n"
            "    return fp;\n"
            "}\n"
            "#else\n"
            "static FILE* generated_open_file_read(const char* file_path) {\n"
            "    if (file_path == 0) return 0;\n"
            "    return fopen(file_path, \"rb\");\n"
            "}\n"
            "#endif\n\n") != 0 ||
        append_weight_runtime_arrays(content, CODE_BUFFER_CAPACITY, &pos, ctx->network, &graph) != 0 ||
        append_format(content, CODE_BUFFER_CAPACITY, &pos,
            "static int generated_weights_load_internal(void* infer_ctx, const char* file_path, int validate_only) {\n"
            "    FILE* fp;\n"
            "    GeneratedWeightsHeader header;\n"
            "    size_t order_index;\n"
            "    if (file_path == 0) return -1;\n"
            "    fp = generated_open_file_read(file_path);\n"
            "    if (fp == 0) return -3;\n"
            "    if (fread(&header, sizeof(header), 1, fp) != 1) { fclose(fp); return -4; }\n"
            "    if (header.network_hash != 0x%016llxULL) { fclose(fp); return -5; }\n"
            "    if (header.layout_hash != 0x%016llxULL) { fclose(fp); return -6; }\n"
            "    if (header.abi_version != 1U || header.leaf_count != GENERATED_LEAF_COUNT) { fclose(fp); return -7; }\n"
            "    for (order_index = 0U; order_index < GENERATED_LEAF_COUNT; ++order_index) {\n"
            "        size_t leaf_index = g_topology_order[order_index];\n"
            "        GeneratedLeafHeader leaf_header;\n"
            "        char id_buffer[128];\n"
            "        char type_buffer[128];\n"
            "        const NNGraphInferContract* contract;\n"
            "        void* native_ctx = validate_only ? 0 : infer_get_leaf_native_context(infer_ctx, leaf_index);\n"
            "        if (fread(&leaf_header, sizeof(leaf_header), 1, fp) != 1) { fclose(fp); return -8; }\n"
            "        if (leaf_header.id_length >= sizeof(id_buffer) || leaf_header.type_length >= sizeof(type_buffer)) { fclose(fp); return -9; }\n"
            "        if (fread(id_buffer, 1, leaf_header.id_length, fp) != leaf_header.id_length ||\n"
            "            fread(type_buffer, 1, leaf_header.type_length, fp) != leaf_header.type_length) { fclose(fp); return -10; }\n"
            "        id_buffer[leaf_header.id_length] = '\\0';\n"
            "        type_buffer[leaf_header.type_length] = '\\0';\n"
            "        if (strcmp(id_buffer, g_expected_leaf_ids[leaf_index]) != 0 ||\n"
            "            strcmp(type_buffer, g_expected_leaf_types[leaf_index]) != 0) { fclose(fp); return -11; }\n"
            "        contract = nn_graph_infer_contract_find(type_buffer);\n"
            "        if (contract == 0 || contract->load_weights == 0) { fclose(fp); return -12; }\n"
            "        if (!validate_only && native_ctx == 0) { fclose(fp); return -13; }\n"
            "        if (contract->load_weights(native_ctx, fp) == 0) { fclose(fp); return -14; }\n"
            "    }\n"
            "    fclose(fp);\n"
            "    return 0;\n"
            "}\n\n"
            "int weights_load_from_file(void* infer_ctx, const char* file_path) {\n"
            "    return generated_weights_load_internal(infer_ctx, file_path, 0);\n"
            "}\n\n"
            "int weights_load_validate(const char* file_path) {\n"
            "    return generated_weights_load_internal(0, file_path, 1);\n"
            "}\n",
            (unsigned long long)ctx->network_hash,
            (unsigned long long)ctx->layout_hash) != 0) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, PROF_STATUS_INTERNAL_ERROR,
            "Failed to assemble weights_load.c");
    }

    st = prof_path_ensure_parent_directory(layout->weights_load.c_path);
    if (st != PROF_STATUS_OK) {
        free(content);
        prof_leaf_graph_cleanup(&graph);
        return prof_error_set(ctx->error, st,
            "Failed to create directory for weights_load.c");
    }

    st = write_file(layout->weights_load.c_path, content);
    free(content);
    prof_leaf_graph_cleanup(&graph);
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, st,
            "Failed to write weights_load.c");
    }

    st = write_optional_header(
        layout->weights_load.h_path,
        "/* weights_load.h - Weights loading interface */\n"
        "#ifndef WEIGHTS_LOAD_H\n"
        "#define WEIGHTS_LOAD_H\n\n"
        "int weights_load_from_file(void* infer_ctx, const char* file_path);\n"
        "int weights_load_validate(const char* file_path);\n\n"
        "#endif\n"
    );
    if (st != PROF_STATUS_OK) {
        return prof_error_set(ctx->error, st,
            "Failed to write weights_load.h");
    }

    return PROF_STATUS_OK;
}

/**
 * @brief Run every module emitter in the documented generation order.
 *
 * Ordering is intentional: metadata and structural headers must exist before the
 * runtime modules that include them, and save/load must observe the same graph
 * facts already embedded into infer/train generation. This function is the
 * concrete implementation of the documented first-fail generation pipeline.
 */
ProfStatus prof_codegen_generate_all(ProfCodegenContext* ctx) {
    ProfStatus st;

    if (ctx == NULL || ctx->network == NULL || ctx->output_layout == NULL) {
        return PROF_STATUS_INVALID_ARGUMENT;
    }

    /* Generation order matters because later modules depend on earlier headers. */
    /* Each call returns immediately on failure to preserve fast-fail diagnostics. */
    st = prof_codegen_metadata(ctx, ctx->output_layout->metadata_path);
    if (st != PROF_STATUS_OK) return st;
    st = prof_codegen_tokenizer(ctx);
    if (st != PROF_STATUS_OK) return st;
    st = prof_codegen_network_init(ctx);
    if (st != PROF_STATUS_OK) return st;
    st = prof_codegen_infer(ctx);
    if (st != PROF_STATUS_OK) return st;
    st = prof_codegen_train(ctx);
    if (st != PROF_STATUS_OK) return st;
    st = prof_codegen_weights_save(ctx);
    if (st != PROF_STATUS_OK) return st;
    st = prof_codegen_weights_load(ctx);
    if (st != PROF_STATUS_OK) return st;
    return PROF_STATUS_OK;
}
