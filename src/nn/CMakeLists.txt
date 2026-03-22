file(GLOB_RECURSE NN_INFER_ENTRY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/types/*/nn_type_*_infer.c"
)
file(GLOB_RECURSE NN_TRAIN_ENTRY_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/types/*/nn_type_*_train.c"
)
file(GLOB_RECURSE NN_INFER_OP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/types/*/*_infer_ops.c"
)
file(GLOB_RECURSE NN_CODE_GEN_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/types/*/*_code_gen.c"
)
file(GLOB_RECURSE NN_TRAIN_OP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/types/*/*_train_ops.c"
)

set(NN_INFER_AUTOGEN_C "${CMAKE_CURRENT_BINARY_DIR}/nn_infer_registry_autogen.c")
set(NN_INFER_AUTOGEN_CONTENT "#include \"nn_infer_registry.h\"\n\n")
foreach(NN_TYPE_SOURCE ${NN_INFER_ENTRY_SOURCES})
    get_filename_component(NN_TYPE_SYMBOL_BASENAME "${NN_TYPE_SOURCE}" NAME_WE)
    string(APPEND NN_INFER_AUTOGEN_CONTENT "extern const NNInferRegistryEntry ${NN_TYPE_SYMBOL_BASENAME}_entry;\n")
endforeach()
string(APPEND NN_INFER_AUTOGEN_CONTENT "\nconst NNInferRegistryEntry* const* nn_infer_registry_builtin_entries(size_t* out_count) {\n")
string(APPEND NN_INFER_AUTOGEN_CONTENT "    static const NNInferRegistryEntry* const entries[] = {\n")
foreach(NN_TYPE_SOURCE ${NN_INFER_ENTRY_SOURCES})
    get_filename_component(NN_TYPE_SYMBOL_BASENAME "${NN_TYPE_SOURCE}" NAME_WE)
    string(APPEND NN_INFER_AUTOGEN_CONTENT "        &${NN_TYPE_SYMBOL_BASENAME}_entry,\n")
endforeach()
string(APPEND NN_INFER_AUTOGEN_CONTENT "    };\n")
string(APPEND NN_INFER_AUTOGEN_CONTENT "    if (out_count != 0) {\n")
string(APPEND NN_INFER_AUTOGEN_CONTENT "        *out_count = sizeof(entries) / sizeof(entries[0]);\n")
string(APPEND NN_INFER_AUTOGEN_CONTENT "    }\n")
string(APPEND NN_INFER_AUTOGEN_CONTENT "    return entries;\n")
string(APPEND NN_INFER_AUTOGEN_CONTENT "}\n")
file(WRITE "${NN_INFER_AUTOGEN_C}" "${NN_INFER_AUTOGEN_CONTENT}")

set(NN_TRAIN_AUTOGEN_C "${CMAKE_CURRENT_BINARY_DIR}/nn_train_registry_autogen.c")
set(NN_TRAIN_AUTOGEN_CONTENT "#include \"nn_train_registry.h\"\n\n")
foreach(NN_TYPE_SOURCE ${NN_TRAIN_ENTRY_SOURCES})
    get_filename_component(NN_TYPE_SYMBOL_BASENAME "${NN_TYPE_SOURCE}" NAME_WE)
    string(APPEND NN_TRAIN_AUTOGEN_CONTENT "extern const NNTrainRegistryEntry ${NN_TYPE_SYMBOL_BASENAME}_entry;\n")
endforeach()
string(APPEND NN_TRAIN_AUTOGEN_CONTENT "\nconst NNTrainRegistryEntry* const* nn_train_registry_builtin_entries(size_t* out_count) {\n")
string(APPEND NN_TRAIN_AUTOGEN_CONTENT "    static const NNTrainRegistryEntry* const entries[] = {\n")
foreach(NN_TYPE_SOURCE ${NN_TRAIN_ENTRY_SOURCES})
    get_filename_component(NN_TYPE_SYMBOL_BASENAME "${NN_TYPE_SOURCE}" NAME_WE)
    string(APPEND NN_TRAIN_AUTOGEN_CONTENT "        &${NN_TYPE_SYMBOL_BASENAME}_entry,\n")
endforeach()
string(APPEND NN_TRAIN_AUTOGEN_CONTENT "    };\n")
string(APPEND NN_TRAIN_AUTOGEN_CONTENT "    if (out_count != 0) {\n")
string(APPEND NN_TRAIN_AUTOGEN_CONTENT "        *out_count = sizeof(entries) / sizeof(entries[0]);\n")
string(APPEND NN_TRAIN_AUTOGEN_CONTENT "    }\n")
string(APPEND NN_TRAIN_AUTOGEN_CONTENT "    return entries;\n")
string(APPEND NN_TRAIN_AUTOGEN_CONTENT "}\n")
file(WRITE "${NN_TRAIN_AUTOGEN_C}" "${NN_TRAIN_AUTOGEN_CONTENT}")

set(NN_CODE_GEN_AUTOGEN_C "${CMAKE_CURRENT_BINARY_DIR}/profiler_code_gen_autogen.c")
set(NN_CODE_GEN_AUTOGEN_CONTENT "#include \"profiler_code_gen.h\"\n\n")
foreach(NN_TYPE_SOURCE ${NN_CODE_GEN_SOURCES})
    get_filename_component(NN_TYPE_SYMBOL_BASENAME "${NN_TYPE_SOURCE}" NAME_WE)
    string(REPLACE "_code_gen" "" NN_TYPE_NAME "${NN_TYPE_SYMBOL_BASENAME}")
    string(APPEND NN_CODE_GEN_AUTOGEN_CONTENT "extern void profiler_code_gen_init_${NN_TYPE_NAME}(void);\n")
endforeach()
string(APPEND NN_CODE_GEN_AUTOGEN_CONTENT "\nint profiler_code_gen_bootstrap(void) {\n")
foreach(NN_TYPE_SOURCE ${NN_CODE_GEN_SOURCES})
    get_filename_component(NN_TYPE_SYMBOL_BASENAME "${NN_TYPE_SOURCE}" NAME_WE)
    string(REPLACE "_code_gen" "" NN_TYPE_NAME "${NN_TYPE_SYMBOL_BASENAME}")
    string(APPEND NN_CODE_GEN_AUTOGEN_CONTENT "    profiler_code_gen_init_${NN_TYPE_NAME}();\n")
endforeach()
string(APPEND NN_CODE_GEN_AUTOGEN_CONTENT "    return 0;\n")
string(APPEND NN_CODE_GEN_AUTOGEN_CONTENT "}\n")
file(WRITE "${NN_CODE_GEN_AUTOGEN_C}" "${NN_CODE_GEN_AUTOGEN_CONTENT}")

add_library(nn_infer_core STATIC
    nn_infer_registry.c
    ${NN_INFER_AUTOGEN_C}
    ${NN_INFER_OP_SOURCES}
    ${NN_CODE_GEN_SOURCES}
    ${NN_CODE_GEN_AUTOGEN_C}
    ${NN_INFER_ENTRY_SOURCES}
)
target_include_directories(nn_infer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/profiler
)

add_library(nn_train_core STATIC
    nn_train_registry.c
    ${NN_TRAIN_AUTOGEN_C}
    ${NN_TRAIN_OP_SOURCES}
    ${NN_TRAIN_ENTRY_SOURCES}
)
target_include_directories(nn_train_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(nn_train_core PUBLIC
    nn_infer_core
)
