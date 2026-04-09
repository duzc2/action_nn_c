set(NN_ENABLED_TYPES "")
set(NN_INFER_ENTRY_SOURCES "")
set(NN_TRAIN_ENTRY_SOURCES "")
set(NN_INFER_OP_SOURCES "")
set(NN_TRAIN_OP_SOURCES "")
set(NN_LAYER_SOURCES "")

function(action_c_append_type_sources TYPE_NAME)
    file(GLOB _infer_entry "${CMAKE_CURRENT_SOURCE_DIR}/types/${TYPE_NAME}/nn_type_*_infer.c")
    file(GLOB _train_entry "${CMAKE_CURRENT_SOURCE_DIR}/types/${TYPE_NAME}/nn_type_*_train.c")
    file(GLOB _infer_ops "${CMAKE_CURRENT_SOURCE_DIR}/types/${TYPE_NAME}/*_infer_ops.c")
    file(GLOB _train_ops "${CMAKE_CURRENT_SOURCE_DIR}/types/${TYPE_NAME}/*_train_ops.c")
    file(GLOB _layers "${CMAKE_CURRENT_SOURCE_DIR}/types/${TYPE_NAME}/*_layers.c")

    list(APPEND NN_INFER_ENTRY_SOURCES ${_infer_entry})
    list(APPEND NN_TRAIN_ENTRY_SOURCES ${_train_entry})
    list(APPEND NN_INFER_OP_SOURCES ${_infer_ops})
    list(APPEND NN_TRAIN_OP_SOURCES ${_train_ops})
    list(APPEND NN_LAYER_SOURCES ${_layers})

    set(NN_INFER_ENTRY_SOURCES "${NN_INFER_ENTRY_SOURCES}" PARENT_SCOPE)
    set(NN_TRAIN_ENTRY_SOURCES "${NN_TRAIN_ENTRY_SOURCES}" PARENT_SCOPE)
    set(NN_INFER_OP_SOURCES "${NN_INFER_OP_SOURCES}" PARENT_SCOPE)
    set(NN_TRAIN_OP_SOURCES "${NN_TRAIN_OP_SOURCES}" PARENT_SCOPE)
    set(NN_LAYER_SOURCES "${NN_LAYER_SOURCES}" PARENT_SCOPE)
endfunction()

if(ACTION_C_ENABLE_NN_MLP)
    list(APPEND NN_ENABLED_TYPES mlp)
    action_c_append_type_sources(mlp)
endif()

if(ACTION_C_ENABLE_NN_TRANSFORMER)
    list(APPEND NN_ENABLED_TYPES transformer)
    action_c_append_type_sources(transformer)
endif()

if(ACTION_C_ENABLE_NN_CNN)
    list(APPEND NN_ENABLED_TYPES cnn)
    action_c_append_type_sources(cnn)
endif()

if(ACTION_C_ENABLE_NN_CNN_DUAL_POOL)
    list(APPEND NN_ENABLED_TYPES cnn_dual_pool)
    action_c_append_type_sources(cnn_dual_pool)
endif()

if(ACTION_C_ENABLE_NN_RNN)
    list(APPEND NN_ENABLED_TYPES rnn)
    action_c_append_type_sources(rnn)
endif()

if(ACTION_C_ENABLE_NN_GNN)
    list(APPEND NN_ENABLED_TYPES gnn)
    action_c_append_type_sources(gnn)
endif()

if(NOT NN_ENABLED_TYPES)
    message(FATAL_ERROR "No network types are enabled for nn core")
endif()

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

add_library(nn_infer_core STATIC
    nn_graph_infer_contract.c
    nn_infer_registry.c
    ${NN_INFER_AUTOGEN_C}
    ${NN_INFER_OP_SOURCES}
    ${NN_INFER_ENTRY_SOURCES}
    ${NN_LAYER_SOURCES}
)
action_c_apply_strict_warnings(nn_infer_core)
target_include_directories(nn_infer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/profiler
)

add_library(nn_train_core STATIC
    nn_graph_train_contract.c
    nn_train_registry.c
    ${NN_TRAIN_AUTOGEN_C}
    ${NN_TRAIN_OP_SOURCES}
    ${NN_TRAIN_ENTRY_SOURCES}
)
action_c_apply_strict_warnings(nn_train_core)
target_include_directories(nn_train_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(nn_train_core PUBLIC
    nn_infer_core
)
