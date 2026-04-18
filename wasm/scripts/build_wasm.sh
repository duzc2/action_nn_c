#!/bin/bash
# ============================================================================
# action_nn_c WebAssembly Build Script (Unix/Linux/macOS)
# ============================================================================
# 
# This script builds the action_nn_c library for WebAssembly.
# It requires the Emscripten SDK to be installed and activated.
#
# Usage:
#   ./build_wasm.sh [options]
#
# Options:
#   --enable-cnn          Enable CNN network type
#   --enable-rnn          Enable RNN network type
#   --enable-gnn          Enable GNN network type
#   --enable-training     Include training support
#   --debug               Build with debug symbols and no optimization
#   --clean               Clean build directory before building
#   --help                Show this help message
#
# Environment Variables:
#   EMSDK_PATH            Path to Emscripten SDK (default: ./emsdk)
#   WASM_MEMORY_MB        Initial memory size in MB (default: 64)
#   WASM_OPT_LEVEL        Optimization level 0-3 (default: 2)
# ============================================================================

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default options
ENABLE_CNN=OFF
ENABLE_RNN=OFF
ENABLE_GNN=OFF
ENABLE_TRAINING=OFF
BUILD_TYPE="Release"
CLEAN_BUILD=FALSE
EMSDK_PATH="${EMSDK_PATH:-./emsdk}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-cnn)
            ENABLE_CNN=ON
            shift
            ;;
        --enable-rnn)
            ENABLE_RNN=ON
            shift
            ;;
        --enable-gnn)
            ENABLE_GNN=ON
            shift
            ;;
        --enable-training)
            ENABLE_TRAINING=ON
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=TRUE
            shift
            ;;
        --help)
            head -30 "$0" | tail -25 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Check for Emscripten
if ! command -v emcc &> /dev/null; then
    echo "Warning: Emscripten (emcc) not found in PATH."
    echo ""
    echo "Please install and activate Emscripten SDK:"
    echo "  1. Clone: git clone https://github.com/emscripten-core/emsdk.git"
    echo "  2. Install: cd emsdk && ./emsdk install latest"
    echo "  3. Activate: ./emsdk activate latest"
    echo "  4. Source: source ./emsdk_env.sh"
    echo ""
    echo "Or set EMSDK_PATH and run:"
    echo "  source \$EMSDK_PATH/emsdk_env.sh"
    echo ""
    
    # Try to find emsdk in common locations
    if [[ -f "$SCRIPT_DIR/emsdk/emsdk_env.sh" ]]; then
        echo "Found emsdk in $SCRIPT_DIR/emsdk, activating..."
        source "$SCRIPT_DIR/emsdk/emsdk_env.sh"
    elif [[ -f "$HOME/emsdk/emsdk_env.sh" ]]; then
        echo "Found emsdk in $HOME/emsdk, activating..."
        source "$HOME/emsdk/emsdk_env.sh"
    else
        echo "Emscripten not found. Please install it first."
        exit 1
    fi
fi

# Verify emcc is now available
if ! command -v emcc &> /dev/null; then
    echo "Error: Failed to activate Emscripten."
    exit 1
fi

echo "Using Emscripten: $(emcc --version | head -1)"

# Build directory
BUILD_DIR="$SCRIPT_DIR/build"

# Clean if requested
if [[ "$CLEAN_BUILD" == "TRUE" ]]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"

# Build configuration
CMAKE_ARGS=(
    "-DCMAKE_TOOLCHAIN_FILE=\$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DACTION_C_WASM_ENABLE_MLP=ON"
    "-DACTION_C_WASM_ENABLE_TRANSFORMER=ON"
    "-DACTION_C_WASM_ENABLE_CNN=$ENABLE_CNN"
    "-DACTION_C_WASM_ENABLE_CNN_DUAL_POOL=OFF"
    "-DACTION_C_WASM_ENABLE_RNN=$ENABLE_RNN"
    "-DACTION_C_WASM_ENABLE_GNN=$ENABLE_GNN"
    "-DACTION_C_WASM_ENABLE_TRAINING=$ENABLE_TRAINING"
)

# Add memory configuration from environment
if [[ -n "$WASM_MEMORY_MB" ]]; then
    CMAKE_ARGS+=("-DACTION_C_WASM_MEMORY_MB=$WASM_MEMORY_MB")
fi

if [[ -n "$WASM_OPT_LEVEL" ]]; then
    CMAKE_ARGS+=("-DACTION_C_WASM_OPT_LEVEL=$WASM_OPT_LEVEL")
fi

# Configure
echo ""
echo "=== Configuring WebAssembly Build ==="
echo "Build Type: $BUILD_TYPE"
echo "Enable CNN: $ENABLE_CNN"
echo "Enable RNN: $ENABLE_RNN"
echo "Enable GNN: $ENABLE_GNN"
echo "Enable Training: $ENABLE_TRAINING"
echo ""

cd "$BUILD_DIR"
emcmake cmake .. ${CMAKE_ARGS[@]}

# Build
echo ""
echo "=== Building WebAssembly Module ==="
emmake cmake --build . --config "$BUILD_TYPE"

# Verify output
if [[ -f "$SCRIPT_DIR/dist/action_nn_c.js" ]] && [[ -f "$SCRIPT_DIR/dist/action_nn_c.wasm" ]]; then
    echo ""
    echo "=== Build Successful ==="
    echo "Output files:"
    echo "  - $SCRIPT_DIR/dist/action_nn_c.js"
    echo "  - $SCRIPT_DIR/dist/action_nn_c.wasm"
    echo ""
    
    # Show file sizes
    JS_SIZE=$(ls -lh "$SCRIPT_DIR/dist/action_nn_c.js" | awk '{print $5}')
    WASM_SIZE=$(ls -lh "$SCRIPT_DIR/dist/action_nn_c.wasm" | awk '{print $5}')
    echo "File sizes:"
    echo "  - action_nn_c.js: $JS_SIZE"
    echo "  - action_nn_c.wasm: $WASM_SIZE"
    echo ""
    echo "To use in a browser, include the JS file:"
    echo '  <script src="action_nn_c.js"></script>'
    echo ""
else
    echo ""
    echo "Error: Build completed but output files not found."
    exit 1
fi
