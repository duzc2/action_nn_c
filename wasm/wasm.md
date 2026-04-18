# action_nn_c WebAssembly Build System

This directory contains the WebAssembly (Wasm) build configuration for action_nn_c.

## Overview

The Wasm build allows action_nn_c to run in web browsers and other WebAssembly runtimes.
The build process is designed to be performed on the server side during deployment.

## Prerequisites

Users must ensure the following tools are available in their deployment environment:

1. **Emscripten SDK** (version 3.1.0 or later)
   - Install from: https://emscripten.org/docs/getting_started/downloads.html
   - Or use: `git clone https://github.com/emscripten-core/emsdk.git && cd emsdk && ./emsdk install latest && ./emsdk activate latest`

2. **CMake** (version 3.20 or later)

3. **A C99-compatible compiler** (for native builds)

## Directory Structure

```
wasm/
├── CMakeLists.txt          # Main CMake configuration for Wasm build
├── wasm.md                 # This documentation file
├── config/
│   └── wasm_config.h       # Wasm-specific configuration header
├── js/
│   ├── action_nn_c.js      # JavaScript binding layer
│   └── action_nn_c.d.ts    # TypeScript definitions
├── examples/
│   └── browser_demo.html   # Basic browser integration example
└── scripts/
    ├── build_wasm.sh       # Unix/Linux build script
    └── build_wasm.ps1      # Windows PowerShell build script
```

## Quick Start

### Server-Side Build (Recommended)

```bash
# On your deployment server with Emscripten installed
cd wasm
./scripts/build_wasm.sh
```

This will produce:
- `dist/action_nn_c.wasm` - The WebAssembly module
- `dist/action_nn_c.js` - JavaScript glue code
- `dist/action_nn_c.d.ts` - TypeScript definitions

### Development Environment Setup

For local development and testing:

```bash
# Install Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh

# Build
cd /path/to/action_nn_c/wasm
./scripts/build_wasm.sh
```

## Configuration

Edit `config/wasm_config.h` to customize:
- Memory limits
- Exported functions
- Feature flags

## Usage in Browser

```html
<script src="action_nn_c.js"></script>
<script>
  ActionNnC().then(function(Module) {
    // Use Module.infer_create(), Module.infer_run(), etc.
  });
</script>
```

See `examples/browser_demo.html` for a complete example.

## API Reference

The Wasm build exports the following core functions:

### Inference Functions
- `infer_create(config_ptr, config_size)` - Create inference context
- `infer_run(ctx_ptr, input_ptr, input_size, output_ptr, output_size)` - Run inference
- `infer_destroy(ctx_ptr)` - Destroy inference context

### Training Functions (if enabled)
- `train_create(config_ptr, config_size)` - Create training context
- `train_step(ctx_ptr, input_ptr, target_ptr, lr)` - Perform one training step
- `train_destroy(ctx_ptr)` - Destroy training context

### Utility Functions
- `get_version()` - Get library version string
- `get_supported_networks()` - Get list of supported network types

## Build Options

Set these CMake variables to customize the build:

| Variable | Default | Description |
|----------|---------|-------------|
| `ACTION_C_WASM_ENABLE_MLP` | ON | Enable MLP network |
| `ACTION_C_WASM_ENABLE_TRANSFORMER` | ON | Enable Transformer network |
| `ACTION_C_WASM_ENABLE_CNN` | OFF | Enable CNN network |
| `ACTION_C_WASM_ENABLE_RNN` | OFF | Enable RNN network |
| `ACTION_C_WASM_ENABLE_GNN` | OFF | Enable GNN network |
| `ACTION_C_WASM_ENABLE_TRAINING` | OFF | Include training support |
| `ACTION_C_WASM_MEMORY_MB` | 64 | Initial memory size in MB |
| `ACTION_C_WASM_OPT_LEVEL` | 2 | Optimization level (0-3) |

Example:
```bash
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
  -DACTION_C_WASM_ENABLE_CNN=ON \
  -DACTION_C_WASM_MEMORY_MB=128 \
  -S .
cmake --build build
```

## Performance Considerations

1. **Memory**: Wasm memory is linear and pre-allocated. Set `ACTION_C_WASM_MEMORY_MB` appropriately.
2. **Optimization**: Use `-O3` for production builds, `-O0` for debugging.
3. **SIMD**: Enable SIMD support for better performance on modern browsers.
4. **Threading**: Web Workers can be used for non-blocking inference (future enhancement).

## Troubleshooting

### Common Issues

1. **"Memory limit exceeded"**
   - Increase `ACTION_C_WASM_MEMORY_MB`
   - Reduce model size or batch size

2. **"Function not exported"**
   - Check `wasm_config.h` for EXPORT_MACRO definitions
   - Verify function is declared with `EMSCRIPTEN_KEEPALIVE`

3. **"Module not found"**
   - Ensure both `.wasm` and `.js` files are deployed together
   - Check CORS headers if loading from different origin

## License

Same as the main project (see LICENSE in root directory).
