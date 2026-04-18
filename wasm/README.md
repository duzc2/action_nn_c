# action_nn_c WebAssembly Build

This directory contains the complete WebAssembly (Wasm) build system for action_nn_c, enabling the neural network library to run in web browsers and other Wasm runtimes.

## Quick Start

### Prerequisites

Users must have Emscripten SDK installed in their deployment environment:

```bash
# Install Emscripten SDK
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

### Build

```bash
cd wasm
./scripts/build_wasm.sh
```

This produces:
- `dist/action_nn_c.js` - JavaScript glue code
- `dist/action_nn_c.wasm` - WebAssembly binary

### Usage in Browser

```html
<script src="dist/action_nn_c.js"></script>
<script>
  ActionNnC().then(function(Module) {
    const version = Module.action_c_wasm_get_version_string();
    console.log('Version:', version);
  });
</script>
```

See `examples/browser_demo.html` for a complete working example.

## Directory Structure

```
wasm/
├── CMakeLists.txt          # Main CMake configuration
├── wasm.md                 # Detailed documentation
├── config/
│   └── wasm_config.h       # Wasm-specific configuration
├── src/
│   └── wasm_exports.c      # Export wrapper functions
├── js/
│   ├── action_nn_c.js      # JS placeholder (generated during build)
│   └── action_nn_c.d.ts    # TypeScript definitions
├── examples/
│   └── browser_demo.html   # Browser demo page
├── scripts/
│   ├── build_wasm.sh       # Unix/Linux/macOS build script
│   └── build_wasm.ps1      # Windows PowerShell build script
└── dist/                   # Build output (created after build)
    ├── action_nn_c.js
    └── action_nn_c.wasm
```

## Configuration Options

All options are configurable via CMake variables or build script flags:

| Option | Default | Description |
|--------|---------|-------------|
| `ACTION_C_WASM_ENABLE_MLP` | ON | Enable MLP network |
| `ACTION_C_WASM_ENABLE_TRANSFORMER` | ON | Enable Transformer network |
| `ACTION_C_WASM_ENABLE_CNN` | OFF | Enable CNN network |
| `ACTION_C_WASM_ENABLE_RNN` | OFF | Enable RNN network |
| `ACTION_C_WASM_ENABLE_GNN` | OFF | Enable GNN network |
| `ACTION_C_WASM_ENABLE_TRAINING` | OFF | Include training support |
| `ACTION_C_WASM_MEMORY_MB` | 64 | Initial memory (MB) |
| `ACTION_C_WASM_OPT_LEVEL` | 2 | Optimization (0-3) |

### Build Script Examples

```bash
# Basic build (MLP + Transformer only)
./scripts/build_wasm.sh

# Enable additional networks
./scripts/build_wasm.sh --enable-cnn --enable-rnn

# Debug build with clean
./scripts/build_wasm.sh --debug --clean

# Custom memory size
WASM_MEMORY_MB=128 ./scripts/build_wasm.sh
```

## Architecture

The Wasm build follows the same architecture as the native build:

1. **Core Libraries**: Re-used from `../src/` (nn, infer, train, profiler)
2. **Export Layer**: `src/wasm_exports.c` provides C functions exported to JS
3. **Configuration**: `config/wasm_config.h` controls Wasm-specific settings
4. **Build System**: CMake + Emscripten toolchain

### Exported Functions

#### Version Info
- `action_c_wasm_get_version_string()` - Version string
- `action_c_wasm_get_version_major/minor/patch()` - Version components

#### Memory Management
- `action_c_wasm_malloc(size)` - Allocate memory
- `action_c_wasm_free(ptr)` - Free memory
- `action_c_wasm_get_memory_size()` - Total memory size
- `action_c_wasm_get_heap_size()` - Heap size

#### Inference
- `action_c_wasm_infer_create(config, size)` - Create context
- `action_c_wasm_infer_run(ctx, input, inSize, output, outSize)` - Run inference
- `action_c_wasm_infer_destroy(ctx)` - Destroy context

#### Training (if enabled)
- `action_c_wasm_train_create(config, size)` - Create training context
- `action_c_wasm_train_step(ctx, input, inSize, target, tSize, lr)` - Training step
- `action_c_wasm_train_get_loss(ctx)` - Get current loss
- `action_c_wasm_train_destroy(ctx)` - Destroy context

#### Network Queries
- `action_c_wasm_get_network_type_count()` - Number of supported networks
- `action_c_wasm_get_network_type_name(index)` - Network name by index
- `action_c_wasm_is_network_type_enabled(name)` - Check if enabled

## Server-Side Deployment

The build is designed for server-side compilation:

1. **Build on Server**: Compile Wasm module on your deployment server
2. **Deploy Artifacts**: Copy `dist/*.js` and `dist/*.wasm` to CDN/web server
3. **Client Usage**: Users load the pre-built module in their browsers

This approach:
- ✅ Reduces client load time
- ✅ Ensures consistent builds
- ✅ Allows server-side optimization
- ✅ Keeps build tools off client machines

## Development Environment

For local development, the build scripts automatically:
1. Check for Emscripten in PATH
2. Search common installation locations
3. Auto-activate if found
4. Provide clear error messages if missing

## Performance Considerations

1. **Memory**: Set appropriate `ACTION_C_WASM_MEMORY_MB` for your models
2. **Optimization**: Use `-O3` for production, `-O0` for debugging
3. **SIMD**: Enabled by default for modern browsers
4. **Bulk Memory**: Enabled for faster memory operations

## Troubleshooting

### "Emscripten not found"
Install and activate Emscripten SDK as shown in Prerequisites.

### "Memory limit exceeded"
Increase `ACTION_C_WASM_MEMORY_MB` or reduce model size.

### "Function not exported"
Check that the function is marked with `WASM_EXPORT` in `wasm_exports.c`.

### Module loading fails in browser
Ensure both `.js` and `.wasm` files are served from the same origin with correct MIME types.

## License

Same as the main project (see LICENSE in root directory).
