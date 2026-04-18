# ============================================================================
# action_nn_c WebAssembly Build Script (Windows PowerShell)
# ============================================================================
# 
# This script builds the action_nn_c library for WebAssembly.
# It requires the Emscripten SDK to be installed and activated.
#
# Usage:
#   .\build_wasm.ps1 [-EnableCNN] [-EnableRNN] [-EnableGNN] [-EnableTraining] 
#                    [-Debug] [-Clean] [-Help]
#
# Examples:
#   .\build_wasm.ps1                           # Basic build with MLP + Transformer
#   .\build_wasm.ps1 -EnableCNN                # Include CNN support
#   .\build_wasm.ps1 -EnableCNN -EnableRNN     # Include CNN and RNN
#   .\build_wasm.ps1 -Debug -Clean             # Debug build, clean first
#
# Environment Variables:
#   $env:WASM_MEMORY_MB    Initial memory size in MB (default: 64)
#   $env:WASM_OPT_LEVEL    Optimization level 0-3 (default: 2)
# ============================================================================

[CmdletBinding()]
param(
    [switch]$EnableCNN,
    [switch]$EnableRNN,
    [switch]$EnableGNN,
    [switch]$EnableTraining,
    [switch]$Debug,
    [switch]$Clean,
    [switch]$Help
)

# Show help if requested
if ($Help) {
    Get-Help $PSCommandPath -Detailed
    exit 0
}

# Script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Default options
$EnableCnnVal = if ($EnableCNN) { "ON" } else { "OFF" }
$EnableRnnVal = if ($EnableRNN) { "ON" } else { "OFF" }
$EnableGnnVal = if ($EnableGNN) { "ON" } else { "OFF" }
$EnableTrainingVal = if ($EnableTraining) { "ON" } else { "OFF" }
$BuildType = if ($Debug) { "Debug" } else { "Release" }

Write-Host ""
Write-Host "=== action_nn_c WebAssembly Build (PowerShell) ===" -ForegroundColor Cyan
Write-Host ""

# Check for Emscripten
$EmccPath = Get-Command emcc -ErrorAction SilentlyContinue
if (-not $EmccPath) {
    Write-Host "Warning: Emscripten (emcc) not found in PATH." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please install and activate Emscripten SDK:"
    Write-Host "  1. Clone: git clone https://github.com/emscripten-core/emsdk.git"
    Write-Host "  2. Install: cd emsdk && .\emsdk install latest"
    Write-Host "  3. Activate: .\emsdk activate latest"
    Write-Host "  4. Import: .\emsdk_env.ps1"
    Write-Host ""
    
    # Try to find emsdk in common locations
    $EmsdkPaths = @(
        "$ScriptDir\emsdk",
        "$HOME\emsdk",
        "$env:USERPROFILE\emsdk"
    )
    
    $FoundEmsdk = $false
    foreach ($Path in $EmsdkPaths) {
        if (Test-Path "$Path\emsdk_env.ps1") {
            Write-Host "Found emsdk in $Path, activating..." -ForegroundColor Green
            & "$Path\emsdk_env.ps1"
            $FoundEmsdk = $true
            break
        }
    }
    
    if (-not $FoundEmsdk) {
        Write-Host "Error: Emscripten not found. Please install it first." -ForegroundColor Red
        exit 1
    }
}

# Re-check after potential activation
$EmccPath = Get-Command emcc -ErrorAction SilentlyContinue
if (-not $EmccPath) {
    Write-Host "Error: Failed to activate Emscripten." -ForegroundColor Red
    exit 1
}

$EmccVersion = & emcc --version 2>&1 | Select-Object -First 1
Write-Host "Using Emscripten: $EmccVersion" -ForegroundColor Green
Write-Host ""

# Build directory
$BuildDir = "$ScriptDir\build"

# Clean if requested
if ($Clean) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    if (Test-Path $BuildDir) {
        Remove-Item -Recurse -Force $BuildDir
    }
}

# Create build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Build configuration
$CMakeArgs = @(
    "-DCMAKE_TOOLCHAIN_FILE=$env:EMSDK\upstream\emscripten\cmake\Modules\Platform\Emscripten.cmake"
    "-DCMAKE_BUILD_TYPE=$BuildType"
    "-DACTION_C_WASM_ENABLE_MLP=ON"
    "-DACTION_C_WASM_ENABLE_TRANSFORMER=ON"
    "-DACTION_C_WASM_ENABLE_CNN=$EnableCnnVal"
    "-DACTION_C_WASM_ENABLE_CNN_DUAL_POOL=OFF"
    "-DACTION_C_WASM_ENABLE_RNN=$EnableRnnVal"
    "-DACTION_C_WASM_ENABLE_GNN=$EnableGnnVal"
    "-DACTION_C_WASM_ENABLE_TRAINING=$EnableTrainingVal"
)

# Add memory configuration from environment
if ($env:WASM_MEMORY_MB) {
    $CMakeArgs += "-DACTION_C_WASM_MEMORY_MB=$env:WASM_MEMORY_MB"
}

if ($env:WASM_OPT_LEVEL) {
    $CMakeArgs += "-DACTION_C_WASM_OPT_LEVEL=$env:WASM_OPT_LEVEL"
}

# Configure
Write-Host "=== Configuring WebAssembly Build ===" -ForegroundColor Cyan
Write-Host "Build Type: $BuildType"
Write-Host "Enable CNN: $EnableCnnVal"
Write-Host "Enable RNN: $EnableRnnVal"
Write-Host "Enable GNN: $EnableGnnVal"
Write-Host "Enable Training: $EnableTrainingVal"
Write-Host ""

Push-Location $BuildDir
try {
    emcmake cmake .. @CMakeArgs
    
    # Build
    Write-Host ""
    Write-Host "=== Building WebAssembly Module ===" -ForegroundColor Cyan
    emmake cmake --build . --config $BuildType
    
    # Verify output
    $DistDir = "$ScriptDir\dist"
    $JsFile = "$DistDir\action_nn_c.js"
    $WasmFile = "$DistDir\action_nn_c.wasm"
    
    if ((Test-Path $JsFile) -and (Test-Path $WasmFile)) {
        Write-Host ""
        Write-Host "=== Build Successful ===" -ForegroundColor Green
        Write-Host "Output files:"
        Write-Host "  - $JsFile"
        Write-Host "  - $WasmFile"
        Write-Host ""
        
        # Show file sizes
        $JsSize = (Get-Item $JsFile).Length
        $WasmSize = (Get-Item $WasmFile).Length
        Write-Host "File sizes:"
        Write-Host "  - action_nn_c.js: $([math]::Round($JsSize / 1KB, 2)) KB"
        Write-Host "  - action_nn_c.wasm: $([math]::Round($WasmSize / 1KB, 2)) KB"
        Write-Host ""
        Write-Host "To use in a browser, include the JS file:"
        Write-Host '  <script src="action_nn_c.js"></script>'
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "Error: Build completed but output files not found." -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}
