# Build script for all demos
# Usage: .\build_demos.ps1

$ErrorActionPreference = "Stop"

$RootDir = "C:\Users\ASUS\Desktop\ai-build-ai\action_c"
$BuildDir = "$RootDir\build"
$DemoRoot = "$RootDir\demo"
$Demos = @("move", "sevenseg", "target", "transformer")

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Building All Demos" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Build each demo
foreach ($demo in $Demos) {
    Write-Host "`n--- Building $demo ---" -ForegroundColor Yellow

    # Create build directories
    $demoBuildDir = "$BuildDir\demo\$demo"
    @("generate", "train", "infer") | ForEach-Object {
        $dir = "$demoBuildDir\$_"
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }

    # Step 1: Configure and build generator
    Write-Host "  [1/3] Configuring generator..." -ForegroundColor Gray
    $genDir = "$demoBuildDir\generate"
    Push-Location $genDir
    Remove-Item CMakeCache.txt, CMakeFiles -Recurse -Force -ErrorAction SilentlyContinue
    cmake -S "$DemoRoot\$demo\generate" -B . 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    Generator configure failed!" -ForegroundColor Red
        Pop-Location
        continue
    }
    Pop-Location

    Write-Host "  [1/3] Building generator..." -ForegroundColor Gray
    cmake --build $genDir --config Debug 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    Generator build failed!" -ForegroundColor Red
        continue
    }

    Write-Host "  [1/3] Running generator..." -ForegroundColor Gray
    & "$genDir\Debug\${demo}_generate.exe" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    Generator run failed!" -ForegroundColor Red
        continue
    }

    Write-Host "  $demo generator: OK" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Build Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nGenerated files are in:" -ForegroundColor White
foreach ($demo in $Demos) {
    Write-Host "  $BuildDir\demo\$demo\data\" -ForegroundColor Gray
}
