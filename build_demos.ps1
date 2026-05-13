$ErrorActionPreference = "Stop"

$demos = @("move", "sevenseg", "target", "transformer", "mnist", "mnist_cnn", "nested_nav", "road_graph_nav", "cnn_rnn_react", "hybrid_route")
$failed = @()

Write-Host "=== Building all demos (generate phase) ===" -ForegroundColor Cyan

foreach ($demo in $demos) {
    Write-Host "`n[$demo]" -ForegroundColor Cyan

    Write-Host "  Configuring..." -NoNewline
    cmake -B "build/demo/$demo/generate" -S "demo/$demo/generate" -DCMAKE_BUILD_TYPE=Debug
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAIL (configure)" -ForegroundColor Red
        $failed += "$demo (configure)"
        continue
    }
    Write-Host " OK" -ForegroundColor Green

    Write-Host "  Building..." -NoNewline
    cmake --build "build/demo/$demo/generate" --config Debug
    if ($LASTEXITCODE -ne 0) {
        Write-Host " FAIL (build)" -ForegroundColor Red
        $failed += "$demo (build)"
        continue
    }
    Write-Host " OK" -ForegroundColor Green
}

Write-Host "`n=== Build Summary ===" -ForegroundColor Cyan
if ($failed.Count -eq 0) {
    Write-Host "All demos built successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Failed demos:" -ForegroundColor Red
    foreach ($f in $failed) {
        Write-Host "  - $f" -ForegroundColor Red
    }
    exit 1
}
