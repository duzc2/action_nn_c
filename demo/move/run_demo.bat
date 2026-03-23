@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ACTION_C_ROOT=%%~fI"
set "BUILD_ROOT=%ACTION_C_ROOT%\build\demo\move"

taskkill /F /IM move_generate.exe >nul 2>&1
taskkill /F /IM move_train.exe >nul 2>&1
taskkill /F /IM move_infer.exe >nul 2>&1

echo [move] step 1/6 configure + build generate
cmake -S "%SCRIPT_DIR%generate" -B "%BUILD_ROOT%\generate"
if errorlevel 1 goto :fail
cmake --build "%BUILD_ROOT%\generate" --config Debug
if errorlevel 1 goto :fail

echo [move] step 2/6 run generate
set "RUN_COMMAND=""%BUILD_ROOT%\generate\Debug\move_generate.exe"""
set "RUN_TIMEOUT=30"
call :run_with_timeout
if errorlevel 1 goto :fail

echo [move] step 3/6 configure + build train
cmake -S "%SCRIPT_DIR%train" -B "%BUILD_ROOT%\train"
if errorlevel 1 goto :fail
cmake --build "%BUILD_ROOT%\train" --config Debug
if errorlevel 1 goto :fail

echo [move] step 4/6 run train
set "RUN_COMMAND=""%BUILD_ROOT%\train\Debug\move_train.exe"""
set "RUN_TIMEOUT=30"
call :run_with_timeout
if errorlevel 1 goto :fail

echo [move] step 5/6 configure + build infer
cmake -S "%SCRIPT_DIR%infer" -B "%BUILD_ROOT%\infer"
if errorlevel 1 goto :fail
cmake --build "%BUILD_ROOT%\infer" --config Debug
if errorlevel 1 goto :fail

echo [move] step 6/6 run infer
set "RUN_COMMAND=(echo 5 5 ^& echo 0 ^& echo 0 ^& echo 0 ^& echo 4) ^| ""%BUILD_ROOT%\infer\Debug\move_infer.exe"""
set "RUN_TIMEOUT=30"
call :run_with_timeout
if errorlevel 1 goto :fail

echo [move] demo completed successfully
exit /b 0

:fail
echo [move] demo failed
exit /b 1

:run_with_timeout
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$cmd = $env:RUN_COMMAND; $timeout = [int]$env:RUN_TIMEOUT; " ^
  "$p = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $cmd -NoNewWindow -PassThru; " ^
  "try { Wait-Process -Id $p.Id -Timeout $timeout -ErrorAction Stop; exit $p.ExitCode } " ^
  "catch { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Write-Host '[move] process timeout'; exit 124 }"
set "STEP_EXIT=%ERRORLEVEL%"
exit /b %STEP_EXIT%
