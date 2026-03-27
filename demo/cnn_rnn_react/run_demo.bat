@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ACTION_C_ROOT=%%~fI"
set "BUILD_ROOT=%ACTION_C_ROOT%\build\demo\cnn_rnn_react"
set "VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

taskkill /F /IM cnn_rnn_react_generate.exe >nul 2>&1
taskkill /F /IM cnn_rnn_react_train.exe >nul 2>&1
taskkill /F /IM cnn_rnn_react_infer.exe >nul 2>&1

echo [cnn_rnn_react] prepare clang x64 toolchain
call "%VCVARS64%"
if errorlevel 1 goto :fail

echo [cnn_rnn_react] step 1/9 configure generate
cmake -S "%SCRIPT_DIR%generate" -B "%BUILD_ROOT%\generate" -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Debug
if errorlevel 1 goto :fail

echo [cnn_rnn_react] step 2/9 build generate
cmake --build "%BUILD_ROOT%\generate"
if errorlevel 1 goto :fail

echo [cnn_rnn_react] step 3/9 run generate
call :resolve_exe_path "%BUILD_ROOT%\generate" "cnn_rnn_react_generate.exe"
if errorlevel 1 goto :fail
set "RUN_COMMAND=""%RESOLVED_EXE%"""
set "RUN_TIMEOUT=30"
call :run_with_timeout
if errorlevel 1 goto :fail

echo [cnn_rnn_react] step 4/9 configure train
cmake -S "%SCRIPT_DIR%train" -B "%BUILD_ROOT%\train" -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Debug
if errorlevel 1 goto :fail

echo [cnn_rnn_react] step 5/9 build train
cmake --build "%BUILD_ROOT%\train"
if errorlevel 1 goto :fail

echo [cnn_rnn_react] step 6/9 run train
call :resolve_exe_path "%BUILD_ROOT%\train" "cnn_rnn_react_train.exe"
if errorlevel 1 goto :fail
set "RUN_COMMAND=""%RESOLVED_EXE%"""
set "RUN_TIMEOUT=60"
call :run_with_timeout
if errorlevel 1 goto :fail

echo [cnn_rnn_react] step 7/9 configure infer
cmake -S "%SCRIPT_DIR%infer" -B "%BUILD_ROOT%\infer" -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_BUILD_TYPE=Debug
if errorlevel 1 goto :fail

echo [cnn_rnn_react] step 8/9 build infer
cmake --build "%BUILD_ROOT%\infer"
if errorlevel 1 goto :fail

echo [cnn_rnn_react] step 9/9 run infer
call :resolve_exe_path "%BUILD_ROOT%\infer" "cnn_rnn_react_infer.exe"
if errorlevel 1 goto :fail
set "RUN_COMMAND=""%RESOLVED_EXE%"""
set "RUN_TIMEOUT=30"
call :run_with_timeout
if errorlevel 1 goto :fail

echo [cnn_rnn_react] demo completed successfully
exit /b 0

:fail
echo [cnn_rnn_react] demo failed
exit /b 1

:resolve_exe_path
set "RESOLVED_EXE="
set "EXE_BASE=%~1"
set "EXE_NAME=%~2"

if exist "%EXE_BASE%\Debug\%EXE_NAME%" (
    set "RESOLVED_EXE=%EXE_BASE%\Debug\%EXE_NAME%"
    exit /b 0
)

if exist "%EXE_BASE%\Release\%EXE_NAME%" (
    set "RESOLVED_EXE=%EXE_BASE%\Release\%EXE_NAME%"
    exit /b 0
)

if exist "%EXE_BASE%\%EXE_NAME%" (
    set "RESOLVED_EXE=%EXE_BASE%\%EXE_NAME%"
    exit /b 0
)

echo [cnn_rnn_react] executable not found: %EXE_BASE%\%EXE_NAME%
exit /b 1

:run_with_timeout
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$cmd = $env:RUN_COMMAND; $timeout = [int]$env:RUN_TIMEOUT; " ^
  "$p = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $cmd -NoNewWindow -PassThru; " ^
  "try { Wait-Process -Id $p.Id -Timeout $timeout -ErrorAction Stop; exit $p.ExitCode } " ^
  "catch { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Write-Host '[cnn_rnn_react] process timeout'; exit 124 }"
set "STEP_EXIT=%ERRORLEVEL%"
exit /b %STEP_EXIT%
