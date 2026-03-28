@echo off
setlocal
goto :main

:run_with_timeout
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$cmd = $env:RUN_COMMAND; $timeout = [int]$env:RUN_TIMEOUT; " ^
  "$p = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c', $cmd -NoNewWindow -PassThru; " ^
  "try { Wait-Process -Id $p.Id -Timeout $timeout -ErrorAction Stop; exit $p.ExitCode } " ^
  "catch { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Write-Host '[road_graph_nav] process timeout'; exit 124 }"
set "STEP_EXIT=%ERRORLEVEL%"
exit /b %STEP_EXIT%

:fail
echo [road_graph_nav] demo failed
exit /b 1

:main
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ACTION_C_ROOT=%%~fI"
set "BUILD_ROOT=%ACTION_C_ROOT%\build\demo\road_graph_nav"
set "VCVARS_BAT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "GENERATE_TIMEOUT=60"
set "TRAIN_TIMEOUT=300"
set "INFER_TIMEOUT=180"

if not exist "%VCVARS_BAT%" (
  echo [road_graph_nav] missing vcvars64.bat: %VCVARS_BAT%
  goto :fail
)

call "%VCVARS_BAT%" >nul
if errorlevel 1 goto :fail

set "CMAKE_GENERATOR=Ninja"
set "CMAKE_COMMON=-G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang-cl"

taskkill /F /IM road_graph_nav_generate.exe >nul 2>&1
taskkill /F /IM road_graph_nav_train.exe >nul 2>&1
taskkill /F /IM road_graph_nav_infer.exe >nul 2>&1

echo [road_graph_nav] step 1/6 configure + build generate
cmake -S "%SCRIPT_DIR%generate" -B "%BUILD_ROOT%\generate" %CMAKE_COMMON%
if errorlevel 1 goto :fail
cmake --build "%BUILD_ROOT%\generate"
if errorlevel 1 goto :fail

echo [road_graph_nav] step 2/6 run generate
set "RUN_COMMAND=""%BUILD_ROOT%\generate\road_graph_nav_generate.exe"""
set "RUN_TIMEOUT=%GENERATE_TIMEOUT%"
call :run_with_timeout
if errorlevel 1 goto :fail

echo [road_graph_nav] step 3/6 configure + build train
cmake -S "%SCRIPT_DIR%train" -B "%BUILD_ROOT%\train" %CMAKE_COMMON%
if errorlevel 1 goto :fail
cmake --build "%BUILD_ROOT%\train"
if errorlevel 1 goto :fail

echo [road_graph_nav] step 4/6 run train
set "RUN_COMMAND=""%BUILD_ROOT%\train\road_graph_nav_train.exe"""
set "RUN_TIMEOUT=%TRAIN_TIMEOUT%"
call :run_with_timeout
if errorlevel 1 goto :fail

echo [road_graph_nav] step 5/6 configure + build infer
cmake -S "%SCRIPT_DIR%infer" -B "%BUILD_ROOT%\infer" %CMAKE_COMMON%
if errorlevel 1 goto :fail
cmake --build "%BUILD_ROOT%\infer"
if errorlevel 1 goto :fail

echo [road_graph_nav] step 6/6 run infer
set "RUN_COMMAND=""%BUILD_ROOT%\infer\road_graph_nav_infer.exe"""
set "RUN_TIMEOUT=%INFER_TIMEOUT%"
call :run_with_timeout
if errorlevel 1 goto :fail

echo [road_graph_nav] demo completed successfully
exit /b 0
